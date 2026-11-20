"""
Polite, cached page fetcher.

- Respects robots.txt, one request per host per `per_host_delay` seconds (or the site's Crawl-delay, capped).
- Never retries 4xx (a 403 means "no"); one retry on timeouts / 5xx.
- TLS verification stays on.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from brain.cache import Cache
from brain.settings import Settings

log = logging.getLogger("brain.fetch")

MAX_BYTES = 2_000_000
MAX_PDF_BYTES = 15_000_000
MAX_CRAWL_DELAY = 15.0


def _pdf_text(data: bytes, max_pages: int) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))[:200_000]
    except Exception:
        return None
PAGE_TTL = 7 * 86400
SKIP_EXT = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".zip", ".jpg", ".jpeg", ".png", ".gif", ".mp4")


@dataclass
class Page:
    url: str
    status: int
    html: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.html)


class Fetcher:
    def __init__(self, settings: Optional[Settings] = None, cache: Optional[Cache] = None,
                 client: Optional[httpx.Client] = None):
        self.settings = settings or Settings()
        self.cache = cache or Cache()
        self.client = client or httpx.Client(
            timeout=15, follow_redirects=True,
            headers={"User-Agent": self.settings.user_agent, "Accept": "text/html,application/xhtml+xml"},
        )
        self._robots: Dict[str, Optional[RobotFileParser]] = {}
        self._host_locks: Dict[str, threading.Lock] = {}
        self._host_last: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.fetched = 0  # network fetches this run (cache hits excluded)
        self.paid_calls: Dict[str, int] = {}  # calls to paid APIs this run, per provider (cache hits excluded)
        self.replay_misses = 0

    def _host_lock(self, host: str) -> threading.Lock:
        with self._lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        with self._lock:
            known = base in self._robots
            rp = self._robots.get(base)
        if not known:
            rp = None
            try:
                r = self.client.get(base + "/robots.txt", timeout=8)
                if r.status_code == 200 and "text" in r.headers.get("content-type", "text"):
                    rp = RobotFileParser()
                    rp.parse(r.text.splitlines())
            except httpx.HTTPError:
                rp = None
            with self._lock:
                self._robots[base] = rp
        return True if rp is None else rp.can_fetch(self.settings.user_agent, url)

    def _wait_turn(self, url: str) -> None:
        """Space requests to one host by the per-host delay, or the site's robots.txt Crawl-delay if longer
        (capped at MAX_CRAWL_DELAY). Call with the host lock held."""
        parts = urlsplit(url)
        delay = self.settings.per_host_delay
        rp = self._robots.get(f"{parts.scheme}://{parts.netloc}")
        if rp is not None:
            try:
                cd = rp.crawl_delay(self.settings.user_agent)
            except Exception:
                cd = None
            if cd:
                delay = max(delay, min(float(cd), MAX_CRAWL_DELAY))
        wait = delay - (time.time() - self._host_last.get(parts.netloc, 0))
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str) -> Page:
        if not url.startswith(("http://", "https://")) or urlsplit(url).path.lower().endswith(SKIP_EXT):
            return Page(url, 0, error="unsupported url")
        cached = self.cache.get("page", url)
        if cached is not None:
            return Page(url, cached["status"], cached.get("html", ""), cached.get("error", ""))
        if self.settings.replay:
            self.replay_misses += 1
            return Page(url, 0, error="replay: not recorded")
        if self.fetched >= self.settings.max_page_fetches:
            return Page(url, 0, error="page fetch budget exhausted")

        host = urlsplit(url).netloc
        with self._host_lock(host):
            try:
                if not self._allowed(url):
                    page = Page(url, 0, error="disallowed by robots.txt")
                else:
                    self._wait_turn(url)
                    page = self._fetch(url)
            finally:
                self._host_last[host] = time.time()
        # Cache failures briefly so one bad host does not stall re-runs.
        self.cache.set("page", url, {"status": page.status, "html": page.html, "error": page.error},
                       ttl=PAGE_TTL if page.ok else 3600)
        return page

    def _fetch(self, url: str) -> Page:
        self.fetched += 1
        err = ""
        for attempt in range(2):
            try:
                r = self.client.get(url)
            except httpx.HTTPError as e:
                err = type(e).__name__
                continue
            if r.status_code >= 500 and attempt == 0:
                err = f"HTTP {r.status_code}"
                time.sleep(1.5)
                continue
            ctype = r.headers.get("content-type", "")
            if r.status_code != 200:
                return Page(url, r.status_code, error=f"HTTP {r.status_code}")
            if "html" not in ctype and "text" not in ctype and "xml" not in ctype:  # xml: RSS feeds
                return Page(url, r.status_code, error=f"not html ({ctype})")
            return Page(url, 200, r.text[:MAX_BYTES])
        log.debug("fetch failed %s: %s", url, err)
        return Page(url, 0, error=err or "fetch failed")

    def get_json(self, url: str, params: Optional[Dict] = None, ttl: float = 3 * 86400,
                 headers: Optional[Dict[str, str]] = None) -> Optional[object]:
        """A public JSON API (job boards, iTunes directory). Cached; None on any failure. Not robots-gated:
        these are documented APIs meant for programmatic use, not pages. `headers` (e.g. an API key) are
        sent but are not part of the cache key."""
        key = Cache.key(url, params or {})
        cached = self.cache.get("api", key)
        if cached is not None:
            return cached.get("data")
        if self.settings.replay:
            self.replay_misses += 1
            return None
        if self.fetched >= self.settings.max_page_fetches:
            return None
        self.fetched += 1
        try:
            r = self.client.get(url, params=params, headers={"Accept": "application/json", **(headers or {})})
            data = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            data = None
        self.cache.set("api", key, {"data": data}, ttl=ttl if data is not None else 3600)
        return data

    def post_json(self, url: str, body: Dict, headers: Optional[Dict[str, str]] = None,
                  ttl: float = 3 * 86400) -> Optional[object]:
        """A JSON API that takes a POST body (e.g. Exa search). Cached by url + body; headers (API keys) are not part
        of the key. Counts as a paid call when the caller says so (see paid_calls)."""
        key = Cache.key(url, body)
        cached = self.cache.get("api_post", key)
        if cached is not None:
            return cached.get("data")
        if self.settings.replay:
            self.replay_misses += 1
            return None
        self.fetched += 1
        try:
            r = self.client.post(url, json=body, headers={"Accept": "application/json", **(headers or {})})
            data = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            data = None
        self.cache.set("api_post", key, {"data": data}, ttl=ttl if data is not None else 3600)
        return data

    def get_api_text(self, url: str, params: Optional[Dict] = None, ttl: float = 3 * 86400) -> Optional[str]:
        """A documented API that answers in text (e.g. arXiv's Atom XML). Cached like get_json; not robots-gated
        (an API meant for programs, used within its stated rate limit by the source's policy)."""
        key = Cache.key(url, params or {})
        cached = self.cache.get("api_text", key)
        if cached is not None:
            return cached.get("text")
        if self.settings.replay:
            self.replay_misses += 1
            return None
        if self.fetched >= self.settings.max_page_fetches:
            return None
        self.fetched += 1
        try:
            r = self.client.get(url, params=params)
            text = r.text[:MAX_BYTES] if r.status_code == 200 else None
        except httpx.HTTPError:
            text = None
        self.cache.set("api_text", key, {"text": text}, ttl=ttl if text is not None else 3600)
        return text

    def get_pdf_text(self, url: str, max_pages: int = 2) -> Optional[str]:
        """Text of a PDF's first pages (where author emails are printed). robots.txt and Crawl-delay are honoured;
        the extracted text (not the PDF) is cached, so a replay needs no download."""
        key = f"{url}#p{max_pages}"
        cached = self.cache.get("pdf", key)
        if cached is not None:
            return cached.get("text")
        if self.settings.replay:
            self.replay_misses += 1
            return None
        if self.fetched >= self.settings.max_page_fetches or not url.startswith(("http://", "https://")):
            return None
        host = urlsplit(url).netloc
        text: Optional[str] = None
        with self._host_lock(host):
            try:
                if self._allowed(url):
                    self._wait_turn(url)
                    self.fetched += 1
                    r = self.client.get(url, headers={"Accept": "application/pdf"})
                    data = r.content[:MAX_PDF_BYTES] if r.status_code == 200 else b""
                    if data.startswith(b"%PDF"):
                        text = _pdf_text(data, max_pages)
            except httpx.HTTPError:
                text = None
            finally:
                self._host_last[host] = time.time()
        self.cache.set("pdf", key, {"text": text}, ttl=30 * 86400 if text else 3600)
        return text

    def get_many(self, urls: Iterable[str]) -> List[Page]:
        urls = list(dict.fromkeys(urls))
        if not urls:
            return []
        with ThreadPoolExecutor(max_workers=max(1, self.settings.fetch_workers)) as pool:
            return list(pool.map(self.get, urls))

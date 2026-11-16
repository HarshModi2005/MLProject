"""
Polite, cached page fetcher.

- Respects robots.txt, one request per host per `per_host_delay` seconds.
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

    def get(self, url: str) -> Page:
        if not url.startswith(("http://", "https://")) or urlsplit(url).path.lower().endswith(SKIP_EXT):
            return Page(url, 0, error="unsupported url")
        cached = self.cache.get("page", url)
        if cached is not None:
            return Page(url, cached["status"], cached.get("html", ""), cached.get("error", ""))
        if self.fetched >= self.settings.max_page_fetches:
            return Page(url, 0, error="page fetch budget exhausted")

        host = urlsplit(url).netloc
        with self._host_lock(host):
            wait = self.settings.per_host_delay - (time.time() - self._host_last.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            try:
                if not self._allowed(url):
                    page = Page(url, 0, error="disallowed by robots.txt")
                else:
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
            if "html" not in ctype and "text" not in ctype:
                return Page(url, r.status_code, error=f"not html ({ctype})")
            return Page(url, 200, r.text[:MAX_BYTES])
        log.debug("fetch failed %s: %s", url, err)
        return Page(url, 0, error=err or "fetch failed")

    def get_many(self, urls: Iterable[str]) -> List[Page]:
        urls = list(dict.fromkeys(urls))
        if not urls:
            return []
        with ThreadPoolExecutor(max_workers=max(1, self.settings.fetch_workers)) as pool:
            return list(pool.map(self.get, urls))

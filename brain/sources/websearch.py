"""Web search providers (Serper, Tavily) behind one interface, with caching and a per-campaign budget.

Three Serper endpoints are exposed, each a separate tool for the agent:
    search  ordinary web results
    places  Google Maps businesses — name, address, phone, website (best for local businesses)
    news    recent articles with their outlet (best for finding journalists)
Every call counts against one budget; cached answers are free.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from brain.cache import Cache
from brain.settings import Settings

log = logging.getLogger("brain.search")

SKIP_HOSTS = (
    "wikipedia.org", "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "reddit.com", "quora.com", "glassdoor.com", "indeed.com", "researchgate.net",
    "scholar.google", "semanticscholar.org", "academia.edu", "amazon.", "pinterest.", "medium.com",
    "arxiv.org", "ieee.org", "acm.org", "springer.com", "sciencedirect.com", "ratemyprofessors.com",
    "scribd.com", "slideshare.net", "issuu.com", "coursehero.com", "figma.com", "github.com", "gitlab.com",
    "investopedia.com", "britannica.com", "dictionary.", "tiktok.com", "threads.net",
)


@dataclass
class Hit:
    url: str
    title: str = ""
    snippet: str = ""


@dataclass
class Place:
    name: str
    website: str = ""
    address: str = ""
    phone: str = ""
    category: str = ""
    rating: float = 0.0
    reviews: int = 0


@dataclass
class NewsHit:
    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""
    date: str = ""


class WebSearch:
    def __init__(self, settings: Optional[Settings] = None, cache: Optional[Cache] = None,
                 client: Optional[httpx.Client] = None):
        self.settings = settings or Settings()
        self.cache = cache or Cache()
        self.client = client or httpx.Client(timeout=20)
        self.calls = 0
        self.budget = self.settings.max_search_calls  # the pipeline lowers this per campaign
        self.by_kind = {"search": 0, "places": 0, "news": 0}
        self.replay_misses = 0
        self.failures = 0  # calls the provider rejected or that errored (out of credits, bad key, rate limit)
        self.last_failure = ""
        self.other_failures: Dict[str, tuple] = {}  # provider -> (failed, of, last message), reported by other sources
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.calls)

    def _spend(self, kind: str, query: str) -> bool:
        with self._lock:  # tools enrich in threads; the budget must hold under concurrency
            if self.settings.replay:
                self.replay_misses += 1
                log.info("replay: no recorded answer for %s %r; skipped", kind, query)
                return False
            if self.calls >= self.budget:
                log.warning("search budget (%d) exhausted; skipping %s %r", self.budget, kind, query)
                return False
            self.calls += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
            return True

    def _note_failure(self, kind: str, query: str, e: Exception) -> None:
        """Count a failed call and keep the provider's own words ("Not enough credits"), so the run can tell the user
        why results are thin instead of advising them to broaden a search that never ran."""
        detail = ""
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                detail = str(resp.json().get("message") or resp.text)[:120]
            except (ValueError, AttributeError):
                detail = (getattr(resp, "text", "") or "")[:120]
            detail = f"HTTP {resp.status_code}: {detail}"
        with self._lock:
            self.failures += 1
            self.last_failure = detail or type(e).__name__
        log.warning("%s search failed for %r: %s", kind, query, detail or e)

    def note_provider_failure(self, provider: str, message: str, failed: int, of: int) -> None:
        with self._lock:
            old = self.other_failures.get(provider, (0, 0, ""))
            self.other_failures[provider] = (old[0] + failed, old[1] + of, message or old[2])

    def provider_problem(self, min_failures: int = 3) -> str:
        """A sentence for the user when a real share of a provider's calls failed; empty when everything behaved."""
        parts = []
        if self.failures >= min_failures and self.failures >= 0.2 * max(1, self.calls):
            parts.append(f"The search provider failed {self.failures} of {self.calls} searches ({self.last_failure}).")
        for name, (failed, of, msg) in self.other_failures.items():
            if failed >= 2 and failed >= 0.2 * max(1, of):
                parts.append(f"{name} failed {failed} of {of} requests ({msg}).")
        if not parts:
            return ""
        return " ".join(parts) + " Results are incomplete because of that, not because nobody matches: fix the account or key and run again."

    @property
    def available(self) -> bool:
        return bool(self.settings.serper_api_key or self.settings.tavily_api_key)

    def search(self, query: str, n: int = 8, raw: bool = False) -> List[Hit]:
        """Web results. Login-walled and reference sites are dropped unless `raw` (used only to read result
        TITLES, e.g. "Name - Role - Company", never to fetch those pages)."""
        if not self.available:
            return []
        key = Cache.key("raw", query, n) if raw else Cache.key(query, n)
        cached = self.cache.get("search", key)
        if cached is not None:
            return [Hit(**h) for h in cached]
        if not self._spend("search", query):
            return []
        try:
            hits = self._serper(query, n) if self.settings.serper_api_key else self._tavily(query, n)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self._note_failure("web", query, e)
            return []
        hits = [h for h in hits if h.url and (raw or not any(s in h.url.lower() for s in SKIP_HOSTS))]
        self.cache.set("search", key, [h.__dict__ for h in hits], ttl=7 * 86400)
        return hits

    def places(self, query: str, gl: str = "") -> List[Place]:
        """Businesses on Google Maps matching the query (Serper only)."""
        if not self.settings.serper_api_key:
            return []
        key = Cache.key("places", query, gl)
        cached = self.cache.get("search", key)
        if cached is not None:
            return [Place(**p) for p in cached]
        if not self._spend("places", query):
            return []
        body = {"q": query}
        if gl:
            body["gl"] = gl.lower()
        try:
            r = self.client.post("https://google.serper.dev/places", json=body,
                                 headers={"X-API-KEY": self.settings.serper_api_key})
            r.raise_for_status()
            raw = r.json().get("places", [])
        except (httpx.HTTPError, ValueError) as e:
            self._note_failure("places", query, e)
            return []
        out = [Place(name=p.get("title", ""), website=p.get("website", "") or "", address=p.get("address", "") or "",
                     phone=p.get("phoneNumber", "") or "", category=p.get("category", "") or "",
                     rating=float(p.get("rating") or 0), reviews=int(p.get("ratingCount") or 0))
               for p in raw if p.get("title")]
        self.cache.set("search", key, [p.__dict__ for p in out], ttl=14 * 86400)
        return out

    def news(self, query: str, n: int = 10, gl: str = "") -> List[NewsHit]:
        """Recent news articles matching the query (Serper only)."""
        if not self.settings.serper_api_key:
            return []
        key = Cache.key("news", query, n, gl)
        cached = self.cache.get("search", key)
        if cached is not None:
            return [NewsHit(**h) for h in cached]
        if not self._spend("news", query):
            return []
        body = {"q": query, "num": n}
        if gl:
            body["gl"] = gl.lower()
        try:
            r = self.client.post("https://google.serper.dev/news", json=body,
                                 headers={"X-API-KEY": self.settings.serper_api_key})
            r.raise_for_status()
            raw = r.json().get("news", [])
        except (httpx.HTTPError, ValueError) as e:
            self._note_failure("news", query, e)
            return []
        out = [NewsHit(url=h.get("link", ""), title=h.get("title", ""), snippet=h.get("snippet", ""),
                       source=h.get("source", ""), date=h.get("date", "")) for h in raw if h.get("link")]
        self.cache.set("search", key, [h.__dict__ for h in out], ttl=2 * 86400)
        return out

    def _serper(self, query: str, n: int) -> List[Hit]:
        r = self.client.post("https://google.serper.dev/search", json={"q": query, "num": n},
                             headers={"X-API-KEY": self.settings.serper_api_key})
        r.raise_for_status()
        return [Hit(o.get("link", ""), o.get("title", ""), o.get("snippet", "")) for o in r.json().get("organic", [])]

    def _tavily(self, query: str, n: int) -> List[Hit]:
        r = self.client.post("https://api.tavily.com/search", json={
            "api_key": self.settings.tavily_api_key, "query": query, "max_results": n, "search_depth": "basic"})
        r.raise_for_status()
        return [Hit(o.get("url", ""), o.get("title", ""), o.get("content", "")) for o in r.json().get("results", [])]

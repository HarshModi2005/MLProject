"""ORCID public API: a researcher's own listed websites and (rarely) public email. Free, no key."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import httpx

from brain.cache import Cache
from brain.settings import Settings

log = logging.getLogger("brain.orcid")


class Orcid:
    def __init__(self, settings: Optional[Settings] = None, cache: Optional[Cache] = None,
                 client: Optional[httpx.Client] = None):
        self.settings = settings or Settings()
        self.cache = cache or Cache()
        self.client = client or httpx.Client(timeout=15, headers={
            "Accept": "application/json", "User-Agent": self.settings.user_agent})

    def person(self, orcid: str) -> Tuple[List[str], List[str]]:
        """Returns (emails, urls) the researcher has made public."""
        hit = self.cache.get("orcid", orcid)
        if hit is not None:
            return hit["emails"], hit["urls"]
        emails: List[str] = []
        urls: List[str] = []
        try:
            r = self.client.get(f"https://pub.orcid.org/v3.0/{orcid}/person")
            if r.status_code == 200:
                data = r.json()
                for e in ((data.get("emails") or {}).get("email") or []):
                    if e.get("email"):
                        emails.append(e["email"].lower())
                for u in ((data.get("researcher-urls") or {}).get("researcher-url") or []):
                    value = (u.get("url") or {}).get("value")
                    if value:
                        urls.append(value)
        except (httpx.HTTPError, ValueError) as e:
            log.debug("ORCID lookup failed for %s: %s", orcid, e)
            return [], []
        self.cache.set("orcid", orcid, {"emails": emails, "urls": urls}, ttl=30 * 86400)
        return emails, urls

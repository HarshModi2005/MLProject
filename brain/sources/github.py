"""
GitHub users as a source of people (recruiting, developer relations, open-source outreach).

Uses the public REST API: user search by skills and location, then each user's profile for name,
company, bio and the email *they chose to make public*. Free; set GITHUB_TOKEN for higher limits.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

import httpx

from brain.cache import Cache
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence
from brain.settings import Settings
from brain.textutil import registrable_domain, squash, to_country_code

log = logging.getLogger("brain.github")
API = "https://api.github.com"
_ROLE_WORDS = {"developer", "developers", "engineer", "engineers", "programmer", "programmers", "contributor",
               "contributors", "maintainer", "maintainers", "practitioner", "practitioners", "expert", "experts",
               "specialist", "specialists", "in", "with", "and", "for", "of", "the", "senior", "junior", "lead",
               "software", "open", "source", "bangalore", "bengaluru", "india"}
_LANGS = {"python", "javascript", "typescript", "go", "rust", "java", "kotlin", "swift", "c++", "c", "ruby", "php",
          "scala", "dart", "elixir", "haskell", "julia", "r", "c#", "solidity", "zig"}


class GitHubSource:
    name = "github"

    def __init__(self, settings: Optional[Settings] = None, cache: Optional[Cache] = None,
                 client: Optional[httpx.Client] = None):
        self.settings = settings or Settings()
        self.cache = cache or Cache()
        headers = {"Accept": "application/vnd.github+json", "User-Agent": self.settings.user_agent}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = client or httpx.Client(timeout=20, headers=headers)
        self.requests = 0
        self.failures = 0
        self.last_failure = ""

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        key = Cache.key(path, params)
        hit = self.cache.get("github", key)
        if hit is not None:
            return hit
        self.requests += 1
        try:
            r = self.client.get(API + path, params=params)
        except httpx.HTTPError as e:
            log.warning("GitHub request failed: %s", e)
            self.failures += 1
            self.last_failure = type(e).__name__
            return None
        if r.status_code != 200:
            log.warning("GitHub %s -> HTTP %s (rate limit? set GITHUB_TOKEN)", path, r.status_code)
            self.failures += 1
            self.last_failure = (f"HTTP {r.status_code}, rate limited: set GITHUB_TOKEN" if r.status_code in (403, 429)
                                 else f"HTTP {r.status_code}")
            return None
        data = r.json()
        self.cache.set("github", key, data, ttl=7 * 86400)
        return data

    def search(self, spec: CampaignSpec, max_users: int = 30) -> List[Contact]:
        # GitHub's user search matches free text against names, so skills must become qualifiers:
        # languages are pulled out of the topics ("Python developers" -> language:python), and role words dropped.
        terms = [t.lower() for t in spec.topics + spec.expanded_terms]
        tokens = [w for t in terms for w in re.findall(r"[a-z+#]+", t)]
        langs = list(dict.fromkeys(w for w in tokens if w in _LANGS))[:2]
        words = []
        for t in terms:
            phrase = " ".join(w for w in re.findall(r"[a-z+#]+", t) if w not in _LANGS and w not in _ROLE_WORDS)
            if phrase and len(phrase) > 2 and phrase not in words:
                words.append(phrase)
        words = words[:1]
        place = (spec.regions or [""])[0]
        qualifiers = [f'language:"{l}"' for l in langs]
        if place:
            qualifiers.append(f'location:"{place}"')
        qualifiers.append("followers:>=20")
        queries = [" ".join(qualifiers)] + [" ".join([f'"{w}"'] + qualifiers) for w in words]
        logins: List[str] = []
        for q in queries:
            data = self._get("/search/users", {"q": q, "per_page": 30, "sort": "followers"})
            for item in (data or {}).get("items", []):
                if item.get("login") and item["login"] not in logins and item.get("type") == "User":
                    logins.append(item["login"])
        out: List[Contact] = []
        for login in logins[:max_users]:
            u = self._get(f"/users/{login}")
            if not u or not u.get("name"):
                continue
            c = Contact(kind="person", name=squash(u["name"], 80), title=squash(u.get("bio") or "", 120) or None,
                        org=(u.get("company") or "").lstrip("@").strip() or None, location_text=u.get("location"),
                        sources=["github"])
            c.add_key(f"github:{login}")
            c.add_url(u.get("html_url"))
            if u.get("blog"):
                blog = u["blog"] if u["blog"].startswith("http") else "https://" + u["blog"]
                c.add_url(blog)
            if u.get("location"):
                parts = [p.strip() for p in u["location"].split(",")]
                c.country = next((to_country_code(p) for p in reversed(parts) if to_country_code(p)), None)
            if u.get("email"):  # published by the user on their own profile
                c.add_email(EmailCandidate(address=u["email"].lower(), confidence=0.9, method="profile",
                                           source_url=u.get("html_url")))
                dom = registrable_domain(u["email"].split("@")[1])
                c.org_domain = c.org_domain or dom
            c.evidence.append(Evidence(kind="profile", url=u.get("html_url"),
                                       text=f"GitHub: {u.get('public_repos', 0)} public repos, {u.get('followers', 0)} followers"
                                            + (f"; bio: {squash(u['bio'], 160)}" if u.get("bio") else "")))
            c.signals = {"followers": float(u.get("followers") or 0), "matched_works": float(u.get("public_repos") or 0)}
            out.append(c)
        return out

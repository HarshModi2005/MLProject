"""crates.io: owners of popular Rust crates on the topic, with what their public GitHub profile says about reaching them.

crates.io lists a crate's owners but no addresses. Each owner's GitHub profile (name, the email they chose to make public,
their own website) fills that in; the website is then read by person enrichment for a contact page.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from brain.discovery.normalize import add_published_email, campaign_terms
from brain.models import Contact, Evidence
from brain.textutil import squash

API = "https://crates.io/api/v1"
_BOTS = {"rust-lang", "github-actions", "dependabot", "rustbot", "bors", "rust-lang-owner"}


def _terms(run) -> List[str]:
    return campaign_terms(run.spec, {"terms": ["topics", "expanded_terms:1"], "max": 3})


def search(run) -> Tuple[List[Contact], Dict[str, Any]]:
    fetcher = run.deps.fetcher
    seen: Dict[str, Contact] = {}
    stats = {"crates": 0, "owners": 0, "profiles": 0}
    for term in _terms(run):
        data = fetcher.get_json(f"{API}/crates", params={"q": term, "per_page": 12, "sort": "downloads"}) or {}
        for crate in (data.get("crates") or [])[:8]:
            stats["crates"] += 1
            owners = (fetcher.get_json(f"{API}/crates/{crate['id']}/owner_user") or {}).get("users") or []
            for u in owners[:2]:
                login = str(u.get("login") or "")
                if not login or login.lower() in _BOTS or u.get("kind") == "team":
                    continue
                stats["owners"] += 1
                ev = Evidence(kind="publication", url=f"https://crates.io/crates/{crate['id']}",
                              text=squash(f"Maintains the Rust crate {crate['id']}: {crate.get('description') or ''}", 220),
                              detail=f"{crate.get('downloads', 0):,} downloads, version {crate.get('max_version')}")
                if login in seen:
                    seen[login].evidence.append(ev)
                    continue
                c = Contact(kind="person", name=squash(u.get("name") or login, 80), title="Open-source maintainer (crates.io)",
                            sources=["crates"], topics=[str(k) for k in (crate.get("keywords") or [])][:6])
                c.add_key(f"github:{login}")
                c.add_url(u.get("url") or f"https://github.com/{login}")
                c.evidence.append(ev)
                c.signals["maintains_package"] = 1.0
                c.signals["open_source"] = 0.8
                prof = fetcher.get_json(f"https://api.github.com/users/{login}")
                if isinstance(prof, dict) and prof.get("login"):
                    stats["profiles"] += 1
                    if prof.get("name"):
                        c.name = squash(prof["name"], 80)
                    if prof.get("blog"):
                        blog = prof["blog"] if str(prof["blog"]).startswith("http") else "https://" + str(prof["blog"])
                        c.add_url(blog)
                    if prof.get("company"):
                        c.org = squash(str(prof["company"]).lstrip("@"), 80)
                    if prof.get("email"):
                        add_published_email(c, prof["email"], 0.85, prof.get("html_url"))
                seen[login] = c
    return list(seen.values()), stats

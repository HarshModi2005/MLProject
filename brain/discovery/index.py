"""
Persistent index: prospects remembered across campaigns stay trustworthy.

    decay     a remembered prospect's proof expires with its signal's shelf life (brain/discovery/signals.py:
              hiring 30 days, coverage 90, sponsorship 540 ...), so a company "hiring now" six months ago is
              not offered as hiring today
    refresh   `brain refresh`: re-check remembered prospects against free sources — public job boards for
              companies found hiring, RSS feeds for podcasts — and update or expire their proof. No search
              budget, no LLM; meant to run on a schedule (cron) so memory is fresh when a campaign starts
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from brain.models import Contact, Evidence
from brain.textutil import squash


def decay(c: Contact, age_days: float) -> List[str]:
    """Drop signals older than their shelf life. Returns the expired signal names (also flagged on the contact)."""
    from brain.discovery.signals import SIGNALS

    expired = []
    for name in list(c.signals):
        sig = SIGNALS.get(name)
        if sig and sig.decay_days is not None and age_days > sig.decay_days and c.signals.get(name, 0) > 0:
            c.signals.pop(name, None)
            expired.append(name)
    if expired:
        c.flags = [f for f in c.flags if not f.startswith("stale:")] + [f"stale:{','.join(expired)}"]
    return expired


def _refresh_jobs(c: Contact, fetcher, stats: Dict) -> bool:
    from brain.channels import fetch_board

    key = next((k for k in c.keys if k.startswith("ats:")), None)
    if not key:
        return False
    ats, _, token = key[4:].partition(":")
    _, jobs = fetch_board(fetcher, ats, token)
    stats["boards_checked"] = stats.get("boards_checked", 0) + 1
    c.evidence = [e for e in c.evidence if not e.text.startswith("Hiring now:")]
    if jobs:
        c.signals["hiring"] = max(0.5, c.signals.get("hiring", 0.0))
        c.signals["open_roles"] = float(len(jobs))
        c.evidence.insert(0, Evidence(kind="page", url=jobs[0]["url"] or None,
                                      text=squash("Hiring now: " + "; ".join(f"{j['title']} ({j['location'] or 'n/a'})"
                                                                            for j in jobs[:4]), 300),
                                      detail=f"{len(jobs)} open roles (checked {datetime.utcnow():%Y-%m-%d})"))
        stats["still_hiring"] = stats.get("still_hiring", 0) + 1
    else:
        c.signals.pop("hiring", None)
        c.signals.pop("open_roles", None)
        stats["stopped_hiring"] = stats.get("stopped_hiring", 0) + 1
    return True


def _refresh_feed(c: Contact, fetcher, stats: Dict) -> bool:
    from brain.channels import _recent
    from brain.discovery.adapters.rss import parse_feed

    feed = next((e.source_url for e in c.emails if e.method == "feed" and e.source_url), None)
    if not feed:
        return False
    page = fetcher.get(feed)
    stats["feeds_checked"] = stats.get("feeds_checked", 0) + 1
    if not page.ok:
        return False
    info = parse_feed(page.html)
    eps = info.get("episodes") or []
    if eps and _recent(eps[0]["date"]):
        c.signals["recent_coverage"] = 1.0
    else:
        c.signals.pop("recent_coverage", None)
    if eps:
        c.evidence = [e for e in c.evidence if not e.text.startswith("Episode:")]
        c.evidence += [Evidence(kind="publication", text=f"Episode: {ep['title']}", detail=ep["date"] or None) for ep in eps[:3]]
    return True


def refresh(store, fetcher, limit: int = 500, progress=None) -> Dict:
    """Re-check remembered prospects against free sources and save what changed."""
    stats: Dict = {"scanned": 0, "updated": 0}
    updated: List[Contact] = []
    for c in store.recent_contacts(limit):
        stats["scanned"] += 1
        try:
            changed = _refresh_jobs(c, fetcher, stats) or _refresh_feed(c, fetcher, stats)
        except Exception:
            changed = False
        if changed:
            c.flags = [f for f in c.flags if not f.startswith("stale:")]
            updated.append(c)
    if updated:
        store.upsert_contacts(updated)
    stats["updated"] = len(updated)
    if progress:
        progress(f"refresh: {stats}")
    return stats

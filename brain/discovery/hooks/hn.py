"""Hacker News "Ask HN: Who is hiring?": companies posting open roles this month (and last), often with the
founder's or hiring manager's own email in the post. Free (HN Algolia API), worldwide, strong on startups."""

from __future__ import annotations

import html
import re
from typing import Dict, List, Tuple

from brain.discovery.normalize import add_published_email
from brain.models import Contact, Evidence
from brain.textutil import registrable_domain, squash

API = "https://hn.algolia.com/api/v1"
_HREF = re.compile(r'href="([^"]+)"')


def _text(comment_html: str) -> str:
    t = re.sub(r"<p>", "\n", comment_html or "")
    return html.unescape(re.sub(r"<[^>]+>", " ", t))


def parse_post(comment_html: str) -> Dict:
    text = _text(comment_html)
    first = squash(text.split("\n", 1)[0], 300)
    parts = [p.strip() for p in first.split("|") if p.strip()]
    links = [html.unescape(u) for u in _HREF.findall(comment_html or "")]
    site = next((u for u in links if u.startswith("http") and "ycombinator.com" not in u
                 and not re.search(r"(greenhouse|lever|ashbyhq|workable|linkedin|forms\.gle|docs\.google|calendly)", u)), None)
    from brain.emailfinder import extract_emails

    return {"company": parts[0] if parts else "", "first_line": first, "text": squash(text, 1200),
            "emails": extract_emails(text), "site": site}


def discover(run) -> Tuple[List[Contact], Dict]:
    from brain.channels import _in_place

    fetcher, spec = run.deps.fetcher, run.spec
    stats = {"threads": 0, "posts": 0, "kept": 0}
    idx = fetcher.get_json(f"{API}/search_by_date", params={"tags": "story,author_whoishiring", "hitsPerPage": 6})
    threads = [h for h in (idx or {}).get("hits", []) if "who is hiring" in (h.get("title") or "").lower()][:2] \
        if isinstance(idx, dict) else []
    stats["threads"] = len(threads)
    found: Dict[str, Contact] = {}
    for th in threads:
        for term in spec.topics[:3]:
            data = fetcher.get_json(f"{API}/search", params={"tags": f"comment,story_{th['objectID']}", "query": term,
                                                             "hitsPerPage": 50})
            for h in (data or {}).get("hits", []) if isinstance(data, dict) else []:
                if str(h.get("parent_id")) != str(th["objectID"]):
                    continue  # replies, not job posts
                stats["posts"] += 1
                post = parse_post(h.get("comment_text") or "")
                name = squash(re.sub(r"\(.*?\)", "", post["company"]), 80)
                if not name or len(name) > 60 or name.lower() in found:
                    continue
                if (spec.regions or spec.countries) and not _in_place(post["first_line"] + " " + post["text"][:400], spec):
                    continue
                c = Contact(kind="org", name=name, org=name, sources=["hn_hiring"])
                if post["site"]:
                    c.org_domain = registrable_domain(post["site"])
                    c.add_url(post["site"])
                url = f"https://news.ycombinator.com/item?id={h.get('objectID')}"
                c.add_url(url)
                c.evidence.append(Evidence(kind="page", url=url, text=squash(f"Hiring now (HN: {th.get('title')}): {post['first_line']}", 300),
                                           detail=squash(post["text"], 300)))
                for e in post["emails"]:
                    if add_published_email(c, e, 0.8, url):
                        break
                c.signals["hiring"] = 1.0
                c.location_text = squash(" | ".join(post["first_line"].split("|")[1:4]), 120) or None
                found[name.lower()] = c
                stats["kept"] += 1
    return list(found.values()), stats

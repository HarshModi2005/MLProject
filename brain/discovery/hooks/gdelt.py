"""GDELT DOC API: recent articles on the topic worldwide, free (no search budget) -> journalists from bylines.
Same reading as the `news` source, so the two merge cleanly."""

from __future__ import annotations

from typing import Dict, List

from brain.discovery.hooks import news


def _article(item: Dict) -> Dict:
    return {"title": item.get("title") or "", "url": item.get("url"), "source": item.get("domain") or "",
            "date": item.get("seendate") or "", "snippet": ""}


def extract(spec, run, hit: Dict, page) -> List[Dict]:
    return news.extract(spec, run, _article(hit), page)


to_contact = news.to_contact
key = news.key
merge = news.merge

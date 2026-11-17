"""
json_api: a JSON API queried once per campaign term.

    "request": {"url": "https://api.example.com/search", "params": {"q": "{term}", "country": "{country|lower}"}},
    "query":   {"terms": ["topics", "expanded_terms:2"], "max": 5, "default_country": "US"},
    "items":   "results",            # path to the list in the answer (omit if the answer is the list)
    "dedupe":  "id",                 # item path that identifies an item across terms
    "follow":  {"field": "feedUrl", "parse": "rss", "as": "feed", "max": 30}   # optional

"request": {"method": "POST", "body": {...templates...}} posts a JSON body (paid search APIs such as Exa).
"request": {"format": "text", ...} reads an API that answers in XML/text; the source's hook
`parse(spec, text) -> [items]` turns it into items.
Without "follow", each item is a record. With it, a record is {"item": <item>, "<as>": <what the URL held>}.
Template variables: term, country, place, role, objective.
"""

from __future__ import annotations

from typing import Any, Dict, List

from brain.discovery.adapters.common import finish, follow_items, get_json, get_text, post_json
from brain.discovery.contracts import Record, SourceSpec
from brain.discovery.normalize import campaign_terms, campaign_vars, get_path, hook, render


def _render_deep(obj: Any, v: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _render_deep(x, v) for k, x in obj.items()}
    if isinstance(obj, list):
        return [_render_deep(x, v) for x in obj]
    return " ".join(render(obj, v).split()) if isinstance(obj, str) else obj


def query_items(spec: SourceSpec, run, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    req = spec.request
    base = campaign_vars(run.spec, spec.query, run.playbook)
    uses_term = "{term" in str(req)
    terms = campaign_terms(run.spec, spec.query) if uses_term else [""]
    items: List[Dict[str, Any]] = []
    seen: set = set()
    for term in terms:
        v = dict(base, term=term)
        url = render(req["url"], v)
        params = {k: render(val, v) for k, val in (req.get("params") or {}).items()}
        if req.get("method", "GET").upper() == "POST":  # a JSON body rendered from the campaign (e.g. Exa)
            body = _render_deep(req.get("body") or {}, v)
            data = post_json(spec, run, url, body)
        elif req.get("format") == "text":  # an API answering in XML/text: the source's hook `parse` reads it
            text = get_text(spec, run, url, params or None)
            parse = hook(spec, "parse")
            data = parse(spec, text or "") if parse and text else []
        else:
            data = get_json(spec, run, url, params or None)
        stats["requests"] = stats.get("requests", 0) + 1
        found = get_path(data, spec.items) if spec.items else data
        for it in found if isinstance(found, list) else []:
            if not isinstance(it, dict):
                continue
            if spec.dedupe:
                k = get_path(it, spec.dedupe)
                if k is None or k in seen:
                    continue
                seen.add(k)
            items.append(it)
    stats["items"] = len(items)
    return items


def collect(spec: SourceSpec, run):
    stats: Dict[str, Any] = {}
    items = query_items(spec, run, stats)
    if spec.follow:
        records = follow_items(spec, run, items, stats)
    else:
        records = [Record(source=spec.name, data=it) for it in items[: spec.max_items]]
    return finish(spec, run, records, stats)

"""
search: a search vertical run through the campaign's SearchPlanner (budget, query log, no repeats, replan).

    "search": {
      "vertical": "web" | "news" | "places",
      "kind": "news",                 # planner query kind (default: the source name); "search"/"news"/"places"
                                      # use the planner's own initial queries when no templates are given
      "templates": ["site:play.google.com/store/apps {term} {place}"],   # optional, rendered per term
      "max_queries": 6, "first_round_max": 5, "reserve": 4, "n": 10, "raw": false,
      "gl": "single_country",         # pass the campaign's country to the engine when it has exactly one
      "match": "regex"                # keep only hits whose url/title/snippet match
    },
    "follow": {"max": 10}             # optional: read each hit's page; its hook "extract" or "select" reads it

Without "follow", each hit {url, title, snippet[, source, date]} is a record.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from brain.discovery.adapters.common import extract_from_page, text_matches
from brain.discovery.contracts import Record, SourceSpec
from brain.discovery.normalize import Collector, campaign_terms, campaign_vars, render


def _queries(spec: SourceSpec, run) -> List[str]:
    cfg = spec.search
    base = campaign_vars(run.spec, spec.query, run.playbook)
    out = []
    for term in campaign_terms(run.spec, spec.query or {"terms": ["topics"], "max": 3}):
        for tpl in cfg.get("templates") or []:
            out.append(" ".join(render(tpl, dict(base, term=term)).split()))
    out = list(dict.fromkeys(q for q in out if q))
    return [q for q in out if not run.ctx.tried(q)][: int(cfg.get("max_queries") or 6)]


def collect(spec: SourceSpec, run):
    cfg = spec.search
    search, fetcher = run.deps.search, run.deps.fetcher
    vertical = cfg.get("vertical", "web")
    gl = (run.spec.countries[0] if len(run.spec.countries) == 1 else "") if cfg.get("gl") == "single_country" else ""
    stats: Dict[str, Any] = {"queries": 0, "hits": 0}
    col = Collector(spec, run)
    follow = spec.follow or {}

    def hits_for(query: str) -> List[Dict[str, Any]]:
        if vertical == "news":
            raw = search.news(query, gl=gl) if "n" not in cfg else search.news(query, n=int(cfg["n"]), gl=gl)
        elif vertical == "places":
            raw = search.places(query, gl=gl)
        else:
            raw = search.search(query, n=int(cfg.get("n") or 10), raw=bool(cfg.get("raw")))
        rows = [asdict(h) for h in raw]
        return [h for h in rows if text_matches(cfg.get("match"), h.get("url", "") or h.get("website", ""),
                                               h.get("title", "") or h.get("name", ""), h.get("snippet", ""))]

    def execute(query: str):
        stats["queries"] += 1
        hits = hits_for(query)
        stats["hits"] += len(hits)
        new = 0
        if follow:
            take = hits[: int(follow.get("max") or 10)]
            for h, page in zip(take, fetcher.get_many([h.get("url") for h in take])):
                if not page.ok:
                    continue
                for row in extract_from_page(spec, run, page, h):
                    new += col.add(Record(source=spec.name, url=h.get("url"), data=row))
        else:
            for h in hits:
                new += col.add(Record(source=spec.name, url=h.get("url") or h.get("website"), data=h))
        return len(hits), new

    reserve = int(cfg.get("reserve", 2))
    start = search.calls

    def left() -> int:
        lim = search.remaining - reserve
        return min(lim, run.budget - (search.calls - start)) if run.budget is not None else lim

    queries = _queries(spec, run) if cfg.get("templates") else None
    run.planner.run(execute, have=lambda: len(col), budget_left=left, kind=cfg.get("kind") or spec.name,
                    first_round_max=int(cfg.get("first_round_max") or 8), queries=queries)
    stats["found"] = len(col)
    return col.contacts(), stats

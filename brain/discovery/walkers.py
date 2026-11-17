"""
Graph walks: expansion tools that start from the best finds so far (stage "expand").

    graph_walk    free: neighbours of the winners in the stored entity graph (co-sponsors of the same events,
                  colleagues at the same outlet, people sharing a paper or a package), from this and earlier campaigns
    sponsor_walk  search: the other events the winning sponsors backed, and who else sponsored those
    beat_walk     search (news): other journalists writing on the topic for the winners' outlets
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from brain.models import Contact, Evidence
from brain.textutil import squash

_EDGE_TEXT = {"sponsored": "Also sponsored {via}", "spoke_at": "Also spoke at {via}", "wrote_for": "Also writes for {via}",
              "works_at": "Works at {via}, like a strong match", "published": "Co-author of {via}",
              "maintains": "Also maintains {via}", "hosts": "Hosts {via}"}


def graph_walk(run) -> Tuple[List[Contact], Dict]:
    from brain.discovery.graph import neighbours

    store = getattr(run.deps, "store", None)
    winners = list(run.winners or [])
    stats = {"seeds": len(winners), "neighbours": 0, "from_memory": 0}
    if not winners or store is None:
        return [], stats
    kind = run.spec.target_kind or run.playbook.target_kind
    out: List[Contact] = []
    for n in neighbours(store, winners):
        if n["type"] not in ("org", "person") or (kind and n["type"] != kind):
            continue
        c = store.contact(n["contact_id"]) if n.get("contact_id") else None
        if c is not None:
            stats["from_memory"] += 1
            c.score, c.reasons, c.prescore, c.prejudge, c.prejudge_reason = None, [], 0.0, None, None
            c.flags = [f for f in c.flags if f not in ("unjudged", "prejudge_reject")]
        else:
            key = n["node"].split(":", 1)[1]
            if n["type"] == "org":
                dom = key if "." in key and " " not in key else None
                c = Contact(kind="org", name=n.get("name") or key, org=n.get("name") or key, org_domain=dom)
                if dom:
                    c.add_url(f"https://{dom}/")
            else:
                c = Contact(kind="person", name=n.get("name") or key.split("|")[0].title())
        c.evidence.append(Evidence(kind="page", url=n.get("url"),
                                   text=squash(_EDGE_TEXT.get(n["edge"], "Connected to {via}").format(via=n["via"]), 200)))
        if n["edge"] == "sponsored":
            c.signals["past_sponsorship"] = max(c.signals.get("past_sponsorship", 0.0), 0.5)
        c.sources = list(dict.fromkeys(c.sources + ["graph_walk"]))
        out.append(c)
    stats["neighbours"] = len(out)
    return out, stats


def sponsor_walk(run) -> Tuple[List[Contact], Dict]:
    from brain.channels import discover_events

    sponsors = [w for w in run.winners or [] if w.kind == "org" and w.signals.get("past_sponsorship", 0) > 0][:3]
    if not sponsors:
        return [], {"seeds": 0}
    place = (run.spec.regions or [""])[0]
    topic = (run.spec.topics or [""])[0]
    queries = []
    for w in sponsors:
        queries.append(squash(f'"{w.org or w.name}" sponsors hackathon {place}'))
        queries.append(squash(f'"{w.org or w.name}" sponsor {topic} event'))
    found, stats = discover_events(run, queries=queries, source="sponsor_walk")
    stats["seeds"] = len(sponsors)
    return found, stats


def beat_walk(run) -> Tuple[List[Contact], Dict]:
    """Journalists who cover the topic for the same outlets as the winners (Serper news, site: filter)."""
    from brain.discovery.contracts import Record
    from brain.discovery.hooks import news
    from brain.discovery.registry import SPECS

    outlets = list(dict.fromkeys(w.org_domain for w in run.winners or [] if w.kind == "person" and w.org_domain))[:3]
    stats = {"outlets": len(outlets), "queries": 0}
    if not outlets or not run.deps.search.available:
        return [], stats
    spec = SPECS.get("news")
    search, fetcher = run.deps.search, run.deps.fetcher
    known = {w.name.lower() for w in run.winners or []}
    found: Dict[str, Contact] = {}
    allowance = run.budget if run.budget is not None else 4
    for dom in outlets:
        for topic in run.spec.topics[:2]:
            if stats["queries"] >= allowance or search.remaining <= 3:
                break
            q = squash(f"site:{dom} {topic}")
            if run.ctx.tried(q):
                continue
            stats["queries"] += 1
            arts = search.news(q)
            run.ctx.log_query(q, "news", len(arts), 0)
            for art, page in zip(arts[:6], fetcher.get_many([a.url for a in arts[:6]])):
                if not page.ok:
                    continue
                hit = {"title": art.title, "url": art.url, "source": art.source, "date": art.date, "snippet": art.snippet}
                for row in news.extract(spec, run, hit, page):
                    c = news.to_contact(spec, Record(source="beat_walk", url=art.url, data=row), run)
                    if c is None or c.name.lower() in known:
                        continue
                    c.sources = ["beat_walk"]
                    if c.name.lower() in found:
                        news.merge(found[c.name.lower()], c, Record(source="beat_walk", data=row))
                    else:
                        found[c.name.lower()] = c
    return list(found.values()), stats

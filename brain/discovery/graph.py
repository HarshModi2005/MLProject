"""
Entity graph: what discovery learns about how prospects connect, kept across campaigns.

    nodes   org (by website), person (by email, else name|organisation), event, outlet, paper, package, show
    edges   sponsored, spoke_at, works_at, wrote_for, published, maintains, hosts

Edges are read off the sourced evidence channels already attach ("Sponsor or partner of <event>" with the event
page's URL, a byline on an outlet's article, a paper, an npm package...), so the graph costs nothing to build.
Walkers (brain/discovery/walkers.py) use it to expand from the best finds: the other sponsors of the events a
winner sponsored, the colleagues of a journalist on the same outlet, a researcher's co-authors.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from brain.models import Contact
from brain.textutil import name_tokens, registrable_domain

EDGE_TYPES = ("sponsored", "spoke_at", "works_at", "wrote_for", "published", "maintains", "hosts")


def node_of(c: Contact) -> str:
    if c.kind == "org":
        return f"org:{registrable_domain(c.org_domain)}" if c.org_domain else f"org:{(c.org or c.name).lower()}"
    best = c.best_email
    if best and best.method not in ("pattern", "pattern_global", "pattern_prior"):
        return f"person:{best.address}"
    return f"person:{' '.join(name_tokens(c.name))}|{(c.org_domain or c.org or '').lower()}"


def _event_key(url: str) -> str:
    """An event is identified by its site and first path segment (hackfest.in/2025, mlh.io/seasons/2026)."""
    parts = urlsplit(url)
    first = next((p for p in parts.path.split("/") if p), "")
    first = "" if re.match(r"(sponsors?|partners?|speakers?|judges?)$", first, re.I) else first
    return f"{registrable_domain(url)}/{first}".rstrip("/")


def extract(c: Contact) -> Tuple[List[Dict], List[Dict]]:
    me = node_of(c)
    nodes = [{"id": me, "type": c.kind, "name": c.name, "contact_id": c.id or None}]
    edges: List[Dict] = []
    src = (c.sources or ["?"])[0]

    def link(dst_type: str, key: str, name: str, edge: str, url: Optional[str] = None):
        if not key:
            return
        dst = f"{dst_type}:{key}"
        nodes.append({"id": dst, "type": dst_type, "name": name})
        edges.append({"src": me, "dst": dst, "type": edge, "source": src, "url": url})

    for ev in c.evidence:
        text = ev.text or ""
        if text.startswith("Sponsor or partner of") and ev.url:
            link("event", _event_key(ev.url), text[len("Sponsor or partner of "):][:200], "sponsored", ev.url)
        elif text.startswith("Speaker, judge or mentor at") and ev.url:
            link("event", _event_key(ev.url), text[len("Speaker, judge or mentor at "):][:200], "spoke_at", ev.url)
        elif text.startswith("Maintains the npm package") and ev.url:
            link("package", ev.url.rsplit("/package/", 1)[-1], text[:120], "maintains", ev.url)
        elif text.startswith("Hosts the podcast"):
            link("show", (c.org or "").lower(), c.org or "", "hosts", ev.url)
        elif ev.kind == "publication" and ev.url and c.kind == "person" and "Journalist" not in (c.title or ""):
            link("paper", ev.url, text[:200], "published", ev.url)
    if c.kind == "person" and c.org_domain:
        journalist = "journalist" in (c.title or "").lower()
        link("outlet" if journalist else "org", registrable_domain(c.org_domain), c.org or c.org_domain,
             "wrote_for" if journalist else "works_at")
    return nodes, edges


def record(store, contacts: List[Contact]) -> int:
    """Write the graph of these candidates. Learning is a bonus: never fails the caller."""
    if store is None or not hasattr(store, "graph_write"):
        return 0
    nodes: Dict[str, Dict] = {}
    edges: Dict[Tuple[str, str, str], Dict] = {}
    for c in contacts:
        ns, es = extract(c)
        for n in ns:
            if n["id"] not in nodes or n.get("contact_id"):
                nodes[n["id"]] = n
        for e in es:
            edges[(e["src"], e["dst"], e["type"])] = e
    try:
        store.graph_write(list(nodes.values()), list(edges.values()))
    except Exception:
        return 0
    return len(edges)


def neighbours(store, seeds: List[Contact], via: Tuple[str, ...] = EDGE_TYPES, limit: int = 60) -> List[Dict]:
    """Two-hop neighbours of the seeds: seed -edge-> shared node <-edge- other. Returns
    [{"node": id, "contact_id", "name", "type", "via": shared node name, "edge": type, "url"}], strongest first
    (a neighbour sharing several nodes with the seeds ranks higher)."""
    if store is None or not hasattr(store, "graph_edges"):
        return []
    mine = {node_of(c) for c in seeds}
    first = [e for e in store.graph_edges(list(mine), list(via)) if e["src"] in mine]
    shared = {e["dst"]: e for e in first}
    if not shared:
        return []
    second = [e for e in store.graph_edges(list(shared), list(via)) if e["dst"] in shared and e["src"] not in mine]
    counts: Dict[str, int] = {}
    for e in second:
        counts[e["src"]] = counts.get(e["src"], 0) + 1
    info = store.graph_nodes(list(counts) + list(shared))
    out = []
    for e in sorted(second, key=lambda e: counts[e["src"]], reverse=True):
        n = info.get(e["src"]) or {}
        if any(o["node"] == e["src"] for o in out):
            continue
        out.append({"node": e["src"], "contact_id": n.get("contact_id"), "name": n.get("name"), "type": n.get("type"),
                    "via": (info.get(e["dst"]) or {}).get("name") or e["dst"], "edge": e["type"], "url": e["url"],
                    "shared": counts[e["src"]]})
        if len(out) >= limit:
            break
    return out

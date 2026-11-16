"""
Adaptive discovery: find enough QUALIFIED prospects for the least spend.

    round 0  seed     memory (free) -> the playbook's tools, plus any tool whose proof the fit
                      criteria need; search budget split between them, primary tool first
    round 1+ adapt    the best finds so far become seeds: lookalike organisations, co-authors;
                      the remaining budget goes to the tools that have produced the most qualified
                      prospects per search call (UCB bandit); tools that stop producing are retired
    every round       resolve duplicates, score against the prospect profile (no LLM), note poor
                      fits (anti-patterns) as negative feedback for the query replanner
    stop              enough qualified prospects, budget spent, or nothing new for a whole round

What counts as qualified is brain/fit.py: a candidate with support for the profile's criteria and
no anti-pattern. Everything here is free except the search calls the tools make (and the cheap
triage / list-extraction calls inside the org tools).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from brain.discovery.ledger import Ledger
from brain.fit import anti_pattern_hits, profile_for, promising, quick_fit
from brain.models import CampaignSpec, Contact, ProspectProfile
from brain.playbooks import Playbook
from brain.resolve import resolve
from brain.tools import TOOLS, ToolRun

log = logging.getLogger("brain.explore")


@dataclass
class ExploreResult:
    contacts: List[Contact] = field(default_factory=list)
    by_source: Dict[str, int] = field(default_factory=dict)
    tool_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    qualified: int = 0
    warnings: List[str] = field(default_factory=list)
    ledger: Dict[str, Dict[str, float]] = field(default_factory=dict)  # per source: spend and yield


def choose_tools(spec: CampaignSpec, pb: Playbook, prof: ProspectProfile, *, has_seeds: bool,
                 use_memory: bool = True) -> Tuple[List[str], List[str]]:
    """(seed tools in run order, expansion tools). See brain/discovery/router.py."""
    from brain.discovery.router import choose

    return choose(spec, pb, prof, has_seeds=has_seeds, use_memory=use_memory)


def _key(c: Contact) -> str:
    if c.kind == "org" and c.org_domain:
        return c.org_domain
    strong = next((k for k in c.keys if k.startswith(("openalex:", "orcid:", "ats:"))), None)
    return strong or (c.name.lower() + "|" + (c.org_domain or c.org or "").lower())


def _merge_stats(into: Dict[str, Any], new: Dict[str, Any]) -> None:
    for k, v in (new or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(into.get(k, 0), (int, float)):
            into[k] = into.get(k, 0) + v
        elif k not in into:
            into[k] = v


def _ucb(st: Dict[str, float], total_runs: int) -> float:
    runs = max(1.0, st.get("runs", 0))
    mean = st.get("qualified", 0) / (st.get("cost", 0) + 1.0)
    return mean + math.sqrt(2 * math.log(max(2, total_runs)) / runs) * 0.5


def explore(spec: CampaignSpec, pb: Playbook, ctx, deps, planner, *, seeds: Optional[List[Contact]] = None,
            budget: int, rounds: int = 3, need: Optional[int] = None,
            progress: Callable[[str], None] = lambda m: None, winners: Optional[List[Contact]] = None,
            tools: Optional[List[str]] = None) -> ExploreResult:
    """Run discovery until `need` qualified prospects exist or `budget` search calls are spent.
    `winners` (e.g. the judge's picks in a backfill) start expansion immediately; `tools` overrides the choice."""
    prof = profile_for(spec, pb)
    need = need or planner.need
    res = ExploreResult()
    raw: List[Contact] = list(seeds or [])
    if seeds:
        res.by_source["your_lists"] = len(seeds)
    search = deps.search
    start_calls = search.calls
    retired: set = set()
    ledger = Ledger()

    def spent() -> int:
        return search.calls - start_calls

    def left() -> int:
        return max(0, budget - spent())

    def pool() -> List[Contact]:
        return resolve([c.model_copy(deep=True) for c in raw])

    def qualified(cs: List[Contact]) -> List[Contact]:
        out = []
        for c in cs:
            ok = promising(c, prof) if c.kind == "org" and "light_read" not in c.flags else quick_fit(c, prof) >= 0.25
            if ok:
                out.append(c)
        return out

    def run_tool(name: str, allowance: Optional[int], win: List[Contact]) -> int:
        tool = TOOLS[name]
        if tool.needs_search and not search.available:
            res.warnings.append(f"Tool '{name}' needs SERPER_API_KEY (or TAVILY_API_KEY); skipped.")
            retired.add(name)
            return 0
        if tool.needs_search and (left() <= 1 or search.remaining <= 2):
            return 0
        progress(f"discover: {tool.description}")
        before = search.calls
        with ledger.measure(name, deps) as entry:
            try:
                found, stats = tool.fn(ToolRun(spec, pb, ctx, deps, planner, budget=allowance, winners=win))
            except Exception as e:  # one failing tool must not sink the campaign
                log.warning("tool %s failed: %s", name, e)
                res.warnings.append(f"Tool '{name}' failed: {e}")
                retired.add(name)
                entry["errors"] = 1
                return 0
            cost = search.calls - before
            for c in found:  # credit the tool that found it (the per-source funnel and ledger key on this)
                if name not in c.sources:
                    c.sources.append(name)
            known = {_key(c) for c in raw}
            fresh = [c for c in found if _key(c) not in known]
            q = len(qualified(fresh))
            entry.update(found=len(found), new=len(fresh), qualified=q)
        ctx.log_tool(name, cost, len(fresh), q)
        res.by_source[name] = res.by_source.get(name, 0) + len(found)
        _merge_stats(res.tool_stats.setdefault(name, {}), stats)
        if stats.get("failed_terms"):
            res.warnings.append("Publication search failed for: " + ", ".join(stats["failed_terms"]) + " (re-run to retry).")
        for c in fresh:  # negative feedback for the replanner: what poor fits look like
            hits = anti_pattern_hits(prof, c)
            if hits and len(ctx.rejected_reasons) < 12:
                ctx.rejected_reasons.append(f"{c.name}: looks like '{hits[0]}'")
        raw.extend(found)
        if not fresh or (tool.needs_search and cost == 0):
            retired.add(name)  # it has nothing more to give in this campaign
        return q

    seed_tools, expand_tools = choose_tools(spec, pb, prof, has_seeds=bool(ctx.seed_names),
                                            use_memory=getattr(deps.settings, "use_memory", True))
    from brain.discovery.router import skipped

    for why in skipped(spec, pb):
        res.warnings.append(f"Source {why}; skipped.")
    if tools is not None:
        seed_tools, expand_tools = [t for t in tools if TOOLS[t].stage != "expand"], [t for t in tools if TOOLS[t].stage == "expand"]

    # ── round 0: seed ────────────────────────────────────────────────────────
    if winners is None:
        paid = [t for t in seed_tools if TOOLS[t].needs_search]
        shares = {t: (2 if i == 0 else 1) for i, t in enumerate(paid)}
        total_share = sum(shares.values()) or 1
        # Hold back a third of the budget for the adaptive rounds when there is something to adapt to.
        seed_budget = budget if rounds <= 1 else int(budget * 0.67)
        # Free tools always run (and first): they cost nothing and often carry what paid tools lack (e.g. podcast
        # feeds publish an email; bylines do not). Paid tools stop once enough qualified prospects exist.
        ordered = [t for t in seed_tools if not TOOLS[t].needs_search] + [t for t in seed_tools if TOOLS[t].needs_search]
        for name in ordered:
            if TOOLS[name].needs_search and len(qualified(pool())) >= need:
                break
            allowance = max(3, int(seed_budget * shares[name] / total_share)) if name in shares else None
            run_tool(name, allowance, [])
        current = pool()
        res.rounds.append({"round": 0, "tools": seed_tools, "spent": spent(), "qualified": len(qualified(current))})
    else:
        current = pool()

    # ── rounds 1+: adapt ─────────────────────────────────────────────────────
    for r in range(1, max(1, rounds)):
        q_now = qualified(current)
        if len(q_now) >= need or (left() <= 1 and not expand_tools):
            break
        win = winners if (winners is not None and r == 1) else sorted(q_now, key=lambda c: quick_fit(c, prof),
                                                                      reverse=True)[:5]
        before_q = len(q_now)
        ran = []
        for name in expand_tools:
            if name in retired or not win:
                continue
            run_tool(name, max(3, left() // 2) if TOOLS[name].needs_search else None, win)
            ran.append(name)
        # Bandit: the rest of this round's budget to the tool with the best qualified-per-search record.
        live = [t for t in seed_tools if TOOLS[t].needs_search and t not in retired]
        if live and left() > 1:
            total_runs = int(sum(ctx.tool_stats.get(t, {}).get("runs", 0) for t in live)) or 1
            best = max(live, key=lambda t: _ucb(ctx.tool_stats.get(t, {}), total_runs))
            rounds_left = max(1, rounds - r)
            run_tool(best, max(3, left() // rounds_left), [])
            ran.append(best)
        current = pool()
        gained = len(qualified(current)) - before_q
        res.rounds.append({"round": r, "tools": ran, "spent": spent(), "qualified": len(qualified(current)),
                           "gained": gained})
        if not ran or (gained <= 0 and left() <= 1):
            break

    res.contacts = current
    res.qualified = len(qualified(current))
    res.ledger = ledger.report()
    return res

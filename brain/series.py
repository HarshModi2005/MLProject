"""
Series: one brief, run in many batches ("find me more") without ever repeating a person.

    brain run "..."          starts a series (its first campaign's id is the series id)
    brain more <series> -n   the next batch: same brief, new people only
    brain more <series> --until 200

How a batch finds NEW good people when the obvious ones are gone (the widening ladder, one rung at a time, only as far
as the user allowed):

    rung 0  the brief again, with everything earlier batches learned: queries already tried are never repeated, the
            sources that paid off keep their budget, the reasons the judge turned people down steer the new searches
    rung 1  lookalikes of the best earlier finds (people who replied or were approved, else the top-scored)
    rung 2  neighbouring topics (the brief's expanded terms become topics)                   needs --widen topics
    rung 3  a wider place (regions dropped, the countries kept)                              needs --widen geo

The quality bar is never lowered to fill a batch. If the ladder runs out and new matches are still very few, people
shown earlier but never decided on (and those on standby) may be offered again, best score first and without being
judged or written again (`recycle`). Rejected people and anyone already emailed are never offered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from brain import touch
from brain.context import CampaignContext
from brain.models import CampaignSpec, Contact
from brain.pipeline import Deps, RunResult, plan_recycle, run_campaign, supply_summary
from brain.spec import new_campaign_id

log = logging.getLogger("brain.series")

WIDEN_RUNG = {"": 1, "topics": 2, "geo": 3}  # the highest rung each --widen level allows
RUNG_NAMES = ["the brief again with new queries", "lookalikes of the best finds", "neighbouring topics", "a wider place"]


class SeriesPaused(RuntimeError):
    """The series stopped itself (bounce rate too high); `brain series-resume` lifts it."""


@dataclass
class BatchOutcome:
    series_id: str
    requested: int
    campaigns: List[str] = field(default_factory=list)
    results: List[RunResult] = field(default_factory=list)
    fresh: int = 0
    recycled: int = 0
    covered_skipped: int = 0
    rung_reached: int = 0
    exhausted: bool = False
    stopped: str = ""  # why it stopped short, in words
    warnings: List[str] = field(default_factory=list)
    search_calls: int = 0

    @property
    def delivered(self) -> int:
        return self.fresh + self.recycled


# ── what carries over between batches ────────────────────────────────────────

def merge_context(store, series_id: str) -> CampaignContext:
    """The series' working memory: everything earlier batches learned, so a new batch does not start from zero."""
    out = CampaignContext()
    seen_q = set()
    for cid in store.series_campaigns(series_id):
        c = store.load_context(cid)
        out.understanding = out.understanding or c.understanding
        out.target_brief = out.target_brief or c.target_brief
        out.assumptions = list(dict.fromkeys(out.assumptions + c.assumptions))
        for f in c.facts:
            out.add_fact(f.text, f.source)
        for n in c.seed_names:
            out.add_seed(name=n)
        for d in c.seed_domains:
            out.add_seed(domain=d)
        for v in c.exclude_emails + c.exclude_domains + c.exclude_names:
            out.exclude(v)
        out.style_examples = list(dict.fromkeys(out.style_examples + c.style_examples))[:3]
        out.uploads = out.uploads or c.uploads
        for q in c.query_log:  # queries already paid for are never run again (ctx.tried)
            key = q.query.lower()
            if key not in seen_q:
                seen_q.add(key)
                out.query_log.append(q)
        for name, st in c.tool_stats.items():  # the explorer keeps allocating budget to what paid off
            mine = out.tool_stats.setdefault(name, {"runs": 0, "cost": 0.0, "found": 0, "qualified": 0})
            for k, v in st.items():
                mine[k] = mine.get(k, 0) + v
        out.rejected_reasons = (out.rejected_reasons + c.rejected_reasons)[-20:]
        out.notes = list(dict.fromkeys(out.notes + c.notes))[:12]
    out.query_log = out.query_log[-300:]
    return out


def good_examples(store, series_id: str, limit: int = 5) -> List[Contact]:
    """Who to find more like: people who replied or were approved or emailed, else the best-scored earlier finds."""
    rows = []
    for cid in store.series_campaigns(series_id):
        rows += [r for r in store.results(cid) if r["status"] != "recycled"]
    liked = [r for r in rows if r["status"] in ("replied", "approved", "sent", "followed_up")]
    pool = liked or [r for r in rows if r["status"] not in ("rejected", "bounced")]
    pool.sort(key=lambda r: r["score"] or 0, reverse=True)
    return [r["contact"] for r in pool[:limit]]


def batch_spec(store, series_id: str, n: int, rung: int, recycle: str = "never") -> CampaignSpec:
    """The next batch's spec: the series' latest brief with a new id, the number wanted, and (higher rungs) widened."""
    ids = store.series_campaigns(series_id)
    base = store.load_campaign(ids[0])["spec"].model_copy(deep=True)  # always the original brief, never an earlier widening
    base.id = new_campaign_id()
    base.series_id = series_id
    base.target_count = n
    base.recycle = recycle
    if rung >= 2:  # neighbouring topics: the terms the brief already expanded to
        extra = [t for t in base.expanded_terms if t.lower() not in {x.lower() for x in base.topics}][:3]
        base.topics = list(base.topics) + extra
        base.search_queries = []  # replan from the wider topics
    if rung >= 3:
        base.regions = []  # countries stay; only the sub-region is dropped
    return base


# ── batches ──────────────────────────────────────────────────────────────────

def resolve_series(store, ref: str) -> str:
    """A series id, or the id of any campaign in it."""
    sid = store.series_id_of(ref)
    if store.get_series(sid) is None:
        if store.load_campaign(ref) is None:
            raise KeyError(f"no series or campaign {ref!r}")
        raise KeyError(f"{ref!r} has no series record; run it once with this version of the brain first")
    return sid


def more(deps: Deps, ref: str, n: int, *, widen: Optional[str] = None, recycle: Optional[str] = None,
         confirm_recycle: Optional[Callable[[Dict[str, Any]], bool]] = None,
         progress: Callable[[str], None] = lambda m: None, sync: bool = True,
         run: Callable[..., RunResult] = run_campaign) -> BatchOutcome:
    """The next batch of an outreach: up to `n` NEW people (plus, only if allowed and needed, undecided earlier ones).
    Climbs the widening ladder while new matches fall short, then stops and says why."""
    store = deps.store
    sid = resolve_series(store, ref)
    if sync:
        try:  # what the user decided in the review UI (rejected, standby, sent, bounced) must count before we search
            from brain.bridge import sync_outcomes

            sync_outcomes(store)
        except Exception as e:  # never block a batch on a sync problem
            log.warning("could not sync review decisions: %s", e)
    row = store.get_series(sid)
    if row["status"] == "paused":
        raise SeriesPaused(row["pause_reason"])
    paused = store.check_series_health(sid)
    if paused:
        raise SeriesPaused(paused)
    if widen is not None:
        if widen not in WIDEN_RUNG:
            raise ValueError(f"widen must be one of {[k for k in WIDEN_RUNG if k]}")
        store.update_series(sid, widen=widen)
        row["widen"] = widen
    recycle = recycle or row["recycle"]
    max_rung = WIDEN_RUNG.get(row["widen"], 1)
    rung = min(row["rung"], max_rung)
    out = BatchOutcome(series_id=sid, requested=n)
    remaining = n
    while remaining > 0:
        ctx = merge_context(store, sid)
        if rung >= 1:
            for ex in good_examples(store, sid):
                ctx.add_seed(name=ex.name, domain=ex.org_domain or "")
            if rung == 1 and not ctx.seed_names:
                if rung < max_rung:
                    rung += 1  # nobody to find more like yet: a lookalike round would just repeat rung 0
                    continue
                break
        spec = batch_spec(store, sid, remaining, rung, recycle="never")  # recycling is the last resort, below
        progress(f"batch: asking for {remaining} new — {RUNG_NAMES[rung]}")
        res = run(spec, deps, ctx=ctx, progress=progress)
        store.update_series(sid, rung=rung)
        out.campaigns.append(spec.id)
        out.results.append(res)
        out.rung_reached = rung
        sup = res.funnel.get("supply") or supply_summary(spec, res.funnel, res)
        out.fresh += sup["fresh"]
        out.covered_skipped += sup["covered_skipped"]
        out.search_calls += int((res.funnel.get("cost") or {}).get("search_calls", 0))
        out.warnings += [w for w in res.warnings if w not in out.warnings]
        remaining = n - out.fresh
        if remaining <= 0:
            break
        if rung < max_rung:
            rung += 1
            continue
        break
    if remaining > 0:
        out.exhausted = True
        out.stopped = (f"Found {out.fresh} new of {n}; {out.covered_skipped} candidates were already covered. "
                       f"The pool for this brief is used up" +
                       ("" if max_rung >= 3 else "; allow wider search with --widen topics or --widen geo") + ".")
        if recycle != "never":
            out = _recycle_step(deps, sid, n, out, recycle, confirm_recycle, progress)
    return out


def _recycle_step(deps: Deps, sid: str, n: int, out: BatchOutcome, recycle: str,
                  confirm: Optional[Callable[[Dict[str, Any]], bool]], progress: Callable[[str], None]) -> BatchOutcome:
    """After the ladder: offer earlier undecided / standby people if new matches are very few (pipeline.plan_recycle).
    They get a campaign of their own so the review shows one clean batch."""
    store = deps.store
    spec = batch_spec(store, sid, n, out.rung_reached, recycle=recycle)
    scope = store.scope_for(spec)
    funnel: Dict[str, Any] = {}
    rows = plan_recycle(spec, scope, store, out.fresh, funnel, out.warnings, confirm, exclude=out.campaigns)
    if not rows:
        return out
    store.save_campaign(spec, "done", funnel)
    scope = store.ensure_series(spec)  # only now is there a batch to register
    moved = store.recycle_into(spec.id, scope, rows)
    funnel["recycled"] = moved
    result = RunResult(spec=spec, shortlist=[r["contact"] for r in rows], funnel=funnel)
    funnel["supply"] = {"requested": n, "fresh": 0, "recycled": len(moved), "covered_skipped": 0, "fresh_yield": 1.0,
                        "exhausted": True}
    store.save_campaign(spec, "done", funnel)
    out.campaigns.append(spec.id)
    out.results.append(result)
    out.recycled += len(moved)
    progress(f"batch: {len(moved)} earlier suggestions offered again")
    return out


def until(deps: Deps, ref: str, total: int, *, batch_size: int = 20, budget_searches: Optional[int] = None,
          progress: Callable[[str], None] = lambda m: None, **kwargs: Any) -> List[BatchOutcome]:
    """Batches until `total` people are delivered, the pool is used up, a budget cap is reached or the series pauses.
    Every email still waits for the user's approval; this only fills the review queue."""
    cap = budget_searches if budget_searches is not None else deps.settings.max_search_calls * 4
    done: List[BatchOutcome] = []
    delivered = spent = 0
    while delivered < total:
        n = min(batch_size, total - delivered)
        try:
            b = more(deps, ref, n, progress=progress, **kwargs)
        except SeriesPaused as e:
            done.append(BatchOutcome(series_id=ref, requested=n, stopped=f"Paused: {e}"))
            break
        done.append(b)
        delivered += b.delivered
        spent += b.search_calls
        if b.exhausted:
            break
        if spent >= cap:
            b.stopped = f"Stopped: {spent} search calls spent (cap {cap}). Raise --budget-searches to continue."
            break
    return done

"""
The campaign pipeline.

    discover -> rank -> enrich -> judge -> draft

Each stage is checkpointed, so a crashed or interrupted run resumes from the last finished stage
without repeating network or LLM spend. Every stage reports into a funnel, which is how
improvements (or regressions) to the brain are measured.

Where the LLM is used, and why it cannot be avoided there:
    campaign spec   1 call   understanding a free-text request
    sender profile  1 call   curating a CV (cached by content, so once per CV)
    extraction      0-15     only for promising pages the heuristics could not read
    judge           ~K/8     nuanced fit between this sender and this person (batched)
    draft           1 each   the writing itself (plus at most one retry on a failed check)
Everything else — discovery, page reading, email finding, dedupe, filtering, pre-ranking — is free.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from brain.cache import Cache
from brain.context import CampaignContext
from brain.orgs import OrgEnricher
from brain.searchplan import SearchPlanner, search_budget
from brain.tools import TOOLS, ToolRun
from brain.draft import draft_email, verify_and_fix
from brain.enrich import Enricher, has_usable_email
from brain.fetch import Fetcher
from brain.judge import judge
from brain.llm import LLM
from brain.models import CampaignSpec, Contact, Draft
from brain.packs import get_pack
from brain.rank import diversify, filter_contacts, prescore
from brain.resolve import resolve
from brain.settings import Settings
from brain.sources.csvlist import load_csv
from brain.sources.openalex import OpenAlexSource
from brain.sources.orcid import Orcid
from brain.sources.websearch import WebSearch
from brain.store import Store

log = logging.getLogger("brain.pipeline")

STAGES = ["discover", "rank", "enrich", "judge", "draft"]
MIN_JUDGED_SCORE = 0.5  # below this the judge considers the fit loose at best


@dataclass
class Deps:
    settings: Settings
    cache: Cache
    llm: LLM
    fetcher: Fetcher
    search: WebSearch
    orcid: Orcid
    openalex: OpenAlexSource
    store: Store


def build_deps(settings: Optional[Settings] = None) -> Deps:
    settings = settings or Settings()
    cache = Cache(settings.cache_path)
    return Deps(
        settings=settings, cache=cache, llm=LLM(settings, cache), fetcher=Fetcher(settings, cache),
        search=WebSearch(settings, cache), orcid=Orcid(settings, cache),
        openalex=OpenAlexSource(settings, cache), store=Store(settings.database_url),
    )


@dataclass
class RunResult:
    spec: CampaignSpec
    shortlist: List[Contact] = field(default_factory=list)
    drafts: Dict[str, Draft] = field(default_factory=dict)
    funnel: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def run_campaign(spec: CampaignSpec, deps: Deps, *, upto: str = "draft", resume: bool = True,
                 csv_path: Optional[str] = None, progress: Callable[[str], None] = lambda m: None,
                 ctx: Optional[CampaignContext] = None, seed_contacts: Optional[List[Contact]] = None,
                 only_seeds: bool = False) -> RunResult:
    """ctx: the campaign context from the intake (loaded from the store when resuming).
    seed_contacts: people/organisations from the user's uploaded lists, always considered.
    only_seeds: skip discovery tools and work only on the user's list."""
    if upto not in STAGES:
        raise ValueError(f"upto must be one of {STAGES}")
    started = time.time()
    st, store, pack = deps.settings, deps.store, get_pack(spec.pack)
    if ctx is None:
        ctx = store.load_context(spec.id) if resume else CampaignContext()
    store.save_context(spec.id, ctx)
    # A per-campaign Serper budget sized to the target, never above the configured ceiling.
    total_budget = search_budget(spec.target_count, st.max_search_calls)
    budget_start = deps.search.calls
    deps.search.budget = budget_start + total_budget

    def blocked(c: Contact) -> Optional[str]:
        return ctx.is_excluded(c.name, [e.address for e in c.emails], c.org_domain or "") or store.blocked_reason(c)
    result = RunResult(spec=spec)
    funnel: Dict[str, Any] = (store.load_campaign(spec.id) or {}).get("funnel", {}) if resume else {}
    store.save_campaign(spec, "running", funnel)
    last = STAGES.index(upto)

    def saved(stage: str) -> Optional[Dict[str, Any]]:
        return store.load_checkpoint(spec.id, stage) if resume else None

    def finish(status: str = "done") -> RunResult:
        funnel["cost"] = {
            "llm": deps.llm.report(), "search_calls": deps.search.calls,
            "openalex_requests": getattr(deps.openalex, "requests", 0), "pages_fetched": deps.fetcher.fetched,
            "seconds": round(time.time() - started, 1),
        }
        funnel["cost"]["search_by_kind"] = dict(deps.search.by_kind)
        result.funnel = funnel
        store.save_campaign(spec, status, funnel)
        store.save_context(spec.id, ctx)
        return result

    # ── 1. discover: the playbook's tools, plus the user's own lists ──────────
    cp = saved("discover")
    if cp:
        contacts = cp["contacts"]
        progress(f"discover: resumed ({len(contacts)} candidates)")
    else:
        raw: List[Contact] = list(seed_contacts or [])
        by_source: Dict[str, int] = {"your_lists": len(raw)} if raw else {}
        if csv_path:
            found = load_csv(csv_path)
            by_source["csv"] = len(found)
            raw += found
        elif not only_seeds:
            planner = SearchPlanner(spec, pack, ctx, deps.llm)
            # Discovery may spend 60% of the searches; the rest is kept for finding contact details.
            deps.search.budget = budget_start + int(total_budget * 0.6)
            for name in pack.sources:
                tool = TOOLS.get(name)
                if tool is None:
                    result.warnings.append(f"Unknown tool '{name}' in playbook {pack.name}; skipped.")
                    continue
                if tool.needs_search and not deps.search.available:
                    result.warnings.append(f"Tool '{name}' needs SERPER_API_KEY (or TAVILY_API_KEY); skipped.")
                    continue
                if tool.needs_search and deps.search.remaining <= 2:
                    result.warnings.append(f"Search budget used up before tool '{name}'.")
                    continue
                progress(f"discover: {tool.description}")
                try:
                    found, stats = tool.fn(ToolRun(spec, pack, ctx, deps, planner))
                except Exception as e:  # one failing tool must not sink the campaign
                    log.warning("tool %s failed: %s", name, e)
                    result.warnings.append(f"Tool '{name}' failed: {e}")
                    continue
                by_source[name] = len(found)
                funnel[f"tool_{name}"] = stats
                if stats.get("failed_terms"):
                    result.warnings.append("Publication search failed for: " + ", ".join(stats["failed_terms"])
                                           + " (re-run to retry).")
                raw += found
        deps.search.budget = budget_start + total_budget
        contacts = resolve(raw)
        funnel.update({"found_raw": len(raw), "by_source": by_source, "unique_people": len(contacts),
                       "search_calls_discovery": deps.search.calls})
        store.checkpoint(spec.id, "discover", contacts)
        store.save_context(spec.id, ctx)
        progress(f"discover: {len(raw)} records → {len(contacts)} unique candidates")
    if not contacts:
        result.warnings.append("Discovery found nobody. Broaden the topics or locations.")
        return finish("empty")
    if last == 0:
        return finish("partial")

    # ── 2. filter + pre-rank ─────────────────────────────────────────────────
    cp = saved("rank")
    if cp:
        ranked = cp["contacts"]
    else:
        kept, dropped = filter_contacts(contacts, spec, pack, is_blocked=blocked,
                                        defer_region=bool(pack.sources_set & {"orgs", "places"}))
        ranked = prescore(kept, spec, pack, backend=st.embeddings)
        funnel.update({"qualified": len(kept), "dropped": dict(dropped)})
        # Keep a wider pool than we enrich: it is what email-pattern learning draws on.
        ranked = ranked[: max(spec.target_count * 12, 150)]
        store.checkpoint(spec.id, "rank", ranked)
        progress(f"rank: {len(kept)} qualify ({sum(dropped.values())} filtered out)")
    if not ranked:
        result.warnings.append(f"Every candidate was filtered out: {funnel.get('dropped')}")
        return finish("empty")
    if last == 1:
        result.shortlist = ranked[: spec.target_count]
        return finish("partial")

    # ── 3. enrich the top of the list only ───────────────────────────────────
    cp = saved("enrich")
    if cp:
        targets = cp["contacts"]
    else:
        n_enrich = min(len(ranked), max(spec.target_count * 4, 30))
        targets = diversify(ranked, n_enrich, max_per_org=st.max_per_org * 4)
        progress(f"enrich: looking up contact details for the top {len(targets)}…")
        # Organisations and people are enriched differently; a campaign can contain both (e.g. a user's list).
        orgs_t = [c for c in targets if c.kind == "org"]
        people_t = [c for c in targets if c.kind != "org"]
        enrich_stats: Dict[str, Any] = {}
        if orgs_t:
            org_enricher = OrgEnricher(deps.fetcher, deps.search, st)
            orgs_t = org_enricher.enrich_all(orgs_t, spec, pack)
            enrich_stats["organisations"] = org_enricher.stats
        if people_t:
            enricher = Enricher(deps.fetcher, deps.search, deps.orcid, st)
            hints = [w for t in spec.topics for w in t.lower().split() if len(w) > 4] + \
                    (["computer", "comput", "cse", "informatics"] if spec.pack == "academic" else [])
            enricher.enrich(people_t, pool=ranked, hint_terms=hints)
            enrich_stats["people"] = enricher.stats
        targets = orgs_t + people_t
        if orgs_t:  # locations are known now that the sites have been read: apply the region rule strictly
            targets, late_drop = filter_contacts(targets, spec, pack)
            funnel["dropped_after_reading_sites"] = dict(late_drop)
        targets = resolve(targets)  # a newly found email can reveal two records are one person
        # Addresses are known now, so the do-not-contact check can be exact.
        targets = [c for c in targets if not blocked(c)]
        inbox_share = sum(1 for c in targets if c.best_email and c.best_email.method == "inbox")
        if targets and inbox_share > len(targets) / 2:
            ctx.note("most organisations publish only a shared inbox, so many emails go to the team rather than a person")
        targets = prescore(targets, spec, pack, backend=st.embeddings)  # pages may have added context
        funnel.update({"enriched": len(targets), "enrichment": enrich_stats})
        store.checkpoint(spec.id, "enrich", targets)
    reachable = [c for c in targets if has_usable_email(c, st.min_email_confidence)]
    funnel["reachable"] = len(reachable)
    funnel["guessed_email_only"] = sum(1 for c in targets if c.emails and c not in reachable)
    progress(f"enrich: {len(reachable)} of {len(targets)} have a confident email")
    if not reachable:
        result.warnings.append("No confident email found for any top candidate.")
        result.shortlist = targets[: spec.target_count]
        return finish("no_emails")
    if last == 2:
        result.shortlist = reachable[: spec.target_count]
        return finish("partial")

    # ── 4. judge (batched LLM) ───────────────────────────────────────────────
    cp = saved("judge")
    if cp and (cp["extra"].get("unjudged") or not cp["contacts"]):
        cp = None  # judging failed last time (e.g. rate limit): redo it rather than resume an empty result
    if cp:
        shortlist = cp["contacts"]
    else:
        to_judge = reachable[: spec.target_count * 2]
        if deps.llm.available:
            progress(f"judge: assessing fit for {len(to_judge)} people…")
            n = judge(to_judge, spec, deps.llm, batch_size=st.judge_batch_size, ctx=ctx, playbook=pack)
            funnel["judged"] = n
            scored = [c for c in to_judge if c.score is not None and c.score >= MIN_JUDGED_SCORE]
            funnel["passed_judge"] = len(scored)
            unjudged = [c for c in to_judge if c.score is None]
            if unjudged:
                result.warnings.append(f"{len(unjudged)} candidates could not be judged and were left out.")
        else:
            result.warnings.append("No LLM key: ranking used the lexical pre-score only and no emails were drafted.")
            for c in to_judge:
                c.score = c.prescore
                c.flags.append("unjudged")
            scored = to_judge
        scored.sort(key=lambda c: c.score or 0, reverse=True)
        shortlist = diversify(scored, spec.target_count, st.max_per_org)
        funnel["shortlisted"] = len(shortlist)
        store.upsert_contacts(shortlist)
        store.checkpoint(spec.id, "judge", shortlist,
                         extra={"unjudged": sum(1 for c in to_judge if "unjudged" in c.flags and deps.llm.available)})
        store.save_results(spec.id, shortlist, {})
    result.shortlist = shortlist
    progress(f"judge: {len(shortlist)} shortlisted")
    if last == 3 or not shortlist or not deps.llm.available:
        return finish("partial" if shortlist else "empty")

    # ── 5. draft ─────────────────────────────────────────────────────────────
    progress(f"draft: writing {len(shortlist)} emails…")
    drafts: Dict[str, Draft] = {}
    failed = 0
    for c in shortlist:
        d = draft_email(c, spec, pack, deps.llm, ctx=ctx)
        if d is None:
            failed += 1
        else:
            drafts[c.id] = d
    progress("draft: fact-checking drafts…")
    funnel["fact_check"] = verify_and_fix(shortlist, drafts, spec, pack, deps.llm, ctx=ctx)
    store.save_results(spec.id, shortlist, drafts)
    result.drafts = drafts
    funnel.update({"drafted_ok": sum(1 for d in drafts.values() if d.ok),
                   "drafts_need_review": sum(1 for d in drafts.values() if not d.ok),
                   "drafts_failed": failed})
    if failed:
        result.warnings.append(f"{failed} drafts could not be generated (LLM error or budget); re-run to retry them.")
    return finish("done")

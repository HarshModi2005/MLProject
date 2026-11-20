"""
The campaign pipeline.

    discover -> rank -> enrich -> judge [-> backfill] -> draft

discover  adaptive exploration (brain/explore.py): the playbook's tools plus any tool the fit criteria
          need, budget moved to whatever finds qualified prospects, expansion from the best finds
rank      hard filters, a homepage-only "light read" of the top organisations, then pre-ranking on topic
          similarity + evidence-based fit to the prospect profile (brain/fit.py)
enrich    contact details; inside organisations, the decision-maker by role
judge     batched LLM check of the confirmed criteria; if too few pass, ONE backfill round searches
          again, seeded with the winners and the judge's rejection reasons
draft     the writing, then a batched fact-check

Each stage is checkpointed, so a crashed or interrupted run resumes from the last finished stage
without repeating network or LLM spend. Every stage reports into a funnel, which is how
improvements (or regressions) to the brain are measured.

Where the LLM is used, and why it cannot be avoided there:
    campaign spec   1 call   understanding a free-text request
    sender profile  1 call   curating a CV (cached by content, so once per CV)
    extraction      0-15     only for promising pages the heuristics could not read
    triage          0-1/query  search hits the heuristics cannot classify (cheap model, batched)
    replan          0-2      new queries from the query log (+ winners and rejections in the backfill)
    judge           ~K/8     nuanced fit between this sender and this person (batched)
    draft           1 each   the writing itself (plus at most one retry on a failed check)
Everything else — discovery, page reading, email finding, dedupe, filtering, pre-ranking — is free.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from brain import touch
from brain.cache import Cache
from brain.context import CampaignContext
from brain.discovery.graph import record as graph_record
from brain.discovery.reach.patterns import learn as learn_patterns, threshold as reach_threshold
from brain.discovery.reach.providers import tag_tech
from brain.discovery.reach.tiers import contact_tier, counts as tier_counts
from brain.discovery.reach.verify import verify_addresses
from brain.discovery.precision.orgtype import gate as org_type_gate
from brain.discovery.precision.prejudge import prejudge, reorder as reorder_screened
from brain.explore import explore
from brain.fit import profile_for
from brain.orgs import OrgEnricher
from brain.searchplan import SearchPlanner, search_budget
from brain.draft import draft_email, verify_and_fix
from brain.enrich import Enricher, has_usable_email
from brain.fetch import Fetcher
from brain.judge import judge
from brain.llm import LLM
from brain.models import CampaignSpec, Contact, Draft
from brain.packs import get_pack
from brain.rank import diversify, filter_contacts, prefer_person, prescore
from brain.resolve import resolve
from brain.settings import Settings
from brain.sources.csvlist import load_csv
from brain.sources.openalex import OpenAlexSource
from brain.sources.orcid import Orcid
from brain.sources.websearch import WebSearch
from brain.store import Store

log = logging.getLogger("brain.pipeline")

STAGES = ["discover", "rank", "enrich", "judge", "draft"]

# Playbook families whose targets are named people inside companies, where free web sources rarely publish an address.
PERSON_AT_COMPANY_FAMILIES = ("funding", "jobs", "media", "selling", "people")


HANDLE_SOURCES = ("npm", "huggingface", "github")  # people known by a username, not a full name


def plausible_person(c: Contact) -> bool:
    """A 'person' must have a person's name (page titles such as "Partner Spotlight" do not), unless the source
    identifies people by handle."""
    if c.kind == "org" or any(s in HANDLE_SOURCES for s in c.sources):
        return True
    from brain.textutil import looks_like_person_name

    return looks_like_person_name(c.name)


def count_sources(contacts: List[Contact]) -> Dict[str, int]:
    """How many of these candidates each source contributed (a candidate found by two sources counts for both)."""
    out: Dict[str, int] = {}
    for c in contacts:
        for s in dict.fromkeys(c.sources or ["unknown"]):
            out[s] = out.get(s, 0) + 1
    return out


def note_prejudge_agreement(funnel: Dict[str, Any], judged: List[Contact]) -> None:
    """Pre-judge vs judge on the candidates that got both. Rejected candidates are never judged, so this only sees
    one side: a high `low_pre_good` (screened low, judged good) means the screen is too strict."""
    st = funnel.setdefault("prejudge_agreement", {"both": 0, "low_pre_good": 0, "high_pre_bad": 0})
    for c in judged:
        if c.prejudge is None or c.score is None:
            continue
        st["both"] += 1
        if c.prejudge < 0.5 and c.score >= 0.7:
            st["low_pre_good"] += 1
        if c.prejudge >= 0.7 and c.score < 0.5:
            st["high_pre_bad"] += 1


def note_sources(funnel: Dict[str, Any], stage: str, contacts: List[Contact], add: bool = False) -> None:
    """funnel["source_funnel"][stage][source] = candidates at that stage; shows which sources survive to the end."""
    sf = funnel.setdefault("source_funnel", {})
    counts = count_sources(contacts)
    if add and stage in sf:
        for k, v in counts.items():
            sf[stage][k] = sf[stage].get(k, 0) + v
    else:
        sf[stage] = counts
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
        openalex=OpenAlexSource(settings, cache), store=Store(settings.database_url, workspace=settings.workspace),
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
                 only_seeds: bool = False,
                 confirm_recycle: Optional[Callable[[Dict[str, Any]], bool]] = None) -> RunResult:
    """ctx: the campaign context from the intake (loaded from the store when resuming).
    seed_contacts: people/organisations from the user's uploaded lists, always considered.
    only_seeds: skip discovery tools and work only on the user's list.
    confirm_recycle: asked when new matches are very few but earlier, undecided people could fill the batch
    (spec.recycle == "ask"); it gets {"fresh", "wanted", "undecided", "standby", "offer"} and returns yes/no."""
    if upto not in STAGES:
        raise ValueError(f"upto must be one of {STAGES}")
    started = time.time()
    st, store, pack = deps.settings, deps.store, get_pack(spec.pack)
    if ctx is None:
        ctx = store.load_context(spec.id) if resume else CampaignContext()
    store.save_context(spec.id, ctx)
    scope = store.ensure_series(spec)  # which earlier batches count as "already covered" (brain/touch.py)
    prior_org = store.org_counts(scope.series_id)  # people per organisation put forward in earlier batches
    org_campaign = (spec.target_kind or pack.target_kind) == "org"
    # A per-campaign Serper budget sized to the target, never above the configured ceiling.
    total_budget = search_budget(spec.target_count, st.max_search_calls)
    reach_min = reach_threshold(st, pack)  # lower when the playbook accepts flagged guesses
    budget_start = deps.search.calls
    deps.search.budget = budget_start + total_budget
    discovery_budget = int(total_budget * 0.55)  # the rest: contact details (~30%) and the backfill (~15%)
    enrich_ceiling = budget_start + int(total_budget * 0.85)

    def blocked(c: Contact) -> Optional[str]:
        return ctx.is_excluded(c.name, [e.address for e in c.emails], c.org_domain or "") or store.blocked_reason(c, scope)
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
        if getattr(deps.fetcher, "paid_calls", None):
            funnel["cost"]["paid_calls"] = dict(deps.fetcher.paid_calls)  # paid data providers (exa, hunter, builtwith)
        misses = getattr(deps.search, "replay_misses", 0) + getattr(deps.fetcher, "replay_misses", 0)
        if misses:
            funnel["cost"]["replay_misses"] = misses
        if funnel.get("source_funnel") and not funnel.get("source_stats_recorded") and status not in ("running", "empty"):
            try:  # which sources paid off: the next campaign of this playbook starts from it (precision/priors.py)
                store.record_source_stats(pack.name, funnel["source_funnel"], funnel.get("ledger") or {})
                funnel["source_stats_recorded"] = True
            except Exception as e:
                log.warning("could not record source stats: %s", e)
        if ctx.query_log and not funnel.get("query_stats_recorded") and status != "running":
            try:
                store.record_query_stats(pack.name, spec, ctx.query_log)
                funnel["query_stats_recorded"] = True
            except Exception as e:  # learning is a bonus; never fail a campaign over it
                log.warning("could not record query stats: %s", e)
        funnel["dedup"] = dedup_summary(funnel)
        funnel["supply"] = supply_summary(spec, funnel, result)
        result.funnel = funnel
        store.save_campaign(spec, status, funnel)
        store.save_context(spec.id, ctx)
        return result

    planner = SearchPlanner(spec, pack, ctx, deps.llm,
                            learned_templates=store.learned_templates(pack.name) if st.use_memory else None)

    # ── stage helpers (shared by the main pass and the backfill) ─────────────
    def rank_stage(contacts: List[Contact]) -> List[Contact]:
        kept, dropped = filter_contacts(contacts, spec, pack, is_blocked=blocked,
                                        defer_region=any(c.kind == "org" for c in contacts))
        if st.warm and hasattr(store, "network_size") and store.network_size():
            from brain.discovery.warm.paths import Network, annotate

            w = annotate(kept, Network(store.network()))  # people the user knows, or can be introduced to
            for k, v in w.items():
                funnel.setdefault("warm_paths", {})[k] = funnel.get("warm_paths", {}).get(k, 0) + v
        ranked = prescore(kept, spec, pack, backend=st.embeddings)
        orgs = [c for c in ranked if c.kind == "org"]
        if orgs and st.light_read_factor > 0:
            # Read the top organisations' homepages before ranking them for real: a snippet is not enough.
            n = OrgEnricher(deps.fetcher, deps.search, st).light_read(
                orgs, spec, limit=min(80, max(15, spec.target_count * st.light_read_factor)))
            funnel["light_read"] = funnel.get("light_read", 0) + n
            ranked = prescore(ranked, spec, pack, backend=st.embeddings)
        tagged = tag_tech([c for c in ranked if c.kind == "org"], profile_for(spec, pack), deps.fetcher)
        if tagged:  # paid (BUILTWITH_API_KEY), only when a criterion needs proof of a technology
            funnel["uses_tech_tagged"] = funnel.get("uses_tech_tagged", 0) + tagged
            ranked = prescore(ranked, spec, pack, backend=st.embeddings)
        if st.org_type_gate:  # demote kinds of organisation the campaign does not target; drop directories
            ranked, typed = org_type_gate(ranked, spec, pack)
            for k, v in typed.items():
                funnel.setdefault("org_types", {})[k] = funnel.get("org_types", {}).get(k, 0) + v
            dropped["directory"] = dropped.get("directory", 0) + typed.get("dropped_directory", 0)
            ranked = prescore(ranked, spec, pack, backend=st.embeddings)
        if st.prejudge and deps.llm.available:  # cheap screen of the top before any enrichment spend
            limit = st.prejudge_limit or max(30, spec.target_count * 8)
            head = [c for c in ranked if not (c.kind == "org" and not c.org_domain and not c.emails)][:limit]
            prejudge(head, spec, deps.llm, prof=profile_for(spec, pack), min_score=st.prejudge_min,
                     stats=funnel.setdefault("prejudge", {}), ctx=ctx)
            ranked = reorder_screened(ranked)
        funnel["qualified"] = funnel.get("qualified", 0) + len(ranked)
        merged = dict(funnel.get("dropped") or {})
        for k, v in dropped.items():
            merged[k] = merged.get(k, 0) + v
        funnel["dropped"] = merged
        return ranked

    def enrich_stage(targets: List[Contact], pool: List[Contact]) -> List[Contact]:
        deps.search.budget = enrich_ceiling
        orgs_t = [c for c in targets if c.kind == "org"]
        people_t = [c for c in targets if c.kind != "org"]
        enrich_stats: Dict[str, Any] = dict(funnel.get("enrichment") or {})
        if orgs_t:
            org_enricher = OrgEnricher(deps.fetcher, deps.search, st, find_people=True,
                                       max_people_searches=st.max_people_searches, store=store,
                                       accept_prior=pack.accept_guessed)
            orgs_t = org_enricher.enrich_all(orgs_t, spec, pack)
            enrich_stats["organisations"] = org_enricher.stats
        if people_t:
            enricher = Enricher(deps.fetcher, deps.search, deps.orcid, st, store=store, accept_prior=pack.accept_guessed)
            hints = [w for t in spec.topics for w in t.lower().split() if len(w) > 4] + \
                    (["computer", "comput", "cse", "informatics"] if spec.pack == "academic" else [])
            enricher.enrich(people_t, pool=pool, hint_terms=hints)
            enrich_stats["people"] = enricher.stats
        out = orgs_t + people_t
        if orgs_t or any("region_unknown" in c.flags for c in out):
            # Locations may be known now (sites read, profile pages fetched): apply the region rule again. Orgs are
            # judged strictly; people whose city is still not stated stay (flagged), people now known elsewhere go.
            out, late_drop = filter_contacts(out, spec, pack)
            merged = dict(funnel.get("dropped_after_reading_sites") or {})
            for k, v in late_drop.items():
                merged[k] = merged.get(k, 0) + v
            funnel["dropped_after_reading_sites"] = merged
        out = resolve(out)  # a newly found email can reveal two records are one person
        # Addresses are known now, so the do-not-contact check can be exact.
        keep_out = []
        for c in out:
            why = blocked(c)
            if why:
                late = funnel.setdefault("dropped_after_enrich", {})
                late[why] = late.get(why, 0) + 1
            else:
                keep_out.append(c)
        out = keep_out
        # Enrichment creates new candidates (decision-makers, people listed on sites): they must look like real people
        # and pass the same cheap screen as everyone else before they can reach the judge.
        implausible = [c for c in out if not plausible_person(c)]
        if implausible:
            funnel["dropped_not_a_person"] = funnel.get("dropped_not_a_person", 0) + len(implausible)
            out = [c for c in out if plausible_person(c)]
        if st.prejudge and deps.llm.available:
            fresh = [c for c in out if c.prejudge is None][:60]
            if fresh:
                prejudge(fresh, spec, deps.llm, prof=profile_for(spec, pack), min_score=st.prejudge_min,
                         stats=funnel.setdefault("prejudge_after_enrich", {}), ctx=ctx)
                out = [c for c in out if "prejudge_reject" not in c.flags]
        inbox_share = sum(1 for c in out if c.best_email and c.best_email.method == "inbox")
        if out and inbox_share > len(out) / 2:
            ctx.note("most organisations publish only a shared inbox, so many emails go to the team rather than a person")
        out = reorder_screened(prescore(out, spec, pack, backend=st.embeddings))  # pages may have added context
        funnel["enrichment"] = enrich_stats
        return out

    from brain import calibrate

    calibrator = calibrate.load(store)  # fitted on human labels, kept only if it beat the raw score (brain/calibrate.py)
    if calibrator is not None:
        funnel["calibrated_on_labels"] = calibrator.n
    judged_log: List[Dict[str, Any]] = []  # every judged candidate's signals: a human label turns each into training data

    def judge_stage(to_judge: List[Contact]) -> List[Contact]:
        n = judge(to_judge, spec, deps.llm, batch_size=st.judge_batch_size, ctx=ctx, playbook=pack,
                  prescore_weight=st.judge_prescore_weight, calibrator=calibrator)
        judged_log.extend(calibrate.row(c) for c in to_judge if c.score is not None)
        funnel["judged"] = funnel.get("judged", 0) + n
        note_prejudge_agreement(funnel, to_judge)
        passed = [c for c in to_judge if c.score is not None and c.score >= st.shortlist_floor]
        for c in to_judge:  # negative feedback for any later search
            if c.score is not None and c.score < MIN_JUDGED_SCORE and c.reasons and len(ctx.rejected_reasons) < 20:
                ctx.rejected_reasons.append(f"{c.name}: {c.reasons[0]}")
        unjudged = [c for c in to_judge if c.score is None]
        if unjudged:
            result.warnings.append(f"{len(unjudged)} candidates could not be judged and were left out.")
        return passed

    # ── 1. discover ──────────────────────────────────────────────────────────
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
            contacts = resolve(raw)
        elif only_seeds:
            contacts = resolve(raw)
        else:
            deps.search.budget = budget_start + discovery_budget
            ex = explore(spec, pack, ctx, deps, planner, seeds=raw, budget=discovery_budget,
                         rounds=st.explore_rounds, progress=progress, blocked=blocked)
            for why, n in ex.skipped.items():
                d = funnel.setdefault("dedup_discovery", {})
                d[why] = d.get(why, 0) + n
            contacts = ex.contacts
            by_source.update({k: v for k, v in ex.by_source.items() if k != "your_lists"})
            for name, stats in ex.tool_stats.items():
                funnel[f"tool_{name}"] = stats
            funnel["explore"] = {"rounds": ex.rounds, "qualified": ex.qualified, "fit": ex.fit,
                                 "tools": {k: dict(v) for k, v in ctx.tool_stats.items()}}
            funnel["ledger"] = ex.ledger
            result.warnings += ex.warnings
        deps.search.budget = budget_start + total_budget
        funnel.update({"found_raw": sum(by_source.values()), "by_source": by_source, "unique_people": len(contacts),
                       "search_calls_discovery": deps.search.calls - budget_start})
        note_sources(funnel, "unique", contacts)
        store.checkpoint(spec.id, "discover", contacts)
        store.save_context(spec.id, ctx)
        progress(f"discover: {funnel['found_raw']} records → {len(contacts)} unique candidates")
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
        funnel.pop("qualified", None)
        funnel.pop("dropped", None)
        ranked = rank_stage(contacts)
        # Keep a wider pool than we enrich: it is what email-pattern learning draws on.
        note_sources(funnel, "qualified", ranked)
        ranked = ranked[: max(spec.target_count * 12, 150)]
        store.checkpoint(spec.id, "rank", ranked)
        progress(f"rank: {funnel['qualified']} qualify ({sum(funnel['dropped'].values())} filtered out)")
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
        # Prefer candidates that can be read and reached: an organisation without a website is a last resort.
        unreadable = lambda c: c.kind == "org" and not c.org_domain and not c.emails  # noqa: E731
        order = [c for c in ranked if not unreadable(c)] + [c for c in ranked if unreadable(c)]
        order = [c for c in order if "prejudge_reject" not in c.flags]  # screened out: not worth the lookups
        targets = diversify(order, n_enrich, max_per_org=st.max_per_org * 4, prior=prior_org,
                            prior_cap=1 if org_campaign else None)
        # Remember who was tried BEFORE enriching: enrichment can drop a candidate or turn an organisation into
        # the person found inside it (a new id), and neither may be enriched twice.
        done = {c.id for c in targets} | {c.org_domain for c in targets if c.org_domain}
        progress(f"enrich: looking up contact details for the top {len(targets)}…")
        targets = enrich_stage(targets, ranked)
        # Many organisations publish no address. While too few are reachable, enrich the next candidates too
        # (page fetches, plus decision-maker searches only while budget remains).
        waves = 1
        while waves < 3 and sum(has_usable_email(c, reach_min) for c in targets) < spec.target_count * 2:
            rest = [c for c in order if c.id not in done and not (c.org_domain and c.org_domain in done)]
            if not rest:
                break
            batch = diversify(rest, min(len(rest), max(spec.target_count * 3, 20)), max_per_org=st.max_per_org * 4)
            progress(f"enrich: only {sum(has_usable_email(c, reach_min) for c in targets)} reachable; "
                     f"trying the next {len(batch)}…")
            done |= {c.id for c in batch} | {c.org_domain for c in batch if c.org_domain}
            targets = reorder_screened(prescore(resolve(targets + enrich_stage(batch, ranked)), spec, pack,
                                                backend=st.embeddings))
            waves += 1
        funnel["enrich_waves"] = waves
        funnel["enriched"] = len(targets)
        store.checkpoint(spec.id, "enrich", targets)
    reachable = [c for c in targets if has_usable_email(c, reach_min)]
    funnel["reachable"] = len(reachable)
    funnel["graph_edges"] = graph_record(store, targets)  # decision-makers and sites read add edges
    funnel["email_conventions_learned"] = learn_patterns(store, targets)
    note_sources(funnel, "enriched", targets)
    note_sources(funnel, "reachable", reachable)
    funnel["guessed_email_only"] = sum(1 for c in targets if c.emails and c not in reachable)
    progress(f"enrich: {len(reachable)} of {len(targets)} have a confident email")
    from brain.discovery.reach.providers import hunter_available

    funnel["contact_provider"] = "hunter" if hunter_available() else "none"
    if len(reachable) < spec.target_count and pack.family in PERSON_AT_COMPANY_FAMILIES and not hunter_available():
        # Said plainly, because it is structural: a named person's address inside a company is rarely published.
        result.warnings.append(
            f"Only {len(reachable)} of {spec.target_count} wanted contacts have a confident address. This kind of campaign "
            f"({pack.title or pack.name}) needs a contact-data provider to reach named people inside companies: set "
            f"HUNTER_API_KEY. Without one, expect mostly shared inboxes and a short list.")
    if reachable:
        try:  # verified, reachable prospects become memory for later campaigns
            store.upsert_contacts(reachable)
        except Exception as e:
            log.warning("could not save prospects to memory: %s", e)
    if not reachable:
        result.warnings.append("No confident email found for any top candidate.")
        result.shortlist = targets[: spec.target_count]
        return finish("no_emails")
    if last == 2:
        result.shortlist = reachable[: spec.target_count]
        return finish("partial")

    # ── 4. judge (batched LLM), then at most one backfill round ──────────────
    cp = saved("judge")
    if cp and (cp["extra"].get("unjudged") or not cp["contacts"]):
        cp = None  # judging failed last time (e.g. rate limit): redo it rather than resume an empty result
    if cp:
        shortlist = cp["contacts"]
        if (cp.get("extra") or {}).get("recycled"):
            funnel["recycled"] = cp["extra"]["recycled"]
    else:
        to_judge = reachable[: spec.target_count * 3]  # a wider pool for the judge: it, not the pre-score, decides
        # Every address is checked before it can be shortlisted (MX always; SMTP / Hunter when enabled): undeliverable
        # ones are dropped, the verdict is kept on the address and decides its tier.
        checked = verify_addresses(to_judge, deps.fetcher, limit=len(to_judge))
        if checked["checked"]:
            funnel["address_verification"] = checked
            to_judge = [c for c in to_judge if has_usable_email(c, reach_min)]
        if deps.llm.available:
            progress(f"judge: assessing fit for {len(to_judge)} people…")
            scored = judge_stage(to_judge)
            funnel["passed_judge"] = len(scored)
            if st.backfill and len(scored) < spec.target_count and not only_seeds and not csv_path:
                scored += backfill(spec, pack, ctx, deps, planner, scored, known=ranked + targets,
                                   rank_stage=rank_stage, enrich_stage=enrich_stage, judge_stage=judge_stage,
                                   funnel=funnel, progress=progress, ceiling=budget_start + total_budget, blocked=blocked)
                funnel["passed_judge"] = len(scored)
        else:
            result.warnings.append("No LLM key: ranking used the lexical pre-score only and no emails were drafted.")
            for c in to_judge:
                c.score = c.prescore
                c.flags.append("unjudged")
            scored = to_judge
        if st.learned_fit:
            funnel["fit_model"] = learn_fit(spec, pack, store, ranked + targets + scored)
        scored.sort(key=lambda c: c.score or 0, reverse=True)
        seen_addr: set = set()
        unique = []
        for c in scored:  # two branches sharing one inbox are one recipient
            addr = c.best_email.address if c.best_email else None
            if addr and addr in seen_addr:
                continue
            seen_addr.add(addr)
            unique.append(c)
        before_pref = len(unique)
        unique = prefer_person(unique)
        if len(unique) < before_pref:
            funnel["dropped_inbox_for_person"] = before_pref - len(unique)
        capped = diversify(unique, spec.target_count, st.max_per_org, prior=prior_org, prior_cap=1 if org_campaign else None)
        if prior_org:
            funnel["dedup_org_cap"] = sum(1 for c in unique if c not in capped and prior_org.get(touch.org_key(c), 0)
                                          >= (1 if org_campaign else st.max_per_org))
        # Take these people for this batch, atomically: a person another batch (running at the same time) claimed first is
        # dropped here, so two batches can never both put the same person in front of the user.
        shortlist, lost = store.claim_batch(spec.id, scope, capped)
        if lost:
            funnel["dedup_claim_lost"] = len(lost)
        recycled_rows = plan_recycle(spec, scope, store, len(shortlist), funnel, result.warnings, confirm_recycle)
        moved = store.recycle_into(spec.id, scope, recycled_rows) if recycled_rows else []
        moved_ids = {m["contact_id"] for m in moved}
        recycled_rows = [r for r in recycled_rows if r["contact_id"] in moved_ids]  # one decided meanwhile is not offered
        for r in recycled_rows:  # undecided / standby people from earlier batches: kept score and draft, flagged
            rc = r["contact"]
            rc.score = rc.score if rc.score is not None else r["score"]
            if "previously_shown" not in rc.flags:
                rc.flags.append("previously_shown")
        funnel["shortlisted"] = len(shortlist) + len(recycled_rows)
        funnel["address_tiers"] = tier_counts(shortlist + [r["contact"] for r in recycled_rows])
        funnel["coverage"] = coverage(spec.target_count, shortlist + [r["contact"] for r in recycled_rows], st.shortlist_floor,
                                      covered=covered_count(funnel), recycled=len(recycled_rows))
        if funnel["coverage"]["short_by"]:
            result.warnings.append(funnel["coverage"]["message"])
        note_sources(funnel, "passed_judge", [c for c in scored if c.score is not None and c.score >= st.shortlist_floor])
        note_sources(funnel, "shortlisted", shortlist)
        store.upsert_contacts(shortlist)
        store.save_results(spec.id, shortlist, {})
        funnel["recycled"] = moved
        shortlist = shortlist + [r["contact"] for r in recycled_rows]
        store.checkpoint(spec.id, "judge", shortlist,
                         extra={"unjudged": sum(1 for c in to_judge if "unjudged" in c.flags and deps.llm.available),
                                "judged": judged_log, "recycled": moved})
    recycled_ids = {m["contact_id"] for m in funnel.get("recycled") or []}
    result.shortlist = shortlist
    progress(f"judge: {len(shortlist)} shortlisted")
    if last == 3 or not shortlist or not deps.llm.available:
        return finish("partial" if shortlist else "empty")

    # ── 5. draft ─────────────────────────────────────────────────────────────
    if hasattr(store, "style_examples"):  # the user's own edited emails teach the writer their style
        ctx.style_examples = list(dict.fromkeys(store.style_examples() + list(ctx.style_examples)))[:3]
    fresh_list = [c for c in shortlist if c.id not in recycled_ids]  # recycled people already have their draft
    progress(f"draft: writing {len(fresh_list)} emails…")
    drafts: Dict[str, Draft] = {}
    failed = 0
    for c in fresh_list:
        d = draft_email(c, spec, pack, deps.llm, ctx=ctx)
        if d is None:
            failed += 1
        else:
            drafts[c.id] = d
    progress("draft: fact-checking drafts…")
    funnel["fact_check"] = verify_and_fix(fresh_list, drafts, spec, pack, deps.llm, ctx=ctx)
    for c in fresh_list:  # a guessed address is never sent without a human looking at it
        best = c.best_email
        d = drafts.get(c.id)
        if d is None:
            continue
        d.address_tier = contact_tier(c)
        if best and best.method in ("pattern", "pattern_confirmed", "pattern_global", "pattern_prior") and d.address_tier != "verified":
            d.issues.append(f"recipient address {best.address} is a guess from the organisation's usual "
                            f"email format ({best.method}); verify it before sending")
        elif d.address_tier == "unattributed":
            d.issues.append(f"nothing ties {best.address if best else 'the address'} to {c.name}; check it is theirs before sending")
    store.save_results(spec.id, fresh_list, drafts)
    result.drafts = dict(drafts)
    for r in store.results(spec.id):  # the stored drafts of recycled people are part of this batch's review
        if r["contact_id"] in recycled_ids and r["body"] and r["email"]:
            result.drafts[r["contact_id"]] = Draft(contact_id=r["contact_id"], to=r["email"], subject=r["subject"] or "",
                                                   body=r["body"], issues=r["issues"])
    funnel.update({"drafted_ok": sum(1 for d in drafts.values() if d.ok),
                   "drafts_need_review": sum(1 for d in drafts.values() if not d.ok),
                   "drafts_failed": failed})
    if failed:
        result.warnings.append(f"{failed} drafts could not be generated (LLM error or budget); re-run to retry them.")
    return finish("done")


COVERED = ("already_surfaced", "already_contacted", "rejected_earlier", "on_standby")


def dedup_summary(funnel: Dict[str, Any]) -> Dict[str, int]:
    """Who was skipped because an earlier batch (or a send, a rejection, a standby) already covered them, by reason and
    across every stage: discovery, ranking, after enrichment, the per-organisation cap and the batch claim."""
    out: Dict[str, int] = {}
    for part in (funnel.get("dedup_discovery"), funnel.get("dropped"), funnel.get("dropped_after_enrich")):
        for k, v in (part or {}).items():
            if k in COVERED or k in ("do_not_contact", "suppressed"):
                out[k] = out.get(k, 0) + int(v)
    if funnel.get("dedup_org_cap"):
        out["org_cap"] = int(funnel["dedup_org_cap"])
    if funnel.get("dedup_claim_lost"):
        out["claim_lost"] = int(funnel["dedup_claim_lost"])
    return out


def covered_count(funnel: Dict[str, Any]) -> int:
    d = dedup_summary(funnel)
    return sum(d.get(k, 0) for k in COVERED) + d.get("claim_lost", 0)


def supply_summary(spec: CampaignSpec, funnel: Dict[str, Any], result: "RunResult") -> Dict[str, Any]:
    """What this batch delivered against what was asked: new people, recycled people, how much was already covered, and
    the fresh yield (qualified candidates that were new / all qualified candidates found). `exhausted` means the pool
    for this brief could not fill the batch with NEW people."""
    recycled = len(funnel.get("recycled") or [])
    fresh = max(0, len(result.shortlist) - recycled)
    covered = covered_count(funnel)
    qualified = int(funnel.get("qualified", 0))
    return {"requested": spec.target_count, "fresh": fresh, "recycled": recycled, "covered_skipped": covered,
            "fresh_yield": round(qualified / (qualified + covered), 3) if (qualified + covered) else 1.0,
            "exhausted": fresh < spec.target_count}


def plan_recycle(spec: CampaignSpec, scope: touch.Scope, store: Store, fresh_n: int, funnel: Dict[str, Any],
                 warnings: List[str], confirm: Optional[Callable[[Dict[str, Any]], bool]],
                 exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """When NEW matches are very few (fewer than max(3, wanted/2)) and the user wants more, earlier people who passed
    the fit floor but were never decided on, or were put on standby, may be offered again, best score first. Never
    rejected people, never anyone already emailed. Their stored score and draft are reused: no new LLM calls."""
    wanted = spec.target_count
    threshold = min(wanted, max(3, math.ceil(wanted / 2)))
    if fresh_n >= threshold or spec.recycle == "never" or fresh_n >= wanted:
        return []
    pool = []
    for r in store.soft_pool(scope, exclude_campaign=[spec.id, *exclude]):  # `exclude`: batches of this same call
        if store.blocked_reason(r["contact"], None) is None:  # a send or suppression since then still wins
            pool.append(r)
    if not pool:
        return []
    offer = pool[: wanted - fresh_n]
    info = {"fresh": fresh_n, "wanted": wanted, "undecided": sum(1 for r in pool if r["status"] != "standby"),
            "standby": sum(1 for r in pool if r["status"] == "standby"), "offer": len(offer)}
    mode = spec.recycle
    yes = mode == "auto" or (mode == "ask" and confirm is not None and bool(confirm(info)))
    funnel["recycle_offer"] = {**info, "accepted": yes}
    if not yes:
        if mode == "ask" and confirm is None:
            warnings.append(f"Only {fresh_n} new of {wanted}. {info['undecided']} earlier suggestions are still undecided and "
                            f"{info['standby']} are on standby: run again with --recycle auto to include them.")
        return []
    return offer


def coverage(wanted: int, shortlist: List[Contact], floor: float, covered: int = 0, recycled: int = 0) -> Dict[str, Any]:
    """What the run actually delivered against what was asked for, stated plainly. A short list is a correct answer
    when the alternative is padding it with weak fits or addresses nothing ties to the person."""
    tiers = tier_counts(shortlist)
    sendable = tiers.get("verified", 0)
    short_by = max(0, wanted - len(shortlist))
    msg = ""
    if short_by:
        msg = (f"Found {len(shortlist)} confident prospect{'s' if len(shortlist) != 1 else ''} of the {wanted} wanted: "
               f"the list stops at fit {floor:g} instead of filling it with weaker matches. Broaden the topics or "
               f"locations, or add a contact-data provider, to find more.")
    if covered and short_by:
        msg += f" {covered} candidates were skipped because earlier batches already covered them."
    return {"wanted": wanted, "delivered": len(shortlist), "short_by": short_by, "floor": floor,
            "verified_address": sendable, "needs_address_check": len(shortlist) - sendable, "message": msg,
            "already_covered": covered, "recycled": recycled}


def learn_fit(spec: CampaignSpec, pack, store: Store, contacts: List[Contact]) -> Dict[str, Any]:
    """Store this campaign's (pre-)judge verdicts as training examples and refit the playbook's fit model
    (brain/discovery/precision/fitmodel.py). Learning is a bonus: it never fails a campaign."""
    from brain.discovery.precision.fitmodel import examples_from, train

    try:
        best: Dict[str, Contact] = {}
        for c in contacts:  # the same candidate can appear as several objects; keep the judged one
            cur = best.get(c.id)
            if cur is None or (cur.score is None and c.score is not None):
                best[c.id] = c
        rows = examples_from(list(best.values()), profile_for(spec, pack))
        store.record_fit_examples(pack.name, spec.id, rows)
        model = train(store.fit_examples(pack.name))
        if model is None:
            return {"examples_added": len(rows), "model": None}
        store.save_fit_model(pack.name, model)
        return {"examples_added": len(rows), "model": {k: model[k] for k in ("n", "positives", "threshold",
                                                                              "train_precision", "train_recall")}}
    except Exception as e:
        log.warning("could not update the fit model: %s", e)
        return {"error": str(e)}


def backfill(spec: CampaignSpec, pack, ctx: CampaignContext, deps: Deps, planner: SearchPlanner,
             winners: List[Contact], *, known: List[Contact], rank_stage, enrich_stage, judge_stage,
             funnel: Dict[str, Any], progress: Callable[[str], None], ceiling: int,
             blocked: Optional[Callable[[Contact], Optional[str]]] = None) -> List[Contact]:
    """Too few candidates passed the judge: search once more, seeded with the ones that did (lookalikes,
    co-authors, the best-performing tool) and steered away from why the others failed. Only NEW candidates
    are ranked, enriched and judged, so nothing is paid for twice."""
    deps.search.budget = ceiling
    left = deps.search.remaining
    if left <= 2 or not deps.search.available and not any(t in pack.sources for t in ("openalex", "github", "podcasts")):
        return []
    missing = spec.target_count - len(winners)
    progress(f"backfill: only {len(winners)} passed; searching once more for ~{missing} more, seeded with the best…")
    planner.winners = [w.org or w.name for w in winners[:8]]
    planner.need = max(10, missing * 4)
    planner.replanned = False  # one more replan, now with winners and rejection reasons
    ex = explore(spec, pack, ctx, deps, planner, budget=max(0, left - 1), rounds=2, need=planner.need,
                 winners=winners[:5], progress=progress, blocked=blocked)
    for why, n in ex.skipped.items():
        d = funnel.setdefault("dedup_discovery", {})
        d[why] = d.get(why, 0) + n
    seen = {c.id for c in known} | {c.org_domain for c in known if c.kind == "org" and c.org_domain}
    fresh = [c for c in ex.contacts if c.id not in seen and not (c.kind == "org" and c.org_domain in seen)]
    stats: Dict[str, Any] = {"found": len(fresh), "spent": ex.rounds[-1]["spent"] if ex.rounds else 0}
    funnel["backfill"] = stats
    # The funnel's totals include what the backfill found, so `qualified` can never exceed `unique_people`.
    funnel["unique_people"] = funnel.get("unique_people", 0) + len(fresh)
    note_sources(funnel, "unique", fresh, add=True)
    for k, v in ex.by_source.items():
        funnel.setdefault("by_source", {})[k] = funnel.get("by_source", {}).get(k, 0) + v
    if not fresh:
        return []
    ranked = rank_stage(fresh)[: max(missing * 6, 20)]
    note_sources(funnel, "qualified", ranked, add=True)
    targets = enrich_stage(diversify(ranked, min(len(ranked), max(missing * 3, 10)), deps.settings.max_per_org * 4),
                           ranked)
    reachable = [c for c in targets if has_usable_email(c, reach_threshold(deps.settings, pack))]
    stats["reachable"] = len(reachable)
    note_sources(funnel, "reachable", reachable, add=True)
    if not reachable or not deps.llm.available:
        return []
    passed = judge_stage(reachable[: missing * 2])
    stats["passed_judge"] = len(passed)
    return passed

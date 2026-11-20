"""
Pre-judge: one cheap, batched LLM screen of the top of the ranking BEFORE enrichment.

Enrichment is the expensive stage (page reads, decision-maker searches), and in v2 it was spent on
whatever ranked highest lexically; the real judge only saw the reachable survivors. Screening first moves
that spend to candidates that can fit, and makes a good prospect without an email visible as a
reachability problem instead of silently lost.

Cost: ~120 input tokens per candidate on the fast model; 40 candidates ≈ 2 calls ≈ $0.001.
A failed call leaves candidates unscreened (never rejected), so the pre-judge can only remove waste.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from brain.models import CampaignSpec, Contact
from brain.textutil import squash

log = logging.getLogger("brain.prejudge")

SYSTEM = """You screen candidates for an outreach campaign before any time is spent on them. From the evidence only,
decide how plausible each one is as a TARGET for this campaign.
Score 0-100:
 70-100  clearly the right kind of target (right field, right kind of person or organisation, right place)
 40-69   plausible but unclear from the evidence
 0-39    wrong kind of target: unrelated field, wrong type of organisation, clearly elsewhere, or not a real
         person/organisation at all (a page title, product, event, directory or article)
Thin evidence alone is not a reason for a low score: judge what the evidence shows. If a quality the campaign wants
cannot be confirmed or ruled out from the evidence ("not clearly independent", "unclear focus"), score 40-60 — below
40 only when the evidence shows the candidate is the wrong kind of target.
Return JSON: {"results": [{"id": "<id>", "score": <int>, "why": "<at most 12 words>"}]}. Include every id once."""

PROOF = ("past_sponsorship", "hiring", "portfolio_fit", "recent_coverage", "spoke_before", "local_business",
         "publications", "open_source", "app_publisher", "recently_funded", "grant_awarded", "exhibits_at")


def brief(spec: CampaignSpec, prof=None) -> str:
    lines = [f"Campaign: {spec.objective}", f"Ask: {spec.ask}" if spec.ask else "",
             f"Looking for: {spec.persona.description}" if spec.persona.description else "",
             "Kinds of organisation: " + ", ".join(spec.persona.org_types) if spec.persona.org_types else "",
             "Roles: " + ", ".join(spec.persona.roles[:6]) if spec.persona.roles else "",
             "Topics: " + ", ".join(spec.topics + spec.expanded_terms[:3]),
             "Where: " + ", ".join(spec.regions + spec.countries) if spec.regions or spec.countries else ""]
    if prof is not None and prof.criteria:
        lines.append("What makes a good target: " + "; ".join(c.name for c in prof.criteria[:6]))
    if prof is not None and prof.anti_patterns:
        lines.append("Poor targets look like: " + ", ".join(prof.anti_patterns[:8]))
    return "\n".join(l for l in lines if l)


def card(c: Contact) -> Dict:
    cls = next((f.split(":", 1)[1] for f in c.flags if f.startswith("org_class:")), None)
    out = {"id": c.id, "kind": c.kind, "name": c.name, "title": c.title, "organisation": c.org if c.org != c.name else None,
           "type": cls if cls not in (None, "company") else None,
           "where": c.location_text or c.country, "about": squash(c.bio or "", 220) or None,
           "evidence": [squash(e.text, 150) for e in c.evidence[:3]],
           "proof": {k: round(v, 2) for k, v in c.signals.items() if k in PROOF and v > 0} or None}
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def prejudge(contacts: List[Contact], spec: CampaignSpec, llm, prof=None, batch_size: int = 20,
             min_score: float = 0.3, stats: Optional[Dict] = None, ctx=None) -> int:
    """Screen the given contacts in place: sets Contact.prejudge (0-1) and .prejudge_reason, and flags
    "prejudge_reject" below `min_score`. Contacts already screened are skipped. Returns how many were screened."""
    todo = [c for c in contacts if c.prejudge is None]
    if not todo or llm is None or not getattr(llm, "available", False):
        return 0
    from brain.safety import guard, scrub

    scrub(todo)
    head = brief(spec, prof)
    done = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        by_id = {c.id: c for c in batch}
        try:
            data = llm.json(system=guard(SYSTEM), user=head + "\n\nCandidates:\n" + json.dumps([card(c) for c in batch],
                                                                                    ensure_ascii=False),
                            purpose="prejudge", tier="fast", max_tokens=40 * len(batch) + 120)
        except Exception as e:  # screening is an optimisation: never let it sink the campaign
            log.warning("pre-judge batch failed: %s", e)
            continue
        for item in (data.get("results") or []) if isinstance(data, dict) else []:
            c = by_id.pop(str(item.get("id")), None) if isinstance(item, dict) else None
            if c is None:
                continue
            try:
                c.prejudge = max(0.0, min(100.0, float(item.get("score")))) / 100.0
            except (TypeError, ValueError):
                continue
            c.prejudge_reason = squash(str(item.get("why") or ""), 120) or None
            done += 1
            if c.prejudge < min_score:
                c.flags.append("prejudge_reject")
                if ctx is not None and c.prejudge_reason and len(ctx.rejected_reasons) < 20:
                    ctx.rejected_reasons.append(f"{c.name}: {c.prejudge_reason}")
    if stats is not None:
        stats["screened"] = stats.get("screened", 0) + done
        stats["rejected"] = stats.get("rejected", 0) + sum(1 for c in todo if "prejudge_reject" in c.flags)
    return done


def reorder(ranked: List[Contact]) -> List[Contact]:
    """Screened candidates first (best blend of screen and pre-score), then unscreened in pre-score order, then
    rejected ones last (kept: they still help learn each organisation's email convention)."""
    screened = [c for c in ranked if c.prejudge is not None and "prejudge_reject" not in c.flags]
    rest = [c for c in ranked if c.prejudge is None]
    rejected = [c for c in ranked if "prejudge_reject" in c.flags]
    screened.sort(key=lambda c: 0.6 * c.prejudge + 0.4 * c.prescore, reverse=True)
    return screened + rest + rejected

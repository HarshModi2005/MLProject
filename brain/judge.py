"""
LLM fit judge. Batched: one call scores several candidates, so judging 50 people costs ~7 calls
instead of 50. It only ever sees candidates that are already filtered, pre-ranked and reachable.

The judge also returns the "hook" — the single most relevant sourced fact about the person —
which the drafting step reuses, so the email writer does not have to rediscover it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from brain.llm import LLM, LLMError
from brain.models import CampaignSpec, Contact
from brain.textutil import squash

log = logging.getLogger("brain.judge")

SYSTEM = """You assess whether a cold email from a specific sender to a specific person is worth sending.
Judge fit strictly on the evidence given. Do not assume facts that are not listed.

Scoring (0-100):
 90-100  the person's current work directly overlaps what the sender has done or wants, and they are the right kind of person to grant the ask
 70-89   clear topical overlap and plausible ability to help
 40-69   loosely related field, or right field but wrong role/seniority
 0-39    unrelated, or evidence too thin to justify contacting them

Candidates with "kind": "org" are organisations reached through a shared inbox: judge the ORGANISATION's fit for the
ask. Candidates with "kind": "person" must be real individuals: if one is actually a page title, department, event or
product, score 0. Also score 0 for publishers, directories and media sites that merely write about the topic.

If CRITERIA are listed, also rate each criterion 0-100 for each candidate, strictly from the evidence (0 when the
evidence says nothing about it), and let the overall score reflect them.

Return JSON: {"results": [{"id": "<id>", "score": <int>, "reasons": ["<short reason>", "<short reason>"],
"criteria": {"<criterion name>": <int>, ...},
"hook": "<the single most relevant fact from this person's evidence, quoted or closely paraphrased, or null>",
"hook_ref": <index of the evidence item the hook comes from, or null>}]}
Include every candidate id exactly once. Keep each reason under 20 words."""


def _card(c: Contact, max_evidence: int = 5) -> Dict:
    card = {
        "id": c.id,
        "kind": c.kind,
        "name": c.name,
        "title": c.title,
        "organisation": c.org,
        "topics": c.topics[:6],
        "bio": squash(c.bio or "", 240) or None,
        "contact_via": (c.best_email.address if c.best_email else None),
        "evidence": [
            {"i": i, "what": squash(ev.text, 180), "year": ev.year, "detail": squash(ev.detail or "", 200) or None}
            for i, ev in _best_evidence(c, max_evidence)
        ],
    }
    if c.decision_maker:
        card["decision_maker"] = f"{c.decision_maker.get('name')} ({c.decision_maker.get('title')})"
    proofs = {k: v for k, v in c.signals.items() if k in _PROOF_KEYS and v > 0}
    if proofs:
        card["verified_signals"] = proofs  # set by discovery channels from direct evidence (e.g. listed as a sponsor)
    return card


_PROOF_KEYS = ("past_sponsorship", "hiring", "portfolio_fit", "recent_coverage", "spoke_before", "open_roles",
               "events_count", "local_business")


def _best_evidence(c: Contact, k: int):
    """The judge sees at most k items: keep the channel proofs (sponsor of X, hiring, portfolio) next to the
    first descriptive items instead of whatever happened to be scraped first. Order is preserved so hook_ref
    indices still point into c.evidence."""
    if len(c.evidence) <= k:
        return list(enumerate(c.evidence))
    proof = [i for i, e in enumerate(c.evidence) if re.match(r"(Sponsor|Hiring|Portfolio|Speaker|Publishes with|Episode)", e.text)]
    keep = sorted(set(list(range(k))[: max(1, k - len(proof[:2]))] + proof[:2]))[:k]
    return [(i, c.evidence[i]) for i in keep]


def _country(code: str) -> str:
    from brain.searchplan import _country_name

    return _country_name(code)


def sender_brief(spec: CampaignSpec) -> str:
    s = spec.sender
    lines = [
        f"Sender: {s.name or 'unknown'} — {s.headline}".rstrip(" —"),
        f"Background: {squash(s.summary, 500)}" if s.summary else "",
        "Facts: " + " | ".join(squash(h, 160) for h in s.highlights[:8]) if s.highlights else "",
        f"Objective: {spec.objective}",
        f"Ask: {spec.ask}" if spec.ask else "",
        f"Looking for: {spec.persona.description}" if spec.persona.description else "",
        "Topics of interest: " + ", ".join(spec.topics),
        ("Target geography: " + ", ".join(spec.regions + [_country(c) for c in spec.countries])
         + " — organisations or people clearly based or active elsewhere are a poor fit") if spec.regions or spec.countries else "",
    ]
    return "\n".join(l for l in lines if l)


def judge(contacts: List[Contact], spec: CampaignSpec, llm: LLM, batch_size: int = 8, ctx=None,
          playbook=None) -> int:
    """Sets score / reasons / hook on each contact in place. Returns the number judged by the LLM.
    Contacts the model could not score keep score=None and are flagged, never silently defaulted."""
    from brain.draft import with_context

    from brain.fit import profile_for, reach_factor

    brief = sender_brief(with_context(spec, ctx))
    if playbook is not None and playbook.judge_rubric:
        brief += f"\nWhat makes a good target here: {playbook.judge_rubric}"
    prof = profile_for(spec, playbook) if playbook is not None else spec.profile
    if prof is not None and prof.criteria:
        brief += "\nCRITERIA (confirmed by the user): " + "; ".join(
            f"{c.name}{' [required]' if c.must else ''}" for c in prof.criteria)
        if prof.anti_patterns:
            brief += "\nPoor targets look like: " + ", ".join(prof.anti_patterns[:8])
    if ctx is not None and ctx.brief():
        brief += "\n" + ctx.brief()
    judged = 0
    for start in range(0, len(contacts), batch_size):
        batch = contacts[start:start + batch_size]
        by_id = {c.id: c for c in batch}
        user = brief + "\n\nCandidates:\n" + json.dumps([_card(c) for c in batch], ensure_ascii=False)
        try:
            data = llm.json(system=SYSTEM, user=user, purpose="judge", tier="strong",
                            max_tokens=180 * len(batch) + 200)
        except LLMError as e:
            log.warning("judge batch failed: %s", e)
            for c in batch:
                c.flags.append("unjudged")
            continue
        results = data.get("results") if isinstance(data, dict) else None
        for item in results or []:
            c = by_id.pop(str(item.get("id")), None) if isinstance(item, dict) else None
            if c is None:
                continue
            try:
                score = max(0.0, min(100.0, float(item.get("score")))) / 100.0
            except (TypeError, ValueError):
                c.flags.append("unjudged")
                continue
            crit = item.get("criteria") if isinstance(item.get("criteria"), dict) else {}
            if prof is not None and prof.criteria and crit:
                per = {}
                for cr in prof.criteria:
                    v = next((val for k, val in crit.items() if str(k).strip().lower() == cr.name.lower()), None)
                    try:
                        per[cr.name] = max(0.0, min(100.0, float(v))) / 100.0 if v is not None else None
                    except (TypeError, ValueError):
                        per[cr.name] = None
                rated = [(cr, per[cr.name]) for cr in prof.criteria if per.get(cr.name) is not None]
                if rated:
                    crit_score = sum(cr.weight * v for cr, v in rated) / (sum(cr.weight for cr, _ in rated) or 1.0)
                    score = 0.5 * score + 0.5 * crit_score
                    c.criteria = {cr.name: round(v, 2) for cr, v in rated}
                    missing = [cr.name for cr, v in rated if cr.must and v < 0.3]
                    if missing:
                        score *= 0.5
                        c.flags.append("missing_required:" + ", ".join(missing))
            reach = reach_factor(c) if c.emails else 1.0
            c.score = round((0.8 * score + 0.2 * c.prescore) * (0.85 + 0.15 * reach), 4)
            c.reasons = [squash(str(r), 160) for r in (item.get("reasons") or [])[:3]]
            hook = item.get("hook")
            ref = item.get("hook_ref")
            # Trust the pointer over the model's wording: the hook must be real evidence.
            if isinstance(ref, int) and 0 <= ref < len(c.evidence):
                c.hook = c.evidence[ref].text
            elif isinstance(hook, str) and hook.strip():
                low = hook.strip().lower()
                match = next((ev for ev in c.evidence if low[:40] in ev.text.lower() or ev.text.lower()[:40] in low), None)
                c.hook = match.text if match else None  # a topic label is not a hook
            if not c.hook:
                pub = next((ev for ev in c.evidence if ev.kind == "publication"), None)
                c.hook = pub.text if pub else None
            judged += 1
        for c in by_id.values():  # ids the model skipped
            c.flags.append("unjudged")
    return judged

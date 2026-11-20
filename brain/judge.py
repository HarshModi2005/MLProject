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
from brain.safety import guard, scrub
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

Role in the ask matters. For asks to distribute, resell, partner, sponsor, buy, invest or hire, an organisation that
makes or sells the SAME thing as the sender (a competitor) is not a fit, and neither is a directory, ranking, marketplace or
publisher that only lists or covers organisations: set "competitor_or_directory" true and score at most 25.
Geography matters. If a target geography is given and the candidate is clearly based or active only elsewhere, set
"region_ok" false (true if it fits, null if the evidence does not say) and score at most 40.

If CRITERIA are listed, also rate each criterion 0-100 for each candidate, strictly from the evidence (0 when the
evidence says nothing about it), and let the overall score reflect them.

Return JSON: {"results": [{"id": "<id>", "score": <int>, "reasons": ["<short reason>", "<short reason>"],
"criteria": {"<criterion name>": <int>, ...},
"hook": "<the single most relevant fact from this person's evidence, quoted or closely paraphrased, or null>",
"region_ok": <true | false | null>, "competitor_or_directory": <true | false>,
"hook_ref": <index of the evidence item the hook comes from, or null>,
"sender_ref": <number of the SENDER FACT most relevant to this candidate, or null>,
"why_now": "<a timing fact quoted from the evidence that makes now a good moment (hiring now, just raised, a recent
article or episode, an open call), or null — never invent one>"}]}
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


def _judge_keys():
    from brain.discovery.signals import judge_signals

    return judge_signals()


_PROOF_KEYS = _judge_keys()  # signals the judge sees as verified (brain/discovery/signals.py)


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
        ("SENDER FACTS:\n" + "\n".join(f"[{i}] {squash(h, 160)}" for i, h in enumerate(s.highlights[:12])))
        if s.highlights else "",
        f"Objective: {spec.objective}",
        f"Ask: {spec.ask}" if spec.ask else "",
        f"Offer: {spec.offer.value_prop}" + (f" (proof: {'; '.join(spec.offer.proof_points[:3])})" if spec.offer.proof_points else "")
        if spec.offer.value_prop else "",
        _constraints_line(spec),
        f"Looking for: {spec.persona.description}" if spec.persona.description else "",
        "Topics of interest: " + ", ".join(spec.topics),
        ("Target geography: " + ", ".join(spec.regions + [_country(c) for c in spec.countries])
         + " — organisations or people clearly based or active elsewhere are a poor fit") if spec.regions or spec.countries else "",
    ]
    return "\n".join(l for l in lines if l)


def _constraints_line(spec: CampaignSpec) -> str:
    k = spec.constraints
    parts = [f"{k.size_min or 0}-{k.size_max or 'any'} employees" if (k.size_min or k.size_max) else "",
             "stage: " + ", ".join(k.stages) if k.stages else "", f"seniority: {k.seniority}" if k.seniority else "",
             "must: " + "; ".join(k.must) if k.must else "", "avoid: " + "; ".join(k.avoid) if k.avoid else ""]
    parts = [p for p in parts if p]
    return ("Limits set by the user (a candidate clearly outside them is a poor fit): " + " · ".join(parts)) if parts else ""


def _in_evidence(text: str, c: Contact) -> bool:
    low = (text or "").strip().lower()
    if len(low) < 8:
        return False
    hay = " ".join([c.bio or "", c.title or ""] + [f"{e.text} {e.detail or ''}" for e in c.evidence]).lower()
    probe = low[:40]
    return probe in hay or any(e.text.lower()[:40] in low for e in c.evidence if len(e.text) >= 12)


def judge(contacts: List[Contact], spec: CampaignSpec, llm: LLM, batch_size: int = 8, ctx=None,
          playbook=None, prescore_weight: float = 0.0, calibrator=None) -> int:
    """Sets fit / score / reasons / hook on each contact in place. Returns the number judged by the LLM.
    Contacts the model could not score keep score=None and are flagged, never silently defaulted.

    `fit` is the judge's view of the match alone. `score` is what ranks: with a calibrator (brain/calibrate.py, fitted on
    human labels and only kept if it beats the raw score) it is the probability a human would call this a good prospect;
    without one it is fit x a reachability factor. The lexical pre-score no longer enters it (prescore_weight=0): it is
    noise relative to the judge and made "0.7 = good" mean different things in different campaigns."""
    from brain.draft import with_context

    from brain.fit import profile_for, reach_factor

    from brain.textutil import is_aggregator

    for c in contacts:  # directories and rankings are never targets: no model call needed
        if is_aggregator(c.org or c.name, c.org_domain or ""):
            c.fit, c.score = 0.0, 0.0
            c.flags += ["competitor_or_directory", "aggregator"]
            c.reasons = ["a directory or aggregator, not a target"]
    contacts = [c for c in contacts if "aggregator" not in c.flags]
    scrub(contacts)  # scraped text may instruct the model: remove it before it reaches the prompt

    full = with_context(spec, ctx)
    brief_highlights = full.sender.highlights[:12]  # the numbered SENDER FACTS the judge's sender_ref points into
    brief = sender_brief(full)
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
            data = llm.json(system=guard(SYSTEM), user=user, purpose="judge", tier="strong",
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
            c.fit = round(score, 4)
            c.score = round(((1 - prescore_weight) * score + prescore_weight * c.prescore) * (0.85 + 0.15 * reach), 4)
            if calibrator is not None:
                c.score = round(calibrator.prob(c), 4)
                if "calibrated" not in c.flags:
                    c.flags.append("calibrated")
            c.reasons = [squash(str(r), 160) for r in (item.get("reasons") or [])[:3]]
            if item.get("competitor_or_directory") is True:  # the model said so; the cap does not depend on its score
                c.fit, c.score = min(c.fit, 0.25), min(c.score, 0.25)
                c.flags.append("competitor_or_directory")
            if item.get("region_ok") is False and (spec.countries or spec.regions):
                c.fit, c.score = min(c.fit, 0.4), min(c.score, 0.4)
                c.flags.append("wrong_region")
            if "injection_suspect" in c.flags:  # its sources tried to instruct the model: half the trust, said out loud
                c.score = round(c.score * 0.5, 4)
                c.reasons.insert(0, "source text tried to instruct the model; treated as untrusted")
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
            ref = item.get("sender_ref")
            c.sender_ref = ref if isinstance(ref, int) and 0 <= ref < len(brief_highlights) else None
            wn = item.get("why_now")
            c.why_now = squash(str(wn), 200) if isinstance(wn, str) and _in_evidence(wn, c) else None
            judged += 1
        for c in by_id.values():  # ids the model skipped
            c.flags.append("unjudged")
    return judged

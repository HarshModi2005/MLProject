"""
Stage-by-stage LLM-as-judge: one rubric per part of the pipeline, run over saved bench runs (no discovery spend).

    python tools/stage_judge.py --runs s2_1,s2_2 --tag s2            # every scenario in those runs
    python tools/stage_judge.py --runs stress_c --only stress_phd_neuro

The earlier judge (tools/llm_judge.py) grades the finished shortlist. This one grades each stage on its own, so a
failure can be traced to the part that caused it. The rubrics:

  R1 intake      Was the request understood? playbook, topics, geography, every explicit constraint kept, nothing invented,
                 vague or contradictory requests flagged rather than guessed.
  R2 candidates  Blind audit of a stratified sample of the raw discovery pool: is each a valid target? -> pool precision.
  R3 cuts        The same sample splits into candidates dropped at triage, at the enrichment cap, at the final cut, and kept.
                 The share of valid targets among the dropped is each stage's false-negative rate.
  R4 recall      A probe: the judge names real entities that clearly fit, WITHOUT seeing the results; how many are in
                 the pool and in the shortlist? (The model may misremember, so only confident, checkable names are used.)
  R5 reach       The address of each shortlisted prospect: does it belong to them, can the addressee grant the ask,
                 should it be sent, looked up again, or dropped? Plus deterministic checks (domain match, free mail, role inbox).
  R6 ranking     Do the pipeline's scores order the list the way an independent fit grade does? Inversions, top vs bottom third,
                 duplicates by organisation, concentration by source and country.
  R7 drafts      Claim by claim: every factual statement in the email against the sender facts and recipient evidence.
  R8 reporting   Does what the system tells the user (warnings, counts) match the real state of the list?
  R9 planning    Were the right discovery tools used for the request, and was the effort spent where precision was?

It is a triage instrument, not ground truth: the judge has no web access, and R4 depends on the model's memory.
Output: evals/results/stage_judge_<tag>.json plus a scorecard on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from brain.cache import Cache  # noqa: E402
from brain.discovery.reach.tiers import contact_tier  # noqa: E402
from brain.llm import LLM  # noqa: E402
from brain.models import CampaignSpec, Contact  # noqa: E402
from brain.playbooks import catalog_text, get_playbook  # noqa: E402
from brain.settings import Settings  # noqa: E402
from brain.textutil import FREE_MAIL, registrable_domain  # noqa: E402
from llm_judge import _load, judge_prospects, lint_email  # noqa: E402

RESULTS = ROOT / "evals" / "results"
SCENARIOS = ROOT / "evals" / "scenarios"

TOOLS = """orgs: organisations from web search and list pages | places: local businesses from Google Maps | news: journalists from recent articles
github: developers via GitHub user search | npm: maintainers of npm packages | openalex: researchers via publications | arxiv: authors of recent papers
web: people on staff/team pages | events: sponsors/speakers of similar events | jobs: companies hiring now | portfolio: investors via portfolio pages
podcasts: podcasts via iTunes + RSS | lookalike: organisations similar to the best finds | coauthors: researchers publishing with the best finds
huggingface: ML model authors | memory: prospects from earlier campaigns"""

# ── prompts ────────────────────────────────────────────────────────────────────────────────────────────────────────

INTAKE_SYSTEM = """You audit the FIRST stage of an outreach system: the intake that turns a user's free-text request into a
structured campaign. You see the REQUEST, the SENDER, the PLAYBOOK CATALOG, and what the system produced. Be strict.
Return JSON:
 playbook_correct: true/false; best_playbook: a name from the catalog; playbook_note: <= 20 words;
 topics_faithful: 1-5 (do the topics say what the TARGETS are/do, not what the sender sells or does?); topics_issue: <= 20 words or "";
 location_faithful: 1-5 (geography exactly as asked; nothing added or dropped); location_issue: <= 20 words or "";
 constraints: [{"constraint": "...", "preserved": "yes"|"partly"|"no"}] for EVERY explicit constraint in the request
   (exclusions, counts, geography, seniority, org types, negative phrases like "not X", "also Y");
 invented: constraints the system added that the user never said (list of short strings; ignore target_count, expanded_terms
   and routine assumptions, which the system sets itself);
 request_quality: "clear" | "vague" | "contradictory";
 handled_well: true/false: for a vague or contradictory request, did the structured output or its assumptions flag/ask/resolve it
   sensibly, instead of guessing silently? (true for a clear request that was understood);
 ask_fit: 1-5 (does the stated ask match what the user wants from the recipient?);
 overall: 1-5; top_problem: <= 25 words or "".
JSON only."""

CAND_CODES = ["wrong_org_type", "wrong_role", "wrong_geography", "off_topic", "directory_or_aggregator", "competitor",
              "wrong_size", "duplicate", "no_evidence", "stale", "excluded_by_user", "other"]

CAND_SYSTEM = f"""You are a careful researcher building a LONGLIST for an outreach campaign. For each candidate decide: is it a valid
target (would you keep it on the longlist for this exact request)? You see only the candidate's own data, not how the system
ranked it. Use the evidence and your general knowledge; do not invent facts.
For each id return: verdict "yes" | "maybe" | "no" | "unsure"; code (when not yes): one of {CAND_CODES}; note <= 15 words.
A directory, listicle, marketplace or site builder is never a valid target. An entity the request explicitly excludes is "no"
with code excluded_by_user. JSON: {{"results": [...]}} with every id once."""

PROBE_SYSTEM = """You know the real world. For the campaign request below, name up to 12 SPECIFIC real organisations or people that
clearly fit it. Only include ones you are confident exist and fit; skip when unsure (fewer is fine; zero is fine for an obscure
niche). Do not include anything the request excludes. For each: name, domain (official website domain if you know it, else ""),
confidence 0-1. JSON: {"entities": [{"name": "...", "domain": "...", "confidence": 0.9}]}."""

REACH_SYSTEM = """You audit how an outreach system chose the ADDRESS for each shortlisted prospect. You see the campaign ask and, per
prospect: name, kind, title, organisation, domain, the address, how it was found (method, tier) and any decision maker the system
attached. Be strict; the sender will email exactly this address.
For each id return:
 address_belongs: "yes" | "probably" | "doubtful" | "no" (does this address look like it reaches THIS person/organisation);
 addressee_can_grant: "yes" | "no" | "unclear" (would whoever reads that address be able to grant the ask: a shop, legal-notice, billing or
   support desk cannot; a generic info@ of a one-person business can);
 action: "send" | "look_up_person" | "drop"; note <= 18 words.
JSON: {"results": [...]} with every id once."""

CLAIM_SYSTEM = """You fact-check cold emails written by a system for a sender. For each email you see SENDER FACTS (the only things the email
may claim about the sender), RECIPIENT EVIDENCE (the only things it may claim about the recipient), and the EMAIL.
List every factual claim in the email (skip greetings, questions, the ask itself and pure politeness).
For each id return:
 claims: [{"text": "<=15 words", "about": "sender"|"recipient"|"world", "status": "supported"|"unsupported"|"contradicted"|"puffery"}]
   (supported = stated or clearly implied by the facts; unsupported = not in the facts; contradicted = conflicts with them;
    puffery = vague praise like "impressive work" with no fact behind it);
 specific_to_recipient: 1-5 (could this have been sent to anyone else of the same type?);
 spammy: true/false (mass-mail tone, hype, pressure); length_ok: true/false; subject_ok: true/false.
JSON: {"results": [...]} with every id once."""

REPORT_SYSTEM = """You audit what an outreach system TELLS THE USER about a finished search. You see the request, how many contacts were wanted,
the funnel counts, the warnings the system printed, and an independent review of the real quality of the list.
Judge honesty and usefulness of the system's messaging:
 truthful: 1-5 (does the messaging match the true state? A list padded with bad prospects, or a big shortfall with no warning, is not);
 actionable: 1-5 (would the user know exactly what to do next?);
 false_reassurance: list of statements or silences that would mislead the user (short strings);
 missing_message: the single most important thing the user should have been told and was not (<= 30 words, or "").
JSON only."""

PLAN_SYSTEM = """You audit the PLANNING stage of an outreach system: which discovery tools it used for a request. Tools available:
""" + TOOLS + """
You see the request, the playbook, the tools the playbook lists, and per tool actually used: candidates found, how many of a blind
sample were valid targets, and how many reached a shortlist. Judge:
 tools_right: 1-5 (were the right tools used for this request?);
 missed_tools: [{"tool": "...", "why": "<= 15 words"}] tools that should have been used and were not;
 wasteful_tools: [{"tool": "...", "why": "<= 15 words"}] tools that produced mostly invalid candidates;
 effort_spent_where_it_pays: 1-5; top_problem: <= 25 words or "".
JSON only."""


# ── loading ────────────────────────────────────────────────────────────────────────────────────────────────────────

def load_stages(db_path: Path) -> Dict[str, Any]:
    db = sqlite3.connect(db_path)
    return {s: json.loads(p) for s, p in db.execute("select stage, payload_json from brain_stages")}


def _contacts(stage: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["id"]: c for c in (stage or {}).get("contacts", [])}


def _pool(stages: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every candidate seen at any stage; later stages hold ids (merged or new organisation contacts) discovery did not."""
    out: Dict[str, Dict[str, Any]] = {}
    for k in ("judge", "enrich", "rank", "discover"):
        out.update(_contacts(stages.get(k)))
    return out


def _blind_view(c: Dict[str, Any]) -> str:
    lines = [f"name: {c.get('name')}  kind: {c.get('kind')}  title: {c.get('title') or '-'}  org: {c.get('org') or '-'}  "
             f"domain: {c.get('org_domain') or '-'}  country: {c.get('country') or '-'}",
             f"location: {(c.get('location_text') or '-')[:90]}"]
    if c.get("topics"):
        lines.append("topics: " + ", ".join(c["topics"][:6]))
    if c.get("bio"):
        lines.append("bio: " + c["bio"][:250])
    for ev in (c.get("evidence") or [])[:3]:
        lines.append(f"- [{ev.get('kind')}] {str(ev.get('text'))[:150]}")
    return "\n".join(lines)


def _request_text(spec: CampaignSpec, prompt: str) -> str:
    return (f"REQUEST (user's words): {prompt}\nGOAL: {spec.objective}\nASK: {spec.ask}\nTARGET: {spec.persona.description}\n"
            f"WHERE: {', '.join(spec.regions + spec.countries) or 'anywhere'}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ── rubrics ────────────────────────────────────────────────────────────────────────────────────────────────────────

def r1_intake(llm: LLM, spec: CampaignSpec, prompt: str, stages: Dict[str, Any], sender: str) -> Dict[str, Any]:
    ctx = stages.get("__context__", {})
    out = {"playbook": spec.pack, "kind": spec.target_kind, "persona": spec.persona.description,
           "roles": spec.persona.roles, "org_types": spec.persona.org_types, "topics": spec.topics,
           "expanded_terms": spec.expanded_terms[:8], "regions": spec.regions, "countries": spec.countries,
           "objective": spec.objective, "ask": spec.ask, "target_count": spec.target_count,
           "exclude_names": ctx.get("exclude_names"), "exclude_domains": ctx.get("exclude_domains"),
           "assumptions": ctx.get("assumptions"), "understanding": ctx.get("understanding")}
    user = (f"REQUEST: {prompt}\nSENDER: {sender}\n\nPLAYBOOK CATALOG:\n{catalog_text()}\n\nSYSTEM OUTPUT:\n"
            f"{json.dumps(out, indent=1, default=str)[:3500]}")
    return llm.json(system=INTAKE_SYSTEM, user=user, purpose="stage_intake", tier="strong", max_tokens=1500)


def sample_pool(stages: Dict[str, Any], final_ids: set, seed: str) -> Dict[str, List[Dict[str, Any]]]:
    A, B, C = (_contacts(stages.get(k)) for k in ("discover", "rank", "enrich"))
    P = _pool(stages)
    rng = random.Random(seed)
    groups = {"dropped_triage": [i for i in A if i not in B],
              "dropped_enrich_cap": [i for i in B if i not in C],
              "dropped_final": [i for i in C if i not in final_ids],
              "kept": [i for i in P if i in final_ids]}
    take = {"dropped_triage": 10, "dropped_enrich_cap": 8, "dropped_final": 10, "kept": 10}
    sample = {}
    for g, ids in groups.items():
        rng.shuffle(ids)
        sample[g] = [dict(P[i], _stratum_size=len(ids)) for i in ids[:take[g]]]
    return sample


def r2_r3_candidates(llm: LLM, spec: CampaignSpec, prompt: str, sample: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    flat = [(g, c) for g, cs in sample.items() for c in cs]
    head = _request_text(spec, prompt) + "\n\nCANDIDATES:\n"
    verdicts: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(flat), 10):
        chunk = flat[i:i + 10]
        body = "\n\n".join(f"### id: {c['id']}\n{_blind_view(c)}" for _, c in chunk)
        data = llm.json(system=CAND_SYSTEM, user=head + body, purpose="stage_candidates", tier="strong", max_tokens=2200)
        for r in data.get("results", []):
            if isinstance(r, dict):
                verdicts[str(r.get("id"))] = r
    strata: Dict[str, Any] = {}
    for g, cs in sample.items():
        v = [verdicts.get(c["id"], {}).get("verdict") for c in cs]
        n = len([x for x in v if x])
        yes = sum(1 for x in v if x == "yes")
        maybe = sum(1 for x in v if x == "maybe")
        strata[g] = {"sampled": len(cs), "size": cs[0]["_stratum_size"] if cs else 0, "yes": yes, "maybe": maybe,
                     "no": sum(1 for x in v if x == "no"), "yes_rate": round(yes / n, 2) if n else None,
                     "codes": dict(Counter(verdicts.get(c["id"], {}).get("code") for c in cs
                                           if verdicts.get(c["id"], {}).get("verdict") in ("no", "maybe"))),
                     "examples": [{"name": c["name"], "org": c.get("org"), "verdict": verdicts.get(c["id"], {}).get("verdict"),
                                   "code": verdicts.get(c["id"], {}).get("code"), "note": verdicts.get(c["id"], {}).get("note"),
                                   "sources": c.get("sources")} for c in cs[:6]]}
    total = sum(s["size"] for s in strata.values())
    est = sum((s["yes_rate"] or 0) * s["size"] for s in strata.values()) / total if total else None
    by_source: Dict[str, List[str]] = defaultdict(list)
    for g, c in flat:
        for s in c.get("sources") or ["?"]:
            by_source[s].append(verdicts.get(c["id"], {}).get("verdict"))
    src = {s: {"n": len(v), "yes": sum(1 for x in v if x == "yes"), "no": sum(1 for x in v if x == "no")} for s, v in by_source.items()}
    return {"pool_precision_est": round(est, 2) if est is not None else None, "strata": strata, "by_source": src}


def r4_recall(llm: LLM, spec: CampaignSpec, prompt: str, stages: Dict[str, Any], final_ids: set) -> Dict[str, Any]:
    data = llm.json(system=PROBE_SYSTEM, user=_request_text(spec, prompt), purpose="stage_recall", tier="strong", max_tokens=900)
    ents = [e for e in data.get("entities", []) if isinstance(e, dict) and (e.get("confidence") or 0) >= 0.7 and e.get("name")]
    A = _pool(stages)

    def match(e: Dict[str, Any], pool: Dict[str, Dict[str, Any]]) -> bool:
        name, dom = _norm(e["name"]), registrable_domain(e.get("domain") or "")
        for c in pool.values():
            if dom and registrable_domain(c.get("org_domain") or "") == dom:
                return True
            cn = [_norm(c.get("name") or ""), _norm(c.get("org") or "")]
            if name and any(x and (name == x or (len(name) > 5 and (name in x or x in name))) for x in cn):
                return True
        return False

    short = {i: A[i] for i in final_ids if i in A}
    rows = [{"name": e["name"], "in_pool": match(e, A), "in_shortlist": match(e, short)} for e in ents]
    n = len(rows)
    return {"probe_n": n, "in_pool": sum(r["in_pool"] for r in rows), "in_shortlist": sum(r["in_shortlist"] for r in rows),
            "recall_pool": round(sum(r["in_pool"] for r in rows) / n, 2) if n else None,
            "recall_shortlist": round(sum(r["in_shortlist"] for r in rows) / n, 2) if n else None,
            "entities": rows}


def r5_reach(llm: LLM, spec: CampaignSpec, items) -> Dict[str, Any]:
    blocks = []
    det: List[Dict[str, Any]] = []
    for c, score, _ in items:
        be = c.best_email
        addr = be.address if be else ""
        dom = addr.split("@")[-1].lower() if addr else ""
        local = addr.split("@")[0].lower() if addr else ""
        match = bool(dom and c.org_domain and registrable_domain(dom) == registrable_domain(c.org_domain))
        flags = []
        if not addr:
            flags.append("no_address")
        if dom in FREE_MAIL and c.kind == "org":
            flags.append("free_mail_for_org")
        if dom and c.org_domain and not match and dom not in FREE_MAIL:
            flags.append("domain_differs_from_org")
        if c.kind == "person" and local in {"info", "contact", "hello", "sales", "office", "admin", "enquiries", "team"}:
            flags.append("role_inbox_for_person")
        det.append({"id": c.id, "name": c.name, "address": addr, "tier": contact_tier(c), "flags": flags})
        blocks.append(f"### id: {c.id}\nname: {c.name}  kind: {c.kind}  title: {c.title or '-'}  org: {c.org or '-'}  domain: {c.org_domain or '-'}\n"
                      f"address: {addr or 'NONE'}  method: {be.method if be else '-'}  tier: {contact_tier(c)}\n"
                      f"decision maker: {c.decision_maker or '-'}")
    res: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(blocks), 10):
        data = llm.json(system=REACH_SYSTEM, user=f"ASK: {spec.ask}\nGOAL: {spec.objective}\nTARGET: {spec.persona.description}\n\n"
                        + "\n\n".join(blocks[i:i + 10]), purpose="stage_reach", tier="strong", max_tokens=2000)
        for r in data.get("results", []):
            if isinstance(r, dict):
                res[str(r.get("id"))] = r
    for d in det:
        d.update({k: res.get(d["id"], {}).get(k) for k in ("address_belongs", "addressee_can_grant", "action", "note")})
    n = len(det)
    return {"n": n, "belongs_bad": sum(1 for d in det if d.get("address_belongs") in ("doubtful", "no")),
            "cannot_grant": sum(1 for d in det if d.get("addressee_can_grant") == "no"),
            "not_send": sum(1 for d in det if d.get("action") in ("look_up_person", "drop")),
            "tiers": dict(Counter(d["tier"] for d in det)),
            "det_flags": dict(Counter(f for d in det for f in d["flags"])), "rows": det}


def _tau(pairs: List[tuple]) -> Optional[float]:
    conc = disc = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a = (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1])
            conc += a > 0
            disc += a < 0
    return round((conc - disc) / (conc + disc), 2) if conc + disc else None


def r6_ranking(items, prospect_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fit = {str(p.get("id")): p.get("fit") for p in prospect_rows}
    seq = [(s, fit.get(c.id)) for c, s, _ in items if isinstance(fit.get(c.id), (int, float))]
    n = len(seq)
    third = max(1, n // 3)
    top = [f for _, f in seq[:third]]
    bot = [f for _, f in seq[-third:]]
    inversions = sum(1 for i in range(n) for j in range(i + 1, n) if seq[j][1] - seq[i][1] >= 2)
    orgs = Counter(registrable_domain(c.org_domain or "") or _norm(c.org or c.name) for c, _, _ in items)
    srcs = Counter(s for c, _, _ in items for s in (c.sources[:1] or ["?"]))
    countries = Counter(c.country or "?" for c, _, _ in items)
    return {"n": n, "kendall_tau_score_vs_fit": _tau(seq), "mean_fit_top_third": round(sum(top) / len(top), 2) if top else None,
            "mean_fit_bottom_third": round(sum(bot) / len(bot), 2) if bot else None, "inversions_ge2": inversions,
            "duplicate_orgs": sum(v - 1 for v in orgs.values() if v > 1), "by_first_source": dict(srcs),
            "by_country": dict(countries.most_common(6)), "max_country_share": round(max(countries.values()) / len(items), 2) if items else None}


def r7_drafts(llm: LLM, spec: CampaignSpec, items, emails: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    from brain.draft import _recipient_facts, body_of

    pairs = [(c, emails[c.id]) for c, _, _ in items if c.id in emails][:12]
    if not pairs:
        return {"n": 0}
    rows: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(pairs), 4):
        blocks = [f"### id: {c.id}\nSENDER FACTS:\n{spec.sender.facts_text()[:1200]}\nRECIPIENT EVIDENCE:\n{_recipient_facts(c)}\n"
                  f"SUBJECT: {e['subject']}\nEMAIL:\n{body_of(e['body'])}" for c, e in pairs[i:i + 4]]
        data = llm.json(system=CLAIM_SYSTEM, user="\n\n".join(blocks), purpose="stage_claims", tier="strong", max_tokens=3000)
        for r in data.get("results", []):
            if isinstance(r, dict):
                rows[str(r.get("id"))] = r
    claims = [cl for r in rows.values() for cl in (r.get("claims") or []) if isinstance(cl, dict)]
    st = Counter(cl.get("status") for cl in claims)
    bad = [{"to": next((e["to"] for c, e in pairs if c.id == i), ""), "claims": [cl for cl in (r.get("claims") or [])
            if cl.get("status") in ("unsupported", "contradicted")]} for i, r in rows.items()]
    lint = Counter(f.split(":")[0] for c, e in pairs for f in lint_email(c, e) if f != "placeholder_sender")
    n = len(rows)
    return {"n": n, "claims": len(claims), "supported": st["supported"], "unsupported": st["unsupported"],
            "contradicted": st["contradicted"], "puffery": st["puffery"],
            "claim_support_rate": round(st["supported"] / max(1, len(claims) - st["puffery"]), 2),
            "emails_with_unsupported": sum(1 for b in bad if b["claims"]),
            "mean_specific": round(sum(r.get("specific_to_recipient") or 0 for r in rows.values()) / max(1, n), 2),
            "spammy": sum(1 for r in rows.values() if r.get("spammy")), "length_bad": sum(1 for r in rows.values() if r.get("length_ok") is False),
            "subject_bad": sum(1 for r in rows.values() if r.get("subject_ok") is False), "lint": dict(lint), "bad_claims": bad[:8]}


def r8_reporting(llm: LLM, spec: CampaignSpec, prompt: str, report_row: Dict[str, Any], prospects: List[Dict[str, Any]],
                 lists: Dict[str, Any]) -> Dict[str, Any]:
    v = Counter(p.get("verdict") for p in prospects)
    facts = {"wanted": spec.target_count, "shortlisted": report_row.get("shortlisted_n", len(prospects)),
             "funnel_source": report_row.get("source_funnel"), "warnings_printed": report_row.get("warnings"),
             "pipeline_precision@K_it_reports": report_row.get("precision@K"),
             "independent_review": {"good": v["good"], "borderline": v["borderline"], "bad": v["bad"]},
             "independent_list_issues": lists.get("systemic_issues"), "independent_missing": lists.get("missing")}
    data = llm.json(system=REPORT_SYSTEM, user=f"REQUEST: {prompt}\n\n{json.dumps(facts, indent=1, default=str)[:3500]}",
                    purpose="stage_report", tier="strong", max_tokens=700)
    short = spec.target_count - len(prospects)
    data["det_shortfall"] = short
    data["det_shortfall_without_warning"] = bool(short > 0 and not report_row.get("warnings"))
    data["det_bad_share_hidden"] = bool((v["bad"] + v["borderline"]) / max(1, len(prospects)) >= 0.4 and not report_row.get("warnings"))
    return data


def r9_planning(llm: LLM, spec: CampaignSpec, prompt: str, report_row: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    pb = get_playbook(spec.pack)
    funnel = report_row.get("source_funnel") or {}
    per = {}
    for s in set((funnel.get("unique") or {})) | set(cand.get("by_source") or {}):
        per[s] = {"found": (funnel.get("unique") or {}).get(s, 0), "blind_sample": (cand.get("by_source") or {}).get(s),
                  "shortlisted": (funnel.get("shortlisted") or {}).get(s, 0)}
    facts = {"playbook": pb.name, "playbook_lists_tools": pb.sources, "tools_used": per,
             "search_calls": report_row.get("search_calls"), "llm_calls": report_row.get("llm_paid_calls")}
    return llm.json(system=PLAN_SYSTEM, user=f"{_request_text(spec, prompt)}\n\n{json.dumps(facts, indent=1, default=str)}",
                    purpose="stage_plan", tier="strong", max_tokens=900)


# ── driver ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def make_llm(model: str, max_calls: int) -> LLM:
    key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY (or JUDGE_API_KEY).")
    st = Settings()
    st.llm_base_url = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
    st.llm_api_key = key
    st.llm_model_strong = st.llm_model_fast = model
    st.llm_fallback_models = ""
    st.llm_max_calls = max_calls
    st.llm_reasoning_effort = ""
    return LLM(st, Cache(str(RESULTS / ".llm_judge_cache.sqlite")))


def judge_scenario(llm: LLM, run: str, name: str, report_row: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = RESULTS / f"run_{run}"
    spec, items = _load(run_dir, name)
    emails = dict(_load.emails)
    stages = load_stages(run_dir / f"{name}.db")
    sc_path = SCENARIOS / f"{name}.json"
    scn = json.loads(sc_path.read_text()) if sc_path.exists() else {}
    prompt = scn.get("prompt") or spec.objective
    sender = f"{spec.sender.headline}. {spec.sender.summary}"
    final_ids = {c.id for c, _, _ in items}
    out: Dict[str, Any] = {"run": run, "prompt": prompt, "playbook": spec.pack, "wanted": spec.target_count,
                           "shortlisted": len(items), "pool": len(_pool(stages))}
    out["R1_intake"] = r1_intake(llm, spec, prompt, stages, sender)
    cand: Dict[str, Any] = {}
    if stages.get("discover"):
        cand = r2_r3_candidates(llm, spec, prompt, sample_pool(stages, final_ids, name))
        out["R2_R3_candidates"] = cand
        out["R4_recall"] = r4_recall(llm, spec, prompt, stages, final_ids)
    prospects: List[Dict[str, Any]] = []
    lists: Dict[str, Any] = {}
    if items:
        p = judge_prospects(llm, spec, items)
        prospects = [dict(r) for r in p.get("results", []) if isinstance(r, dict)]
        lists = p.get("list", {})
        by_id = {str(r.get("id")): r for r in prospects}
        out["prospect_verdicts"] = dict(Counter(r.get("verdict") for r in prospects))
        out["prospect_problems"] = dict(Counter(c for r in prospects for c in (r.get("problems") or [])))
        out["prospects"] = [{"name": c.name, "org": c.org, "email": c.best_email.address if c.best_email else None, "score": s,
                             **{k: by_id.get(c.id, {}).get(k) for k in ("verdict", "fit", "why", "problems")}} for c, s, _ in items]
        out["R5_reach"] = r5_reach(llm, spec, items)
        out["R6_ranking"] = r6_ranking(items, prospects)
        out["R7_drafts"] = r7_drafts(llm, spec, items, emails)
    out["list_level"] = lists
    rr = dict(report_row, shortlisted_n=len(items))
    out["R8_reporting"] = r8_reporting(llm, spec, prompt, rr, prospects, lists)
    out["R9_planning"] = r9_planning(llm, spec, prompt, rr, cand)
    return out


def _report_rows(run: str) -> Dict[str, Dict[str, Any]]:
    p = RESULTS / f"{run}.json"
    if not p.exists():
        return {}
    return {r["name"]: r for r in json.loads(p.read_text()).get("rows", []) if "name" in r}


def scorecard(res: Dict[str, Any]) -> None:
    cols = ("intake", "pool_p", "FN_tri", "FN_cap", "FN_fin", "recall", "badaddr", "tau", "fitgap", "claims", "unsup", "honest")
    print(f"\n{'scenario':34} " + " ".join(f"{c:>7}" for c in cols))

    def f(x):
        return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    for name, s in res.items():
        if "error" in s:
            print(f"{name:34} ERROR {s['error']}")
            continue
        c = s.get("R2_R3_candidates") or {}
        st = c.get("strata") or {}
        r6, r7, r5 = s.get("R6_ranking") or {}, s.get("R7_drafts") or {}, s.get("R5_reach") or {}
        gap = (r6.get("mean_fit_top_third") or 0) - (r6.get("mean_fit_bottom_third") or 0) if r6.get("n") else None
        row = [(s.get("R1_intake") or {}).get("overall"), c.get("pool_precision_est"),
               st.get("dropped_triage", {}).get("yes_rate"), st.get("dropped_enrich_cap", {}).get("yes_rate"),
               st.get("dropped_final", {}).get("yes_rate"), (s.get("R4_recall") or {}).get("recall_pool"),
               f"{r5.get('belongs_bad', 0)}/{r5.get('n', 0)}" if r5 else None, r6.get("kendall_tau_score_vs_fit"),
               round(gap, 1) if gap is not None else None, r7.get("claim_support_rate"), r7.get("emails_with_unsupported"),
               (s.get("R8_reporting") or {}).get("truthful")]
        print(f"{name:34} " + " ".join(f"{f(x):>7}" for x in row))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of bench run labels")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4.1"))
    ap.add_argument("--max-calls", type=int, default=600)
    args = ap.parse_args()
    llm = make_llm(args.model, args.max_calls)
    only = {s for s in args.only.split(",") if s}
    res: Dict[str, Any] = {}
    for run in args.runs.split(","):
        rows = _report_rows(run)
        for db in sorted((RESULTS / f"run_{run}").glob("*.db")):
            if only and db.stem not in only:
                continue
            print(f"{db.stem}: judging stages …", flush=True)
            try:
                res[db.stem] = judge_scenario(llm, run, db.stem, rows.get(db.stem, {}))
            except Exception as e:  # one broken scenario must not lose the others
                res[db.stem] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  failed: {e}")
    tag = args.tag or args.runs.replace(",", "_")
    path = RESULTS / f"stage_judge_{tag}.json"
    path.write_text(json.dumps({"runs": args.runs, "judge_model": args.model, "scenarios": res,
                                "cost": {"calls": llm.calls, "usd": round(llm.cost_usd, 3)}}, indent=2, default=str), encoding="utf-8")
    scorecard(res)
    print(f"\nWrote {path}  ({llm.calls} calls, ~${llm.cost_usd:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

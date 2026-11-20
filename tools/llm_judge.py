"""
Independent LLM-as-judge over a saved bench run. No discovery spend: it reads the run's databases.

    python tools/llm_judge.py --run v3_live --drafts evals/results/drafts_v3_live.json

The pipeline judges its own candidates with one model (Groq gpt-oss by default). This judge is a *different*
model from a different vendor (default OpenAI gpt-4.1), sees the same evidence the writer saw, and is
asked to be strict. It grades (a) every shortlisted prospect, (b) the address, (c) the whole list, (d) each draft.

It is a triage tool, not ground truth: it has no web access, so "is this really their address" is judged from
the name, organisation and domain, and "is this person real" from its own knowledge. Human labels
(`brain bench-label`) stay the reference. Output: evals/results/llm_judge_<run>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # keys live in .env
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from brain.cache import Cache  # noqa: E402
from brain.discovery.reach.tiers import contact_tier  # noqa: E402
from brain.llm import LLM  # noqa: E402
from brain.models import CampaignSpec, Contact  # noqa: E402
from brain.settings import Settings  # noqa: E402

RESULTS = ROOT / "evals" / "results"

PROSPECT_CODES = ["wrong_org_type", "wrong_role_or_seniority", "wrong_geography", "off_topic", "weak_evidence",
                  "stale_or_inactive", "aggregator_or_directory", "duplicate_org", "inbox_where_person_needed",
                  "address_not_theirs", "not_a_real_target", "other"]
EMAIL_CODES = ["fabricated_claim", "unsupported_claim", "wrong_salutation", "generic_opening", "flattery",
               "overclaims_sender", "undermines_sender", "ask_unclear", "ask_misfit", "too_long", "wrong_register",
               "ignores_recipient_context", "other"]

PROSPECT_SYSTEM = f"""You are a strict, independent reviewer of a cold-outreach prospecting system. A campaign gives a
goal and an ask; the system returned a shortlist. Judge each prospect AS A SENSIBLE HUMAN SENDER WOULD: would you
really email this person or organisation for this exact ask, at the address given?
Use only the evidence shown and your own general knowledge. Do not invent facts. If you cannot tell, say "unsure".

For each prospect return:
 id; verdict: "good" (email them), "borderline" (plausible, you would hesitate), or "bad";
 fit: 1-5 (5 = ideal target for this ask); problems: list of codes from {PROSPECT_CODES} (empty if none);
 address: "plausible" (looks like theirs/appropriate), "inbox" (shared inbox, fine for an org target),
          "doubtful" (domain or name pattern odd), or "wrong" (clearly another person/organisation);
 address_note: <= 15 words; why: <= 25 words, the decisive reason.
Also return a "list" object judging the shortlist as a whole:
 coverage: 1-5 (does it cover the kinds of target the campaign needs?), diversity: 1-5,
 missing: kinds of target a better list would contain that this one lacks (<= 30 words),
 systemic_issues: up to 3 short strings describing patterns of failure across the list.
Return JSON: {{"results": [...], "list": {{...}}}}. Every id exactly once."""

EMAIL_SYSTEM = f"""You are a hostile, independent editor of cold emails written by an automated system on behalf of a sender.
Assume every email has at least one flaw and look for it: a claim not in the facts, a mismatch between the recipient and
the ask, a greeting that names someone the evidence does not support, anything that reveals how the address was found,
a recipient who could not grant the ask. Each email comes with the PROSPECT VERDICT from an earlier review: an email to
a prospect judged "bad", or to one with fit below 4 or an address judged doubtful/wrong, is never "would_send".
You see the SENDER FACTS (the only things the email may claim about the sender), the RECIPIENT EVIDENCE (the only things
it may claim about the recipient), the campaign ask, and the email. Judge as the recipient and as a careful editor.
For each email return: id;
 grounded: 1-5 (5 = every claim about sender and recipient is supported by the facts above; 1 = invented or distorted);
 unsupported: list of the specific claims you could not find in the facts (empty if none);
 personal: 1-5 (could only have been written to THIS recipient); ask: 1-5 (one clear, easy, appropriate ask);
 tone: 1-5; salutation_ok: true/false (title and name suit this recipient; "Professor" for a non-professor is false);
 reply_odds: "high" | "medium" | "low" (would a busy recipient answer?); would_send: true only if you would send it unedited;
 problems: list of codes from {EMAIL_CODES}; fix: the single most useful change, <= 25 words.
Return JSON: {{"results": [...]}}. Every id exactly once."""


def _contact_view(c: Contact, score: float, reasons: List[str]) -> str:
    be = c.best_email
    lines = [f"name: {c.name}  kind: {c.kind}  title: {c.title or '-'}  org: {c.org or '-'}  country: {c.country or '-'}",
             f"address: {be.address if be else 'NONE'}  (pipeline tier: {contact_tier(c)}, method: "
             f"{be.method if be else '-'}, confidence: {be.confidence if be else 0:.2f})",
             f"sources: {', '.join(c.sources) or '-'}  pipeline score: {score:.2f}",
             f"pipeline reasons: {' | '.join(reasons[:2]) or '-'}"]
    if c.decision_maker:
        lines.append(f"decision maker: {c.decision_maker}")
    if c.topics:
        lines.append("topics: " + ", ".join(c.topics[:6]))
    if c.bio:
        lines.append("bio: " + c.bio[:300])
    for ev in c.evidence[:5]:
        lines.append(f"- [{ev.kind}] {ev.text[:160]}" + (f" ({ev.year})" if ev.year else "")
                     + (f": {ev.detail[:180]}" if ev.detail else ""))
    return "\n".join(lines)


def _load(run_dir: Path, name: str):
    db = sqlite3.connect(run_dir / f"{name}.db")
    spec_json, = db.execute("select spec_json from brain_campaigns limit 1").fetchone()
    spec = CampaignSpec.model_validate_json(spec_json)
    rows = db.execute("select r.contact_id, r.score, r.reasons_json, c.data_json, r.status, r.email, r.subject, r.body "
                      "from brain_results r join brain_contacts c on c.id = r.contact_id "
                      "where r.status in ('shortlisted', 'needs_review', 'drafted', 'approved') "
                      "order by r.score desc").fetchall()
    items, seen, emails = [], set(), {}
    for cid, score, reasons, data, status, addr, subject, body in rows:
        if cid in seen:
            continue
        seen.add(cid)
        if len(items) >= max(spec.target_count, 1):
            continue
        c = Contact.model_validate_json(data)
        c.id = cid
        items.append((c, float(score or 0), json.loads(reasons or "[]")))
        if body:
            emails[cid] = {"to": addr or "", "subject": subject or "", "body": body}
    _load.emails = emails  # drafts the run itself stored
    return spec, items


HEADING_WORDS = {"governance", "spotlight", "workplace", "partner", "partners", "corporate", "about", "contact",
                 "careers", "newsroom", "overview", "sponsors", "team", "services", "solutions", "products"}
STOP_NAMES = {"at", "the", "of", "and", "team", "sir", "madam", "mr", "mrs", "ms", "dr"}
ROLE_LOCALS = {"info", "sales", "hello", "team", "connect", "build", "marketing", "apac", "partners", "contact",
               "support", "admin", "office", "enquiry", "inquiry", "press", "media", "tips", "news", "editor"}


def lint_email(contact: Contact, e: Dict[str, Any]) -> List[str]:
    """Deterministic defects an LLM judge reliably misses (the judge approved all of these)."""
    import re

    body = e.get("body") or ""
    text = body.split("\n--\n")[0]  # without the opt-out footer
    lines = text.split("\n")
    greeting = lines[0].strip() if lines else ""
    flags: List[str] = []
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", "\n".join(lines[1:])):
        flags.append("raw_address_in_body")
    if re.search(r"\binbox\b|hoping this (message )?reaches", text, re.I):
        flags.append("pipeline_meta_in_body")
    m = re.match(r"(?:Hi|Hello|Dear)\s+(.+?),?$", greeting)
    if m:
        name = m.group(1).strip()
        first = name.split()[0].lower().strip(".,")
        if "|" in name or len(first) <= 2 or first in STOP_NAMES:
            flags.append(f"junk_greeting:{name}")
        local = (e.get("to") or "").split("@")[0].lower()
        if local in ROLE_LOCALS and not re.search(r"team|folks|everyone", name, re.I) and contact.decision_maker is None:
            flags.append(f"person_greeting_on_role_inbox:{name}")
    if contact.kind == "person" and not contact.title and set(re.findall(r"[a-z]+", contact.name.lower())) & HEADING_WORDS:
        flags.append(f"page_heading_as_person:{contact.name}")
    if re.search(r"\bTest (Founder|Student|Manager|Organiser|Candidate)\b", text):
        flags.append("placeholder_sender")  # bench artefact, ignore in reports
    return flags


def judge_prospects(llm: LLM, spec: CampaignSpec, items) -> Dict[str, Any]:
    head = (f"CAMPAIGN GOAL: {spec.objective}\nASK: {spec.ask}\nTARGET: {spec.persona.description} "
            f"(roles: {', '.join(spec.persona.roles) or 'any'}; org types: {', '.join(spec.persona.org_types) or 'any'})\n"
            f"TOPICS: {', '.join(spec.topics[:8])}\nWHERE: {', '.join(spec.regions + spec.countries) or 'anywhere'}\n"
            f"SENDER: {spec.sender.headline} {spec.sender.summary}\n\nSHORTLIST ({len(items)}):\n")
    results: List[Dict[str, Any]] = []
    lists: List[Dict[str, Any]] = []
    for i in range(0, len(items), 8):  # large shortlists: 8 per call keeps the answers complete
        chunk = items[i:i + 8]
        body = "\n\n".join(f"### id: {c.id}\n{_contact_view(c, s, r)}" for c, s, r in chunk)
        data = llm.json(system=PROSPECT_SYSTEM, user=head.replace(f"({len(items)})", f"(part {i // 8 + 1}, {len(chunk)} of {len(items)})") + body,
                        purpose="judge_prospects", tier="strong", max_tokens=2800)
        results += [r for r in data.get("results", []) if isinstance(r, dict)]
        if isinstance(data.get("list"), dict):
            lists.append(data["list"])
    merged: Dict[str, Any] = {}
    if lists:  # whole-list judgement: average the grades, join the notes
        for k in ("coverage", "diversity"):
            vals = [l[k] for l in lists if isinstance(l.get(k), (int, float))]
            merged[k] = round(sum(vals) / len(vals), 1) if vals else None
        merged["missing"] = " | ".join(str(l.get("missing")) for l in lists if l.get("missing"))[:400]
        merged["systemic_issues"] = [x for l in lists for x in (l.get("systemic_issues") or [])][:6]
    return {"results": results, "list": merged}


def judge_emails(llm: LLM, spec: CampaignSpec, pairs, verdicts: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    from brain.draft import _recipient_facts, body_of

    head = (f"SENDER FACTS:\n{spec.sender.facts_text()[:2500]}\nGOAL: {spec.objective}\nASK: {spec.ask}\n\n")
    verdicts = verdicts or {}

    def verdict(cid: str) -> str:
        v = verdicts.get(cid) or {}
        return (f"PROSPECT VERDICT: {v.get('verdict')} (fit {v.get('fit')}, address {v.get('address')}, "
                f"problems {v.get('problems')}): {v.get('why')}\n") if v else ""

    blocks = [f"### id: {c.id}\n{verdict(c.id)}RECIPIENT EVIDENCE:\n{_recipient_facts(c)}\nTO: {e['to']}\nSUBJECT: {e['subject']}\n"
              f"EMAIL:\n{body_of(e['body'])}" for c, e in pairs]
    return llm.json(system=EMAIL_SYSTEM, user=head + "\n\n".join(blocks), purpose="judge_emails", tier="strong",
                    max_tokens=2500)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="v3_live", help="bench run label (evals/results/run_<label>)")
    ap.add_argument("--drafts", default="", help="bench-drafts JSON with the emails to grade (optional)")
    ap.add_argument("--only", default="", help="comma list of scenario names")
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4.1"))
    ap.add_argument("--max-calls", type=int, default=60)
    args = ap.parse_args()

    key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("Set OPENAI_API_KEY (or JUDGE_API_KEY): the judge must be a different model from the pipeline.")
        return 2
    st = Settings()
    st.llm_base_url = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
    st.llm_api_key = key
    st.llm_model_strong = st.llm_model_fast = args.model
    st.llm_fallback_models = ""
    st.llm_max_calls = args.max_calls
    st.llm_reasoning_effort = ""
    llm = LLM(st, Cache(str(RESULTS / ".llm_judge_cache.sqlite")))

    run_dir = RESULTS / f"run_{args.run}"
    emails_by_scn: Dict[str, List[Dict[str, Any]]] = {}
    if args.drafts:
        for row in json.loads(Path(args.drafts).read_text())["rows"]:
            emails_by_scn[row["name"]] = row.get("emails") or []
    only = {s for s in args.only.split(",") if s}
    out: Dict[str, Any] = {"run": args.run, "judge_model": args.model, "scenarios": {}}

    for db in sorted(run_dir.glob("*.db")):
        name = db.stem
        if only and name not in only:
            continue
        spec, items = _load(run_dir, name)
        if not items:
            out["scenarios"][name] = {"error": "empty shortlist"}
            print(f"{name}: empty shortlist")
            continue
        print(f"{name}: judging {len(items)} prospects …")
        p = judge_prospects(llm, spec, items)
        by_id = {str(r.get("id")): r for r in p.get("results", []) if isinstance(r, dict)}
        prospects = []
        for c, s, reasons in items:
            r = by_id.get(c.id, {})
            be = c.best_email
            prospects.append({"id": c.id, "name": c.name, "kind": c.kind, "title": c.title, "org": c.org,
                              "email": be.address if be else None, "tier": contact_tier(c), "score": s,
                              "sources": c.sources, "pipeline_reason": reasons[:1], **r})
        sc: Dict[str, Any] = {"playbook": spec.pack, "objective": spec.objective, "ask": spec.ask,
                              "prospects": prospects, "list": p.get("list", {})}
        emails = emails_by_scn.get(name) or [dict(e, to=e.get("to") or "") for e in _load.emails.values()]
        if emails:
            by_addr = {(c.best_email.address if c.best_email else ""): c for c, _, _ in items}
            pairs = [(by_addr[e["to"]], e) for e in emails if e.get("to") in by_addr][:12]
            if pairs:
                print(f"{name}: judging {len(pairs)} emails …")
                g = judge_emails(llm, spec, pairs, by_id)
                eb = {str(r.get("id")): r for r in g.get("results", []) if isinstance(r, dict)}
                sc["emails"] = [{"id": c.id, "to": e["to"], "subject": e["subject"], "body": e["body"],
                                 "pipeline_grade": e.get("grade"),
                                 "lint": lint_email(c, e), **eb.get(c.id, {})} for c, e in pairs]
        out["scenarios"][name] = sc

    out["cost"] = {"paid_calls": llm.calls, "est_usd": round(llm.cost_usd, 4)}
    path = RESULTS / f"llm_judge_{args.run}{'_partial' if only else ''}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    _print_summary(out)
    print(f"\nWrote {path}  ({llm.calls} paid calls, ~${llm.cost_usd:.3f})")
    return 0


def _print_summary(out: Dict[str, Any]) -> None:
    print(f"\n{'scenario':32} {'n':>2} {'good':>4} {'bord':>4} {'bad':>3} {'fit':>4} {'addr✗':>5} {'send':>5} {'ground':>6}")
    codes: Counter = Counter()
    for name, sc in out["scenarios"].items():
        ps = sc.get("prospects") or []
        if not ps:
            continue
        v = Counter(p.get("verdict") for p in ps)
        fit = sum(p.get("fit") or 0 for p in ps) / len(ps)
        bad_addr = sum(1 for p in ps if p.get("address") in ("doubtful", "wrong"))
        es = sc.get("emails") or []
        send = f"{sum(1 for e in es if e.get('would_send'))}/{len(es)}" if es else "-"
        gr = f"{sum(e.get('grounded') or 0 for e in es) / len(es):.1f}" if es else "-"
        print(f"{name:32} {len(ps):>2} {v['good']:>4} {v['borderline']:>4} {v['bad']:>3} {fit:>4.1f} {bad_addr:>5} {send:>5} {gr:>6}")
        for p in ps:
            codes.update(p.get("problems") or [])
    print("\nProspect problems:", dict(codes.most_common()))
    lint: Counter = Counter()
    for sc in out["scenarios"].values():
        for e in sc.get("emails") or []:
            lint.update(f.split(":")[0] for f in e.get("lint", []) if f != "placeholder_sender")
    print("Email lint (deterministic, missed by the judge):", dict(lint.most_common()))


if __name__ == "__main__":
    raise SystemExit(main())

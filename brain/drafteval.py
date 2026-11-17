"""
Measuring email quality (docs/intent-and-email-plan.md step 8).

    grade(items)          one batched fast-model call per 5 emails: a rubric (1-5) for specificity, relevance of the
                          sender fact, clarity of the ask, tone and honesty, a would-send verdict and a one-line
                          critique. An LLM grading an LLM: reported next to the deterministic checks, never used to send.
    summarise(drafts)     pass rate, the most common check failures, rubric averages, would-send rate
    draft_eval(run)       drafts + fact-check + grading for the top shortlisted contacts of a bench run, from its stored
                          checkpoints — no discovery spend (`brain bench-drafts --run <label>`)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from brain.models import Contact, Draft
from brain.textutil import squash

log = logging.getLogger("brain.drafteval")

RUBRIC_SYSTEM = """You review cold emails for a sender who will only send excellent ones. For each email score 1-5:
 specificity  it could only have been written to THIS recipient (5) vs. a template with the name swapped in (1)
 relevance    the sender facts chosen are the ones that matter to this recipient
 ask          exactly one clear, easy-to-answer ask
 tone         respectful, plain, confident; no flattery, hype or filler
 honesty      no overclaiming about the sender or the recipient, judged against the facts given
Then "would_send": true only if you would send it unedited, and "critique": the single most useful fix (under 20 words).
Return JSON: {"results": [{"id": "<id>", "specificity": n, "relevance": n, "ask": n, "tone": n, "honesty": n,
"would_send": true|false, "critique": "<text>"}]}. Include every id once."""

DIMENSIONS = ("specificity", "relevance", "ask", "tone", "honesty")


def grade(items: List[Tuple[Contact, Draft]], spec, llm, batch_size: int = 5) -> Dict[str, Dict[str, Any]]:
    """{contact_id: rubric} for the given (contact, draft) pairs."""
    from brain.draft import _recipient_facts, body_of

    out: Dict[str, Dict[str, Any]] = {}
    sender = spec.sender.facts_text()[:2000]
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        blocks = [f"### id: {c.id}\nRECIPIENT FACTS:\n{_recipient_facts(c)}\nSUBJECT: {d.subject}\nEMAIL:\n{body_of(d.body)}"
                  for c, d in batch]
        user = f"SENDER FACTS:\n{sender}\nGOAL: {spec.objective}\nASK: {spec.ask}\n\n" + "\n\n".join(blocks)
        try:
            data = llm.json(system=RUBRIC_SYSTEM, user=user, purpose="draft_grade", tier="fast",
                            max_tokens=120 * len(batch) + 100)
        except Exception as e:
            log.warning("grading failed: %s", e)
            continue
        ids = {c.id for c, _ in batch}
        for r in (data.get("results") or []) if isinstance(data, dict) else []:
            if isinstance(r, dict) and str(r.get("id")) in ids:
                row: Dict[str, Any] = {}
                for k in DIMENSIONS:
                    try:
                        row[k] = max(1, min(5, int(r.get(k))))
                    except (TypeError, ValueError):
                        row[k] = None
                row["would_send"] = bool(r.get("would_send"))
                row["critique"] = squash(str(r.get("critique") or ""), 160)
                out[str(r["id"])] = row
    return out


def summarise(drafts: Dict[str, Draft], grades: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    n = len(drafts)
    if not n:
        return {"drafts": 0}
    issues = Counter()
    for d in drafts.values():
        for i in d.issues:
            issues[i.split(":")[0].split("(")[0].strip()[:60]] += 1
    out: Dict[str, Any] = {"drafts": n, "pass_rate": round(sum(1 for d in drafts.values() if d.ok) / n, 3),
                           "top_issues": issues.most_common(6)}
    g = [x for x in (grades or {}).values() if x]
    if g:
        for k in DIMENSIONS:
            vals = [x[k] for x in g if x.get(k)]
            out[f"avg_{k}"] = round(sum(vals) / len(vals), 2) if vals else None
        out["would_send_rate"] = round(sum(1 for x in g if x["would_send"]) / len(g), 3)
        out["critiques"] = [x["critique"] for x in g if x.get("critique")][:5]
    return out


def draft_eval(run_label: str, only: Optional[List[str]] = None, per: int = 3, progress=print) -> Dict[str, Any]:
    """Write, fact-check and grade drafts for the top `per` shortlisted contacts of each scenario of a bench run."""
    import sqlite3

    from brain.bench import CACHE_DIR, RESULT_DIR, load_scenarios, spec_from_scenario
    from brain.cache import Cache
    from brain.context import CampaignContext
    from brain.draft import draft_email, verify_and_fix
    from brain.llm import LLM
    from brain.packs import get_pack
    from brain.settings import Settings

    st = Settings()
    rows, all_drafts, all_grades = [], {}, {}
    for sc in load_scenarios(only):
        db = RESULT_DIR / f"run_{run_label}" / f"{sc['name']}.db"
        if not db.exists():
            continue
        con = sqlite3.connect(str(db))
        stages = {s: json.loads(d) for s, d in con.execute("select stage, payload_json from brain_stages")}
        con.close()
        short = [Contact(**c) for c in (stages.get("judge") or {}).get("contacts", [])][:per]
        if not short:
            continue
        spec = spec_from_scenario(sc)
        pack = get_pack(spec.pack)
        llm = LLM(st, Cache(str(CACHE_DIR / f"{sc['name']}.sqlite")))
        ctx = CampaignContext(understanding=spec.objective, target_brief=spec.persona.description)
        for f in sc.get("facts", []):
            ctx.add_fact(f, "scenario")
        drafts = {}
        for c in short:
            d = draft_email(c, spec, pack, llm, ctx=ctx)
            if d is not None:
                drafts[c.id] = d
        fact = verify_and_fix(short, drafts, spec, pack, llm, ctx=ctx)
        grades = grade([(c, drafts[c.id]) for c in short if c.id in drafts], spec, llm)
        summary = summarise(drafts, grades)
        summary.update({"name": sc["name"], "fact_check": fact, "llm": llm.report(),
                        "emails": [{"to": d.to, "name": c.name, "subject": d.subject, "body": d.body, "issues": d.issues,
                                    "grade": grades.get(c.id)} for c in short if (d := drafts.get(c.id))]})
        rows.append(summary)
        all_drafts.update(drafts)
        all_grades.update(grades)
        progress(f"[drafts] {sc['name']}: {summary.get('drafts')} drafts, pass {summary.get('pass_rate')}, "
                 f"would-send {summary.get('would_send_rate')}")
    report = {"run": run_label, "rows": rows, "summary": summarise(all_drafts, all_grades),
              "llm_usd": round(sum(r["llm"].get("est_cost_usd", 0) for r in rows), 4)}
    path = RESULT_DIR / f"drafts_{run_label}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["path"] = str(path)
    return report

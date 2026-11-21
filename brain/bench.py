"""
Benchmarks for discovery: the same scenarios, run the same way, scored the same way.

    python main.py brain bench                      # every scenario in evals/scenarios, live, results saved
    python main.py brain bench --only sponsors_fest # one scenario
    python main.py brain bench --replay             # re-run from recorded answers only: free, deterministic
    python main.py brain bench --compare evals/results/baseline.json

Each scenario is a JSON file holding a ready campaign spec (so no intake LLM call is spent) and,
optionally, labels {"good": [...], "bad": [...]}. Runs stop after judging (drafting is not discovery
and costs one call per email) unless --draft is given.

Every scenario records its network answers in its own cache file (evals/cache/<name>.sqlite), so a
later `--replay` reproduces the run exactly without any search, fetch or (with --strict) LLM spend.
That is what makes before/after comparisons of a code change cheap and fair.

Metrics (per scenario, and summed/averaged):
    unique, qualified, reachable, passed_judge, shortlisted   the funnel
    good@K        shortlisted candidates with judge score >= 0.7, K = target_count
    precision@K   good@K / K
    distinct_orgs organisations among the shortlist (diversity)
    proof_rate    shortlisted candidates carrying direct channel proof (sponsored, hiring, portfolio...)
    cost          search calls, paid LLM calls, est. LLM USD, pages fetched, seconds
    usd_per_good  (LLM USD + search calls x SERPER_USD) / good@K
    label_precision@K  against the scenario's labels, if any (brain/evalkit.py)
    good_reachable@K   good AND the address is attributed / verified / a shared inbox (not a guess or a namesake's)
    good_verified@K    good AND a mailbox check confirmed the address (brain/discovery/reach/tiers.py)

Honest reading of the numbers:
    * "good" is the pipeline's own LLM judge: treat it as a proxy. Human labels are the truth (`brain bench-label`).
    * every pooled figure carries a 95% Wilson interval; with a few dozen slots, differences inside it are noise.
    * scenarios have a "split": `dev` (tuned on) or `holdout` (never tuned on). Report both; trust holdout.
    * `--repeat N` re-runs each scenario with a fresh cache, so web and model noise show up instead of being replayed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain.models import CampaignSpec, ProspectProfile, SenderProfile, TargetPersona
from brain.textutil import split_locations

log = logging.getLogger("brain.bench")

SCENARIO_DIR = Path("evals/scenarios")
LABEL_DIR = Path("evals/labels")
RESULT_DIR = Path("evals/results")
CACHE_DIR = Path("evals/cache")
SERPER_USD = 0.001  # approximate price of one search call
GOOD = 0.7


def label_ident(c: Dict[str, Any]) -> Optional[str]:
    """How a label sheet row is identified: its address when it has one. A person with no address is "Name | Org",
    so that a label on one Priya Nair is not applied to a different Priya Nair at another organisation."""
    if c.get("email"):
        return c["email"]
    name = c.get("name")
    if name and c.get("kind", "person") == "person" and c.get("org"):
        return f"{name} | {c['org']}"
    return name


def load_labels(name: str, folder: Path = LABEL_DIR) -> Dict[str, List[str]]:
    """Human labels for a scenario from evals/labels/<name>.json: {"good": [...], "bad": [...]} and/or a label
    sheet's "candidates" with "label": "good" | "bad" | "wrong_address" (see export_label_sheet). Each identity is
    listed once per verdict, however many places it was written."""
    path = folder / f"{name}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    good, bad, wrong = list(data.get("good") or []), list(data.get("bad") or []), list(data.get("wrong_address") or [])
    for c in data.get("candidates") or []:
        verdict = str(c.get("label") or "").strip().lower()
        ident = label_ident(c)
        if ident and verdict in ("good", "bad", "wrong_address"):
            {"good": good, "bad": bad, "wrong_address": wrong}[verdict].append(ident)
    return {"good": list(dict.fromkeys(good)), "bad": list(dict.fromkeys(bad)), "wrong_address": list(dict.fromkeys(wrong))}


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write then rename, so a crash or Ctrl-C mid-write cannot leave a half-written (unreadable) label sheet."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


SPLITS = ("dev", "holdout", "stress")


def _scenario_problems(data: Dict[str, Any], stem: str) -> List[str]:
    """What is wrong with a scenario file. A typo here (a split called "holdout ", a playbook that does not exist and so
    falls back to "general") would not crash anything: it would quietly change what a benchmark measures."""
    from brain.playbooks import PLAYBOOKS

    bad = []
    if data.get("split") not in SPLITS:
        bad.append(f"split {data.get('split')!r} is not one of {', '.join(SPLITS)}")
    for pb in [data.get("playbook")] + list(data.get("playbook_ok") or []):
        if pb not in PLAYBOOKS:
            bad.append(f"playbook {pb!r} does not exist")
    if not (data.get("prompt") or data.get("spec")):
        bad.append("has neither a prompt nor a spec")
    try:
        if int(data.get("target_count", 8)) < 1:
            bad.append("target_count must be at least 1")
    except (TypeError, ValueError):
        bad.append(f"target_count {data.get('target_count')!r} is not a number")
    return [f"{stem}: {b}" for b in bad]


def load_scenarios(only: Optional[List[str]] = None, folder: Path = SCENARIO_DIR,
                   label_dir: Path = LABEL_DIR, split: str = "") -> List[Dict[str, Any]]:
    """Scenario files, validated as a set (every file is checked, not only the ones selected, so a bad file is caught
    the day it is added). Raises ValueError listing every problem."""
    if split and split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}: use one of {', '.join(SPLITS)}")
    out, problems, seen = [], [], {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("name", path.stem)
        data.setdefault("split", "dev")
        problems += _scenario_problems(data, path.stem)
        if data["name"] in seen:
            problems.append(f"{path.stem}: name {data['name']!r} is already used by {seen[data['name']]}")
        seen[data["name"]] = path.stem
        if only and data["name"] not in only:
            continue
        if split and data["split"] != split:
            continue
        extra = load_labels(data["name"], label_dir)
        if extra.get("good") or extra.get("bad") or extra.get("wrong_address"):
            labels = data.setdefault("labels", {})
            for key in ("good", "bad", "wrong_address"):
                labels[key] = list(dict.fromkeys((labels.get(key) or []) + extra.get(key, [])))
        out.append(data)
    if problems:
        raise ValueError("Scenario files need fixing:\n  - " + "\n  - ".join(problems))
    return out


def export_label_sheet(run_label: str, name: str, folder: Path = LABEL_DIR, limit: int = 40) -> Path:
    """Write evals/labels/<name>.json from a bench run: the candidates that reached enrichment (best first),
    each with what a human needs to decide, and an empty "label" to set to "good" or "bad". Labels already in
    the sheet are kept, so re-exporting after a new run only adds new candidates."""
    import sqlite3

    db = RESULT_DIR / f"run_{run_label}" / f"{name}.db"
    if not db.exists():
        raise FileNotFoundError(f"No bench database {db}; run `brain bench --label {run_label}` first")
    con = sqlite3.connect(str(db))
    stages = {s: json.loads(d) for s, d in con.execute("select stage, payload_json from brain_stages")}
    con.close()
    judged = {c["id"]: c for c in (stages.get("judge") or {}).get("contacts", [])}
    pool = (stages.get("enrich") or stages.get("rank") or {}).get("contacts", [])
    # every candidate the judge saw, with the signals it used: a label on one of these becomes training data for
    # brain/calibrate.py (the shortlist alone would hold only the good ones)
    signals = {r["id"]: r for r in ((stages.get("judge") or {}).get("extra") or {}).get("judged", [])}
    have = {c["id"] for c in pool}
    pool = sorted(pool, key=lambda c: c["id"] not in signals)  # stable: judged candidates first
    pool += [{"id": i, "name": r["name"], "kind": "person", "emails": [{"address": r["email"], "confidence": 1.0}]
              if r.get("email") else [], "org": None, "sources": [], "evidence": []}
             for i, r in signals.items() if i not in have]
    path = folder / f"{name}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    known = {label_ident(c): c for c in old.get("candidates") or []}
    rows = list(old.get("candidates") or [])
    for c in pool[:limit]:
        best = max(c.get("emails") or [{}], key=lambda e: e.get("confidence", 0))
        ident = label_ident({"email": best.get("address"), "name": c["name"], "kind": c["kind"], "org": c.get("org")})
        if ident in known:
            continue
        j = judged.get(c["id"], {})
        rows.append({"label": "", "name": c["name"], "kind": c["kind"], "org": c.get("org"), "email": best.get("address"),
                     "sources": c.get("sources"), "title": c.get("title"),
                     "evidence": [e["text"][:200] for e in (c.get("evidence") or [])[:3]],
                     "url": (c.get("urls") or [None])[0], "judge_score": j.get("score"),
                     "judge_reason": (j.get("reasons") or [None])[0],
                     **({"features": signals[c["id"]]["features"]} if c["id"] in signals else {})})
    folder.mkdir(parents=True, exist_ok=True)
    out = {"scenario": name, "how": "Set each candidate's \"label\" to \"good\", \"bad\" or \"wrong_address\" "
                                   "(leave \"\" if unsure). good = you would really email them for this scenario's "
                                   "ask; wrong_address = right person, but the address shown is not theirs.",
           "good": old.get("good", []), "bad": old.get("bad", []),
           "wrong_address": old.get("wrong_address", []), "candidates": rows}
    _write_json_atomic(path, out)
    return path


def spec_from_scenario(sc: Dict[str, Any]) -> CampaignSpec:
    raw = dict(sc["spec"])
    countries, regions = split_locations(raw.pop("locations", []) or [])
    persona = TargetPersona(**raw.pop("persona", {}))
    profile = ProspectProfile(**raw.pop("profile", {})) if raw.get("profile") else ProspectProfile()
    raw.pop("profile", None)
    sender = SenderProfile(**sc.get("sender", {}))
    return CampaignSpec(id=f"bench_{sc['name']}"[:32], pack=sc.get("playbook", "general"), persona=persona,
                        profile=profile, countries=countries, regions=regions, sender=sender,
                        target_count=int(sc.get("target_count", 8)), confirmed=True, **raw)


def metrics(result, spec: CampaignSpec, labels: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    from brain.fit import SIGNAL_KEYS

    f = result.funnel or {}
    cost = f.get("cost") or {}
    llm = cost.get("llm") or {}
    short = result.shortlist
    k = spec.target_count
    good = [c for c in short if (c.score or 0) >= GOOD]
    from brain.discovery.reach.tiers import contact_tier, counts as tier_counts

    tier_of = {c.id: contact_tier(c) for c in short}
    reachable_tiers = ("verified", "attributed", "inbox")
    usd = float(llm.get("est_cost_usd") or 0) + SERPER_USD * float(cost.get("search_calls") or 0)
    m = {
        "unique": f.get("unique_people", 0), "qualified": f.get("qualified", 0), "reachable": f.get("reachable", 0),
        "passed_judge": f.get("passed_judge", 0), "shortlisted": len(short),
        "good@K": len(good), "precision@K": round(len(good) / k, 3) if k else None,
        "good_reachable@K": sum(1 for c in good if tier_of[c.id] in reachable_tiers),
        "good_verified@K": sum(1 for c in good if tier_of[c.id] == "verified"),
        "address_tiers": tier_counts(short),
        "mean_score": round(sum(c.score or 0 for c in short) / len(short), 3) if short else 0.0,
        "distinct_orgs": len({(c.org_domain or c.org or c.name).lower() for c in short}),
        "proof_rate": round(sum(1 for c in short if any(c.signals.get(s, 0) > 0 for s in SIGNAL_KEYS
                                                        if s not in ("publications",))) / len(short), 3) if short else 0.0,
        "personal_email_rate": round(sum(1 for c in short if c.kind == "person") / len(short), 3) if short else 0.0,
        "search_calls": cost.get("search_calls", 0), "llm_paid_calls": llm.get("paid_calls", 0),
        "llm_usd": llm.get("est_cost_usd", 0), "pages_fetched": cost.get("pages_fetched", 0),
        "seconds": cost.get("seconds", 0), "usd_per_good": round(usd / len(good), 4) if good else None,
        "llm_by_purpose": llm.get("by_purpose", {}), "by_source": f.get("by_source", {}),
        "replay_misses": cost.get("replay_misses", 0), "warnings": result.warnings[:6],
        "source_funnel": f.get("source_funnel", {}), "ledger": f.get("ledger", {}),
        "shortlist": [{"name": c.name, "org": c.org, "email": c.best_email.address if c.best_email else None,
                       "score": c.score, "sources": c.sources, "reasons": c.reasons[:1]} for c in short],
    }
    problem = f.get("provider_problem") or ""
    if problem:  # a failing search/GitHub account makes the numbers a measure of the account, not the pipeline
        m["provider_problem"] = problem
        if not short or not f.get("unique_people"):
            m["invalid"] = "provider_failure"  # discovery came back empty because calls failed: not a real zero
    if labels and (labels.get("good") or labels.get("bad") or labels.get("wrong_address")):
        from brain.evalkit import evaluate_labels

        rows = [{"name": c.name, "email": c.best_email.address if c.best_email else "", "score": c.score,
                 "kind": c.kind, "org": c.org or ""} for c in short]
        ev = evaluate_labels(rows, labels, ks=(k,))
        m["label_precision@K"] = ev.get(f"precision@{k}")
        m["label_precision@K_ci95"] = ev.get(f"precision@{k}_ci95")
        m["label_good_k"], m["label_n_k"] = ev.get(f"good@{k}_labeled", 0), ev.get(f"labeled@{k}", 0)
        m["label_wrong_address"] = ev.get("wrong_address", 0)
        if ev.get("conflicts"):
            m["label_conflicts"] = ev["conflicts"]
        m["labeled"] = ev.get("labeled")
    return m


def run_scenario(sc: Dict[str, Any], *, replay: str = "", draft: bool = False, out_dir: Path,
                 from_prompt: bool = True, cache_tag: str = "") -> Dict[str, Any]:
    from brain.cache import Cache
    from brain.context import CampaignContext
    from brain.fetch import Fetcher
    from brain.llm import LLM
    from brain.pipeline import Deps, run_campaign
    from brain.settings import Settings
    from brain.sources.openalex import OpenAlexSource
    from brain.sources.orcid import Orcid
    from brain.sources.websearch import WebSearch
    from brain.store import Store

    st = Settings()
    st.replay = replay
    st.database_url = f"sqlite:///{out_dir / (sc['name'] + '.db')}"  # isolated: no memory leaks between scenarios
    st.use_memory = bool(sc.get("use_memory", False))
    for k, v in (sc.get("settings") or {}).items():
        setattr(st, k, v)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = Cache(str(CACHE_DIR / f"{sc['name']}{cache_tag}.sqlite"), ignore_ttl=bool(replay))
    deps = Deps(settings=st, cache=cache, llm=LLM(st, cache), fetcher=Fetcher(st, cache), search=WebSearch(st, cache),
                orcid=Orcid(st, cache), openalex=OpenAlexSource(st, cache), store=Store(st.database_url))
    intake: Dict[str, Any] = {}
    if from_prompt and sc.get("prompt") and deps.llm.available:
        # End to end: the request as a user writes it goes through the intake, as in a real campaign.
        from brain.intake import ScriptedUI, run_intake

        sender = SenderProfile(**sc.get("sender", {}))
        res = run_intake(sc["prompt"], deps.llm, ScriptedUI([]), sender, interactive=False,
                         max_searches_ceiling=st.max_search_calls, target_count=int(sc.get("target_count", 8)))
        spec, ctx = res.spec, res.ctx
        spec.id = f"bench_{sc['name']}"[:32]
        ok = [sc.get("playbook")] + list(sc.get("playbook_ok") or [])
        want = spec_from_scenario(sc)
        intake = {"intake_playbook": spec.pack, "intake_playbook_ok": spec.pack in ok,
                  "intake_kind_ok": spec.target_kind == (want.target_kind or spec.target_kind),
                  "intake_where": spec.regions + spec.countries,
                  "intake_where_ok": set(want.countries) <= set(spec.countries)
                  and all(any(r.lower() in x.lower() or x.lower() in r.lower() for x in spec.regions) for r in want.regions),
                  "intake_topics": spec.topics[:5], "intake_ask_type": spec.ask_detail.type,
                  "intake_confidence": spec.playbook_confidence, "intake_llm_calls": res.llm_calls}
    else:
        spec = spec_from_scenario(sc)
        ctx = CampaignContext(understanding=spec.objective, target_brief=spec.persona.description)
    for f in sc.get("facts", []):
        ctx.add_fact(f, "scenario")
    for name in sc.get("seeds", []):
        ctx.add_seed(name)
    started = time.time()
    try:
        result = run_campaign(spec, deps, upto="draft" if draft else "judge", resume=False, ctx=ctx)
    except Exception as e:  # one broken scenario must not stop the benchmark
        log.exception("scenario %s failed", sc["name"])
        return {"name": sc["name"], "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - started, 1),
                "split": sc.get("split", "dev"), "target": int(sc.get("target_count", 8))}
    m = metrics(result, spec, sc.get("labels"))
    m.update(intake)
    if draft and result.drafts:
        from brain.drafteval import grade, summarise

        by_id = {c.id: c for c in result.shortlist}
        grades = grade([(by_id[cid], d) for cid, d in result.drafts.items() if cid in by_id], spec, deps.llm)
        m["drafts"] = summarise(result.drafts, grades)
        m["emails"] = [{"to": d.to, "subject": d.subject, "body": d.body, "issues": d.issues, "grade": grades.get(cid)}
                       for cid, d in result.drafts.items()]
        m["llm_usd"] = deps.llm.report().get("est_cost_usd", m.get("llm_usd"))
        m["llm_paid_calls"] = deps.llm.report().get("paid_calls", m.get("llm_paid_calls"))
    m["name"] = sc["name"]
    m["split"] = sc.get("split", "dev")
    m["playbook"] = spec.pack
    m["target"] = spec.target_count
    return m


SUMMARY_KEYS = ["good@K", "good_reachable@K", "good_verified@K", "precision@K", "mean_score", "distinct_orgs",
                "proof_rate", "reachable", "search_calls", "llm_paid_calls", "llm_usd", "seconds"]


def _pooled(ok: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pooled precision with 95% intervals: the judge's good@K over every slot, and human-label precision over the
    shortlisted candidates a human labelled. Pooling across scenarios is what makes the interval meaningful."""
    from brain.evalkit import cluster_ci

    out: Dict[str, Any] = {}
    slots = sum(int(r.get("target") or 0) for r in ok)
    good = sum(int(r.get("good@K") or 0) for r in ok)
    if slots:
        out["judge_precision"] = round(good / slots, 3)
        c = cluster_ci([(int(r.get("good@K") or 0), int(r.get("target") or 0)) for r in ok])
        out["judge_precision_ci95"], out["judge_precision_ci_method"] = c["ci"], c["method"]
        out["sendable_precision"] = round(sum(int(r.get("good_reachable@K") or 0) for r in ok) / slots, 3)
        out["verified_precision"] = round(sum(int(r.get("good_verified@K") or 0) for r in ok) / slots, 3)
    n = sum(int(r.get("label_n_k") or 0) for r in ok)
    if n:
        g = sum(int(r.get("label_good_k") or 0) for r in ok)
        out["label_precision"] = round(g / n, 3)
        c = cluster_ci([(int(r.get("label_good_k") or 0), int(r.get("label_n_k") or 0)) for r in ok])
        out["label_precision_ci95"], out["label_precision_ci_method"] = c["ci"], c["method"]
        out["labeled_slots"] = f"{n}/{slots}"
        out["wrong_address_labels"] = sum(int(r.get("label_wrong_address") or 0) for r in ok)
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r and not r.get("invalid")]
    errored = [r for r in rows if "error" in r]
    invalid = [r for r in rows if "error" not in r and r.get("invalid")]
    out: Dict[str, Any] = {"scenarios": len(rows), "failed": len(errored)}
    if invalid:  # discovery came back empty because the search/GitHub account failed: not a result about the pipeline
        out["invalid"] = len(invalid)
        out["invalid_names"] = [r["name"] for r in invalid]
    degraded = [r["name"] for r in ok if r.get("provider_problem")]
    if degraded:  # ran, but a real share of calls failed, so these numbers are lower bounds
        out["degraded_names"] = degraded
    for k in SUMMARY_KEYS:
        vals = [r.get(k) for r in ok if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        out[k] = round(sum(vals) / len(vals), 3) if k in ("precision@K", "mean_score", "proof_rate") else round(sum(vals), 4)
    with_intake = [r for r in ok if "intake_playbook_ok" in r]
    if with_intake:
        out["intake_playbook_ok"] = f"{sum(r['intake_playbook_ok'] for r in with_intake)}/{len(with_intake)}"
        out["intake_where_ok"] = f"{sum(bool(r.get('intake_where_ok')) for r in with_intake)}/{len(with_intake)}"
    drafted = [r["drafts"] for r in ok if r.get("drafts", {}).get("drafts")]
    if drafted:
        n = sum(d["drafts"] for d in drafted)
        out["drafts"] = n
        out["draft_pass_rate"] = round(sum(d["pass_rate"] * d["drafts"] for d in drafted) / n, 3)
        ws = [(d["would_send_rate"], d["drafts"]) for d in drafted if d.get("would_send_rate") is not None]
        if ws:
            out["would_send_rate"] = round(sum(a * b for a, b in ws) / sum(b for _, b in ws), 3)
    goods = out.get("good@K") or 0
    spend = (out.get("llm_usd") or 0) + SERPER_USD * (out.get("search_calls") or 0)
    out["usd_per_good"] = round(spend / goods, 4) if goods else None
    out.update(_pooled(ok))
    lost = [r for r in errored + invalid if r.get("target")]
    if lost and out.get("judge_precision") is not None:  # the number if every scenario that did not count had scored zero
        slots = sum(int(r.get("target") or 0) for r in ok) + sum(int(r["target"]) for r in lost)
        out["judge_precision_worst_case"] = round(sum(int(r.get("good@K") or 0) for r in ok) / slots, 3)
    splits = sorted({r.get("split", "dev") for r in ok})
    if len(splits) > 1 or splits == ["holdout"]:  # trust the holdout line: nothing was tuned on it
        out["by_split"] = {sp: {"scenarios": len(sub), **_pooled(sub)}
                           for sp in splits for sub in [[r for r in ok if r.get("split", "dev") == sp]]}
    noisy = [r["repeat"]["spread"] for r in ok if r.get("repeat")]
    if noisy:
        out["repeat_noise_good@K"] = round(sum(noisy) / len(noisy), 2)  # average max-min across repeats, per scenario
    return out


def _scenario_row(sc: Dict[str, Any], *, replay: str, draft: bool, out_dir: Path, from_prompt: bool, repeat: int,
                  progress=print) -> Dict[str, Any]:
    """Run one scenario (repeat times with fresh caches) and return its report row."""
    runs = []
    for i in range(max(1, repeat)):
        progress(f"[bench] {sc['name']} ({sc.get('playbook')}, {sc.get('split', 'dev')})"
                 + (f" run {i + 1}/{repeat}" if repeat > 1 else "") + " …")
        sub = out_dir / f"rep{i + 1}" if repeat > 1 else out_dir
        sub.mkdir(parents=True, exist_ok=True)
        runs.append(run_scenario(sc, replay=replay, draft=draft, out_dir=sub, from_prompt=from_prompt,
                                 cache_tag=f".rep{i + 1}" if repeat > 1 else ""))
    return _combine_runs(runs)


def _mean(vals: List[float]) -> float:
    m = sum(vals) / len(vals)
    return int(m) if m == int(m) else round(m, 2)


def _combine_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One row from several runs of the same scenario: every numeric metric is the MEAN over the runs that counted, so
    the headline uses all the data instead of the first run. Runs that errored or were invalid (a failing provider)
    are left out when any run was fine. The per-run values and the spread are kept; per-candidate detail (shortlist,
    emails, warnings) is from the first counted run."""
    if len(runs) == 1:
        return runs[0]
    done = [r for r in runs if "error" not in r]
    if not done:
        return runs[0]
    counted = [r for r in done if not r.get("invalid")] or done
    row = dict(counted[0])
    for k, v in counted[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals = [r[k] for r in counted if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            row[k] = _mean(vals)
    goods = [r["good@K"] for r in counted]
    row["repeat"] = {"runs": len(counted), "of": len(runs), "good@K": goods, "mean": _mean(goods),
                     "good_reachable@K": [r.get("good_reachable@K") for r in counted],
                     "min": min(goods), "max": max(goods), "spread": max(goods) - min(goods),
                     "left_out": [r.get("error") or r.get("invalid") for r in runs if not any(r is c for c in counted)]}
    return row


def _bench_worker(args: tuple) -> Dict[str, Any]:  # top level so a process pool can import it
    sc, kw = args
    return _scenario_row(sc, **kw, progress=lambda m: None)


def _row_line(row: Dict[str, Any], name: str, repeat: int) -> str:
    return f"[bench] {name}: " + (row.get("error") or
            f"good@K {row['good@K']}/{row['target']}, reachable {row['reachable']}, "
            f"{row['search_calls']} searches, {row['llm_paid_calls']} LLM calls (${row['llm_usd']})"
            + (f", over {repeat} runs {row['repeat']['good@K']}" if row.get("repeat") else ""))


def run_bench(only: Optional[List[str]] = None, *, replay: str = "", draft: bool = False, label: str = "",
              progress=print, from_prompt: bool = True, split: str = "", repeat: int = 1, parallel: int = 1) -> Dict[str, Any]:
    """split: only the "dev" or "holdout" scenarios. repeat: run every scenario this many times, each with its own
    fresh cache (a repeat that reads the first run's cache measures nothing), and report the spread.
    parallel: scenarios run side by side in separate processes (they share the search and LLM accounts, so the
    provider's rate limits bind sooner). The report is rewritten after every scenario, so an interrupted run keeps
    what finished (<label>.partial.json)."""
    if repeat > 1 and replay:
        raise ValueError("--repeat measures live noise; it cannot be combined with --replay")
    scenarios = load_scenarios(only, split=split)
    if not scenarios:
        raise ValueError(f"No scenarios found in {SCENARIO_DIR} (filter: {only}, split: {split or 'any'})")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = RESULT_DIR / f"run_{label or stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    partial = RESULT_DIR / f"{label or stamp}.partial.json"

    def checkpoint(done: bool = False) -> Dict[str, Any]:
        rep = {"label": label or stamp, "replay": replay, "split": split or "all", "repeat": repeat, "rows": rows,
               "summary": summarize(rows) if rows else {}, "complete": done}
        (RESULT_DIR / f"{label or stamp}.json" if done else partial).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        return rep

    kw = dict(replay=replay, draft=draft, out_dir=out_dir, from_prompt=from_prompt, repeat=repeat)
    if parallel > 1 and len(scenarios) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=min(parallel, len(scenarios))) as ex:
            futures = {ex.submit(_bench_worker, (sc, kw)): sc for sc in scenarios}
            for sc in scenarios:
                progress(f"[bench] {sc['name']} queued")
            for fut in as_completed(futures):
                sc = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:  # a crashed worker must not lose the others
                    row = {"name": sc["name"], "error": f"{type(e).__name__}: {e}", "split": sc.get("split", "dev"),
                           "target": int(sc.get("target_count", 8))}
                rows.append(row)
                progress(_row_line(row, sc["name"], repeat))
                checkpoint()
        order = {sc["name"]: i for i, sc in enumerate(scenarios)}
        rows.sort(key=lambda r: order.get(r.get("name"), 0))
    else:
        for sc in scenarios:
            row = _scenario_row(sc, **kw, progress=progress)
            rows.append(row)
            progress(_row_line(row, sc["name"], repeat))
            checkpoint()
    report = checkpoint(done=True)
    path = RESULT_DIR / f"{label or stamp}.json"
    partial.unlink(missing_ok=True)
    report["path"] = str(path)
    return report


# ── labels: the ground truth ─────────────────────────────────────────────────

def label_status(min_labels: int = 30, folder: Path = LABEL_DIR, scenarios: Optional[List[Dict[str, Any]]] = None
                 ) -> List[Dict[str, Any]]:
    """How far each scenario's human labels are from usable: labelled vs wanted, per verdict. A person written in both
    the sheet and the top-level lists counts once; one given two different verdicts is reported as a conflict."""
    from brain.evalkit import label_conflicts

    rows = []
    for sc in scenarios if scenarios is not None else load_scenarios(label_dir=folder):
        path = folder / f"{sc['name']}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        cands = data.get("candidates") or []
        extra = load_labels(sc["name"], folder)
        from brain.evalkit import _key

        labeled = len({_key(i) for v in extra.values() for i in v})
        rows.append({"name": sc["name"], "split": sc.get("split", "dev"), "candidates": len(cands), "labeled": labeled,
                     "good": len(extra.get("good", [])), "bad": len(extra.get("bad", [])),
                     "wrong_address": len(extra.get("wrong_address", [])), "needed": max(0, min_labels - labeled),
                     "skipped": sum(1 for c in cands if c.get("skipped") and not str(c.get("label") or "").strip()),
                     "conflicts": label_conflicts(extra), "sheet": path.exists()})
    return rows


def label_interactive(name: str, ask=input, say=print, folder: Path = LABEL_DIR, limit: int = 0,
                      revisit_skipped: bool = False) -> int:
    """Walk through a scenario's unlabelled candidates in the terminal. g good, b bad, w wrong address (right person,
    address not theirs), s skip, u undo the last answer, q quit. The sheet is saved (atomically) after every answer,
    a skipped candidate is not asked again next time unless revisit_skipped, and Ctrl-C / Ctrl-D end the session
    cleanly. Returns the number labelled."""
    path = folder / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No label sheet {path}; run `brain bench-labels --run <label>` first")
    data = json.loads(path.read_text(encoding="utf-8"))
    cands = data.get("candidates") or []
    n, i, history = 0, 0, []  # history: (index, labelled?, was it skipped before)
    while i < len(cands):
        c = cands[i]
        if str(c.get("label") or "").strip() or (c.get("skipped") and not revisit_skipped):
            i += 1
            continue
        say(f"\n{c.get('name')}  ·  {c.get('org') or '-'}  ·  {c.get('title') or '-'}\n  email: {c.get('email') or '(none)'}"
            f"\n  source: {', '.join(c.get('sources') or [])}  judge: {c.get('judge_score')}")
        for ev in (c.get("evidence") or [])[:3]:
            say(f"  - {ev}")
        try:
            a = str(ask("[g]ood  [b]ad  [w]rong address  [s]kip  [u]ndo  [q]uit > ")).strip().lower()[:1]
        except (EOFError, KeyboardInterrupt):
            break
        if a == "q":
            break
        if a == "u":
            if not history:
                say("Nothing to undo.")
                continue
            j, was_label, was_skipped = history.pop()
            cands[j]["label"] = ""
            cands[j].pop("skipped", None)
            if was_skipped:
                cands[j]["skipped"] = True
            n -= was_label
            i = j
            _write_json_atomic(path, data)
            continue
        if a == "s":
            history.append((i, 0, bool(c.get("skipped"))))
            c["skipped"] = True
        elif a in ("g", "b", "w"):
            history.append((i, 1, bool(c.get("skipped"))))
            c["label"] = {"g": "good", "b": "bad", "w": "wrong_address"}[a]
            c.pop("skipped", None)
            n += 1
        else:
            continue  # not an option: ask again about the same candidate
        _write_json_atomic(path, data)
        i += 1
        if limit and n >= limit:
            break
    return n


def compare(new: Dict[str, Any], old: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-scenario deltas on the headline metrics (new - old)."""
    olds = {r["name"]: r for r in old.get("rows", [])}
    out = []
    for r in new.get("rows", []):
        o = olds.get(r["name"])
        if not o or "error" in r or "error" in o:
            out.append({"name": r["name"], "note": "missing or failed on one side"})
            continue
        if r.get("invalid") or o.get("invalid"):
            out.append({"name": r["name"], "note": "invalid on one side (the provider failed): not comparable"})
            continue
        row = {"name": r["name"]}
        for k in SUMMARY_KEYS + ["usd_per_good"]:
            a, b = r.get(k), o.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                row[k] = (b, a)
        spreads = [x["repeat"]["spread"] for x in (r, o) if x.get("repeat")]
        if spreads and "good@K" in row:  # a change no bigger than the run-to-run spread is not evidence of anything
            row["noise"] = max(spreads)
            row["within_noise"] = abs(row["good@K"][1] - row["good@K"][0]) <= max(spreads)
        out.append(row)
    return out


def screen_check(run_label: str, only: Optional[List[str]] = None, progress=print) -> Dict[str, Any]:
    """Shadow-evaluate the pre-judge against the judge on the SAME candidates: those a bench run judged (its
    enrich checkpoint). The judge's calls replay from the scenario's LLM cache, so only the pre-judge costs
    anything (~1 fast call per scenario). Reports how many judge-good candidates the screen would have rejected
    (must stay ~0) and how many judge-bad ones it catches. Free of replay artifacts, unlike an end-to-end replay."""
    import sqlite3

    from brain.cache import Cache
    from brain.context import CampaignContext
    from brain.discovery.precision.prejudge import prejudge
    from brain.enrich import has_usable_email
    from brain.fit import profile_for
    from brain.judge import judge
    from brain.llm import LLM
    from brain.models import Contact
    from brain.playbooks import get_playbook
    from brain.settings import Settings

    st = Settings()
    totals = {"good": 0, "good_rejected": 0, "mid": 0, "mid_rejected": 0, "bad": 0, "bad_rejected": 0}
    rows: List[Dict[str, Any]] = []
    for sc in load_scenarios(only):
        db = RESULT_DIR / f"run_{run_label}" / f"{sc['name']}.db"
        if not db.exists():
            continue
        con = sqlite3.connect(str(db))
        stages = {s: json.loads(d) for s, d in con.execute("select stage, payload_json from brain_stages")}
        con.close()
        if "enrich" not in stages:
            continue
        spec = spec_from_scenario(sc)
        pb = get_playbook(spec.pack)
        llm = LLM(st, Cache(str(CACHE_DIR / f"{sc['name']}.sqlite"), ignore_ttl=True))
        ctx = CampaignContext(understanding=spec.objective, target_brief=spec.persona.description)
        for f in sc.get("facts", []):
            ctx.add_fact(f, "scenario")
        targets = [Contact(**c) for c in stages["enrich"]["contacts"]]
        judged = [c for c in targets if has_usable_email(c, st.min_email_confidence)][: spec.target_count * 2]
        judge(judged, spec, llm, batch_size=st.judge_batch_size, ctx=ctx, playbook=pb)
        screened = [c.model_copy(deep=True) for c in judged]
        for c in screened:
            c.prejudge, c.prejudge_reason = None, None
            c.flags = [f for f in c.flags if f != "prejudge_reject"]
        prejudge(screened, spec, llm, prof=profile_for(spec, pb), min_score=st.prejudge_min)
        for c, p in zip(judged, screened):
            if c.score is None:
                continue
            band = "good" if c.score >= GOOD else "bad" if c.score < 0.5 else "mid"
            rejected = "prejudge_reject" in p.flags
            totals[band] += 1
            totals[band + "_rejected"] += int(rejected)
            rows.append({"scenario": sc["name"], "name": c.name, "judge": c.score, "band": band,
                         "prejudge": p.prejudge, "rejected": rejected, "why": p.prejudge_reason})
        progress(f"[screen] {sc['name']}: {sum(1 for r in rows if r['scenario'] == sc['name'])} candidates checked")
    return {"totals": totals, "rows": rows}

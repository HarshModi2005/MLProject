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


def load_labels(name: str, folder: Path = LABEL_DIR) -> Dict[str, List[str]]:
    """Human labels for a scenario from evals/labels/<name>.json: {"good": [...], "bad": [...]} and/or a label
    sheet's "candidates" with "label": "good" | "bad" (see export_label_sheet)."""
    path = folder / f"{name}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    good, bad = list(data.get("good") or []), list(data.get("bad") or [])
    for c in data.get("candidates") or []:
        verdict = str(c.get("label") or "").strip().lower()
        ident = c.get("email") or c.get("name")
        if ident and verdict in ("good", "bad"):
            (good if verdict == "good" else bad).append(ident)
    return {"good": good, "bad": bad}


def load_scenarios(only: Optional[List[str]] = None, folder: Path = SCENARIO_DIR,
                   label_dir: Path = LABEL_DIR) -> List[Dict[str, Any]]:
    out = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("name", path.stem)
        if only and data["name"] not in only:
            continue
        extra = load_labels(data["name"], label_dir)
        if extra.get("good") or extra.get("bad"):
            labels = data.setdefault("labels", {})
            labels["good"] = list(dict.fromkeys((labels.get("good") or []) + extra["good"]))
            labels["bad"] = list(dict.fromkeys((labels.get("bad") or []) + extra["bad"]))
        out.append(data)
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
    path = folder / f"{name}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    known = {(c.get("email") or c.get("name")): c for c in old.get("candidates") or []}
    rows = list(old.get("candidates") or [])
    for c in pool[:limit]:
        best = max(c.get("emails") or [{}], key=lambda e: e.get("confidence", 0))
        ident = best.get("address") or c["name"]
        if ident in known:
            continue
        j = judged.get(c["id"], {})
        rows.append({"label": "", "name": c["name"], "kind": c["kind"], "org": c.get("org"), "email": best.get("address"),
                     "sources": c.get("sources"), "title": c.get("title"),
                     "evidence": [e["text"][:200] for e in (c.get("evidence") or [])[:3]],
                     "url": (c.get("urls") or [None])[0], "judge_score": j.get("score"),
                     "judge_reason": (j.get("reasons") or [None])[0]})
    folder.mkdir(parents=True, exist_ok=True)
    out = {"scenario": name, "how": "Set each candidate's \"label\" to \"good\" or \"bad\" (leave \"\" if unsure). "
                                   "good = you would really email them for this scenario's ask.",
           "good": old.get("good", []), "bad": old.get("bad", []), "candidates": rows}
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
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
    usd = float(llm.get("est_cost_usd") or 0) + SERPER_USD * float(cost.get("search_calls") or 0)
    m = {
        "unique": f.get("unique_people", 0), "qualified": f.get("qualified", 0), "reachable": f.get("reachable", 0),
        "passed_judge": f.get("passed_judge", 0), "shortlisted": len(short),
        "good@K": len(good), "precision@K": round(len(good) / k, 3) if k else None,
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
    if labels and (labels.get("good") or labels.get("bad")):
        import tempfile

        from brain.evalkit import evaluate

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(labels, fh)
        rows = [{"name": c.name, "email": c.best_email.address if c.best_email else "", "score": c.score} for c in short]
        ev = evaluate(rows, fh.name, ks=(k,))
        os.unlink(fh.name)
        m["label_precision@K"] = ev.get(f"precision@{k}")
        m["labeled"] = ev.get("labeled")
    return m


def run_scenario(sc: Dict[str, Any], *, replay: str = "", draft: bool = False, out_dir: Path) -> Dict[str, Any]:
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
    cache = Cache(str(CACHE_DIR / f"{sc['name']}.sqlite"), ignore_ttl=bool(replay))
    deps = Deps(settings=st, cache=cache, llm=LLM(st, cache), fetcher=Fetcher(st, cache), search=WebSearch(st, cache),
                orcid=Orcid(st, cache), openalex=OpenAlexSource(st, cache), store=Store(st.database_url))
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
        return {"name": sc["name"], "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - started, 1)}
    m = metrics(result, spec, sc.get("labels"))
    m["name"] = sc["name"]
    m["playbook"] = spec.pack
    m["target"] = spec.target_count
    return m


SUMMARY_KEYS = ["good@K", "precision@K", "mean_score", "distinct_orgs", "proof_rate", "reachable", "search_calls",
                "llm_paid_calls", "llm_usd", "seconds"]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    out: Dict[str, Any] = {"scenarios": len(rows), "failed": len(rows) - len(ok)}
    for k in SUMMARY_KEYS:
        vals = [r.get(k) for r in ok if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        out[k] = round(sum(vals) / len(vals), 3) if k in ("precision@K", "mean_score", "proof_rate") else round(sum(vals), 4)
    goods = out.get("good@K") or 0
    spend = (out.get("llm_usd") or 0) + SERPER_USD * (out.get("search_calls") or 0)
    out["usd_per_good"] = round(spend / goods, 4) if goods else None
    return out


def run_bench(only: Optional[List[str]] = None, *, replay: str = "", draft: bool = False, label: str = "",
              progress=print) -> Dict[str, Any]:
    scenarios = load_scenarios(only)
    if not scenarios:
        raise ValueError(f"No scenarios found in {SCENARIO_DIR} (filter: {only})")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_dir = RESULT_DIR / f"run_{label or stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in scenarios:
        progress(f"[bench] {sc['name']} ({sc.get('playbook')}) …")
        row = run_scenario(sc, replay=replay, draft=draft, out_dir=out_dir)
        rows.append(row)
        progress(f"[bench] {sc['name']}: " + (row.get("error") or
                 f"good@K {row['good@K']}/{row['target']}, reachable {row['reachable']}, "
                 f"{row['search_calls']} searches, {row['llm_paid_calls']} LLM calls (${row['llm_usd']})"))
    report = {"label": label or stamp, "replay": replay, "rows": rows, "summary": summarize(rows)}
    path = RESULT_DIR / f"{label or stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["path"] = str(path)
    return report


def compare(new: Dict[str, Any], old: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-scenario deltas on the headline metrics (new - old)."""
    olds = {r["name"]: r for r in old.get("rows", [])}
    out = []
    for r in new.get("rows", []):
        o = olds.get(r["name"])
        if not o or "error" in r or "error" in o:
            out.append({"name": r["name"], "note": "missing or failed on one side"})
            continue
        row = {"name": r["name"]}
        for k in SUMMARY_KEYS + ["usd_per_good"]:
            a, b = r.get(k), o.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                row[k] = (b, a)
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

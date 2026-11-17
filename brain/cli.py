"""Command line for the brain:  python main.py brain --help"""

from __future__ import annotations

import csv
import json
import logging
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from brain.llm import LLMError
from brain.models import SenderProfile

app = typer.Typer(help="Discovery, ranking and drafting engine for any cold-outreach campaign.", no_args_is_help=True)
console = Console()


def _deps():
    from brain.pipeline import build_deps

    return build_deps()


def _print_funnel(funnel: dict) -> None:
    t = Table(title="Funnel", show_header=False)
    for label, key in [("Records found", "found_raw"), ("Unique candidates", "unique_people"), ("Qualified", "qualified"),
                       ("Enriched (top of list)", "enriched"), ("Confident email", "reachable"),
                       ("Judged by LLM", "judged"), ("Passed judge", "passed_judge"), ("Shortlisted", "shortlisted"),
                       ("Drafts ready", "drafted_ok"), ("Drafts needing review", "drafts_need_review")]:
        if key in funnel:
            t.add_row(label, str(funnel[key]))
    console.print(t)
    if funnel.get("dropped"):
        console.print(f"[dim]Filtered out: {funnel['dropped']}[/dim]")
    cost = funnel.get("cost") or {}
    if cost:
        llm = cost.get("llm", {})
        console.print(
            f"[dim]Cost — LLM calls: {llm.get('paid_calls', 0)} paid / {llm.get('cached_hits', 0)} cached "
            f"{llm.get('by_purpose', {})}; search calls: {cost.get('search_calls', 0)}; "
            f"by kind {cost.get('search_by_kind', {})}; OpenAlex requests: {cost.get('openalex_requests', 0)}; "
            f"pages fetched: {cost.get('pages_fetched', 0)}; "
            f"{cost.get('seconds', 0)}s[/dim]"
        )


def _print_shortlist(contacts, drafts=None) -> None:
    t = Table(title=f"Shortlist ({len(contacts)})")
    for col in ("#", "Name", "Organisation", "Email", "Conf.", "Score", "Why"):
        t.add_column(col, overflow="fold")
    for i, c in enumerate(contacts, 1):
        best = c.best_email
        t.add_row(str(i), c.name, (c.org or "")[:32], best.address if best else "—",
                  f"{best.confidence:.2f}" if best else "", f"{(c.score if c.score is not None else c.prescore):.2f}",
                  (c.reasons[0] if c.reasons else (c.hook or ""))[:70])
    console.print(t)


@app.command()
def run(
    description: str = typer.Argument("", help="What you want, in plain words."),
    cv: Optional[str] = typer.Option(None, help="Path to your CV / résumé / deck (PDF, DOCX, TXT)."),
    about: Optional[str] = typer.Option(None, help="A few sentences about yourself or your offer."),
    name: Optional[str] = typer.Option(None, help="Your name (overrides what the CV says)."),
    n: Optional[int] = typer.Option(None, help="How many recipients (default: what you asked for, else 25)."),
    location: Optional[List[str]] = typer.Option(None, help="Country or 'City, Country'. Repeatable."),
    upload: Optional[List[str]] = typer.Option(None, help="An older list that could help (any format). Repeatable."),
    playbook: Optional[str] = typer.Option(None, "--playbook", "--pack", help="Force a playbook (see `brain playbooks`)."),
    topic: Optional[List[str]] = typer.Option(None, help="Skip intake: topic to target. Repeatable (needs --objective)."),
    objective: Optional[str] = typer.Option(None, help="With --topic: what you want from them."),
    people_csv: Optional[str] = typer.Option(None, "--csv", help="Work only on this list of people (no discovery)."),
    upto: str = typer.Option("draft", help="Stop after: discover | rank | enrich | judge | draft."),
    to_review: bool = typer.Option(False, help="Also copy finished drafts into the review UI."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip questions and verification (use defaults)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a campaign: verify what you want, find recipients, rank them, find emails, draft."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    deps = _deps()
    try:
        _run(deps, description, cv, about, name, n, location, upload or [], playbook, topic, objective, people_csv,
             upto, to_review, yes)
    except LLMError as e:
        console.print(f"[red]Stopped: the language model is unavailable — {e}[/red]")
        console.print("[dim]Any finished stages are saved; continue later with `python main.py brain resume <id>`.[/dim]")
        raise typer.Exit(2)
    except KeyboardInterrupt:
        console.print("[yellow]Cancelled — nothing was searched or sent.[/yellow]")
        raise typer.Exit(1)


def _sender(deps, cv, about, name):
    from brain.spec import sender_from_pdf, sender_from_text

    source = "your profile"
    if cv and deps.llm.available:
        if cv.lower().endswith(".pdf"):
            sender = sender_from_pdf(cv, deps.llm)
        else:
            from brain.uploads import load

            sender = sender_from_text(load(cv).text, deps.llm, attachment_path=cv)
        source = f"CV: {cv.rsplit('/', 1)[-1]}"
    elif about and deps.llm.available:
        sender = sender_from_text(about, deps.llm)
        source = "your description"
    else:
        sender = SenderProfile(summary=about or "", attachment_path=cv)
    if name:
        sender.name = name
    return sender, source


def _run(deps, description, cv, about, name, n, location, uploads, playbook, topic, objective, people_csv, upto,
         to_review, yes):
    from brain.context import CampaignContext
    from brain.intake import ConsoleUI, run_intake
    from brain.pipeline import run_campaign
    from brain.spec import spec_from_fields

    sender, source = _sender(deps, cv, about, name)
    ctx, seeds, only = None, [], False
    if topic:  # no LLM planning at all
        spec = spec_from_fields(topics=list(topic), objective=objective or description or "Get in touch",
                                sender=sender, locations=location, pack=playbook or "academic", target_count=n or 25)
        ctx = CampaignContext()
        for h in sender.highlights:
            ctx.add_fact(h, source)
    else:
        if not description:
            raise typer.BadParameter("Describe what you want, e.g. brain run \"find sponsors for our college fest\".")
        if not deps.llm.available:
            raise typer.BadParameter("No LLM key set. Use --topic/--objective to run without one.")
        res = run_intake(description, deps.llm, ConsoleUI(), sender, upload_paths=uploads, target_count=n,
                         locations=location, playbook=playbook, interactive=not yes,
                         max_searches_ceiling=deps.settings.max_search_calls, sender_source=source)
        spec, ctx, seeds, only = res.spec, res.ctx, res.seed_contacts, res.only_seeds
        console.print(f"[dim]Intake used {res.llm_calls} LLM call(s).[/dim]")

    where = ", ".join(spec.regions + spec.countries) or "anywhere"
    console.print(Panel(
        f"[bold]{spec.objective}[/bold]\nPlaybook: {spec.pack} · Looking for: {', '.join(spec.topics)}\n"
        f"Where: {where} · Target: {spec.target_count} · Campaign id: [cyan]{spec.id}[/cyan]", title="Campaign"))
    result = run_campaign(spec, deps, upto=upto, csv_path=people_csv, ctx=ctx, seed_contacts=seeds, only_seeds=only,
                          progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    _print_shortlist(result.shortlist, result.drafts)
    _print_funnel(result.funnel)
    for w in result.warnings:
        console.print(f"[yellow]! {w}[/yellow]")
    if to_review and result.drafts:
        from brain.bridge import export_to_review

        console.print(f"[green]{export_to_review(deps.store, spec)} drafts copied to the review UI "
                      f"(python main.py review).[/green]")
    console.print(f"\nSee drafts: [cyan]python main.py brain show {spec.id} --drafts[/cyan]")


@app.command()
def playbooks():
    """List the playbooks (kinds of outreach) and discovery tools available."""
    from brain.playbooks import PLAYBOOKS
    from brain.tools import describe_tools

    t = Table("Playbook", "Targets", "Tools", "When to use")
    for pb in PLAYBOOKS.values():
        t.add_row(pb.name, pb.target_kind, ", ".join(pb.sources), pb.description[:80])
    console.print(t)
    console.print("[bold]Tools[/bold]\n" + describe_tools())
    console.print("[dim]Add your own playbook: a JSON file in ./playbooks/ (see brain/playbooks.py for the fields).[/dim]")


@app.command()
def resume(campaign_id: str, upto: str = "draft"):
    """Continue an interrupted campaign from its last finished stage."""
    from brain.pipeline import run_campaign

    deps = _deps()
    saved = deps.store.load_campaign(campaign_id)
    if not saved:
        raise typer.BadParameter(f"No campaign {campaign_id}")
    result = run_campaign(saved["spec"], deps, upto=upto, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    _print_shortlist(result.shortlist, result.drafts)
    _print_funnel(result.funnel)
    for w in result.warnings:
        console.print(f"[yellow]! {w}[/yellow]")


@app.command("list")
def list_campaigns():
    """List campaigns."""
    t = Table("Id", "Status", "Created", "Objective")
    for c in _deps().store.list_campaigns():
        t.add_row(c["id"], c["status"], str(c["created_at"])[:16], c["objective"][:70])
    console.print(t)


@app.command()
def show(campaign_id: str, drafts: bool = typer.Option(False, help="Print the full drafts.")):
    """Show a campaign's shortlist, funnel and (optionally) drafts."""
    store = _deps().store
    saved = store.load_campaign(campaign_id)
    if not saved:
        raise typer.BadParameter(f"No campaign {campaign_id}")
    rows = store.results(campaign_id)
    t = Table("#", "Name", "Organisation", "Email", "Score", "Status", "Why")
    for i, r in enumerate(rows, 1):
        t.add_row(str(i), r["name"], (r["org"] or "")[:30], r["email"] or "—", f"{r['score'] or 0:.2f}", r["status"],
                  (r["reasons"][0] if r["reasons"] else "")[:60])
    console.print(t)
    _print_funnel(saved["funnel"])
    if drafts:
        for r in rows:
            if r["body"]:
                issues = f"\n\n[red]Needs attention: {'; '.join(r['issues'])}[/red]" if r["issues"] else ""
                console.print(Panel(f"[bold]To:[/bold] {r['email']}\n[bold]Subject:[/bold] {r['subject']}\n\n{r['body']}{issues}",
                                    title=r["name"]))


@app.command()
def export(campaign_id: str, out: Optional[str] = typer.Option(None, help="Write a CSV here."),
           to_review: bool = typer.Option(False, help="Copy drafts into the review UI."),
           include_flagged: bool = typer.Option(False, help="Include drafts that failed checks.")):
    """Export results as CSV and/or hand drafts to the review UI."""
    store = _deps().store
    saved = store.load_campaign(campaign_id)
    if not saved:
        raise typer.BadParameter(f"No campaign {campaign_id}")
    if out:
        rows = store.results(campaign_id)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "organisation", "email", "score", "status", "reasons", "hook", "subject", "body", "issues"])
            for r in rows:
                w.writerow([r["name"], r["org"], r["email"], r["score"], r["status"], " | ".join(r["reasons"]),
                            r["hook"], r["subject"], r["body"], " | ".join(r["issues"])])
        console.print(f"Wrote {len(rows)} rows to {out}")
    if to_review:
        from brain.bridge import export_to_review

        console.print(f"{export_to_review(store, saved['spec'], include_flagged)} drafts copied to the review UI.")


@app.command()
def suppress(value: str, reason: str = "manual"):
    """Never contact this email address or domain again, in any campaign."""
    _deps().store.suppress(value, reason)
    console.print(f"Added {value} to the do-not-contact list.")


@app.command("eval")
def evaluate_cmd(campaign_id: str, labels: str):
    """Score a campaign's ranking against a labels file ({"good": [...], "bad": [...]})."""
    from brain.evalkit import evaluate

    console.print_json(json.dumps(evaluate(_deps().store.results(campaign_id), labels)))


@app.command()
def bench(
    only: Optional[List[str]] = typer.Option(None, help="Scenario name (file stem in evals/scenarios). Repeatable."),
    replay: bool = typer.Option(False, help="Answer every search/fetch from the recorded cache only (free)."),
    strict: bool = typer.Option(False, help="With --replay: LLM calls must also come from the cache."),
    draft: bool = typer.Option(False, help="Also draft emails (costs one LLM call per email)."),
    label: str = typer.Option("", help="Name for this run; results go to evals/results/<label>.json."),
    compare_to: Optional[str] = typer.Option(None, "--compare", help="A previous results JSON to diff against."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Benchmark discovery on the scenarios in evals/scenarios (stops after judging unless --draft)."""
    from brain.bench import compare, run_bench

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report = run_bench(list(only) if only else None, replay=("strict" if strict else "replay") if replay else "",
                       draft=draft, label=label, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    t = Table(title=f"Benchmark {report['label']}")
    for col in ("Scenario", "label prec", "good@K", "prec@K", "mean", "orgs", "proof", "reach", "search", "LLM", "$", "s"):
        t.add_column(col)
    for r in report["rows"]:
        if "error" in r:
            t.add_row(r["name"], f"[red]{r['error'][:60]}[/red]", *[""] * 10)
            continue
        lp = r.get("label_precision@K")
        t.add_row(r["name"], "-" if lp is None else f"{lp} ({r.get('labeled', 0)} lab.)", f"{r['good@K']}/{r['target']}",
                  str(r["precision@K"]), str(r["mean_score"]), str(r["distinct_orgs"]), str(r["proof_rate"]),
                  str(r["reachable"]), str(r["search_calls"]), str(r["llm_paid_calls"]), str(r["llm_usd"]), str(r["seconds"]))
    console.print(t)
    stages = ("unique", "qualified", "reachable", "passed_judge", "shortlisted")
    srcs = Table(title="Per source: unique → qualified → reachable → passed judge → shortlisted  (search calls)")
    for col in ("Scenario", "Source", *stages, "searches"):
        srcs.add_column(col)
    for r in report["rows"]:
        sf, led = r.get("source_funnel") or {}, r.get("ledger") or {}
        for src in sorted({k for st in stages for k in (sf.get(st) or {})}):
            srcs.add_row(r["name"], src, *[str((sf.get(st) or {}).get(src, 0)) for st in stages],
                         str(int((led.get(src) or {}).get("search_calls", 0))))
    if srcs.row_count:
        console.print(srcs)
    console.print_json(json.dumps(report["summary"]))
    console.print(f"[dim]Saved {report['path']}[/dim]")
    if compare_to:
        with open(compare_to, encoding="utf-8") as fh:
            old = json.load(fh)
        d = Table(title=f"Δ vs {old.get('label')}  (old → new)")
        keys = ["good@K", "precision@K", "proof_rate", "search_calls", "llm_usd", "usd_per_good"]
        d.add_column("Scenario")
        for k in keys:
            d.add_column(k)
        for row in compare(report, old):
            d.add_row(row["name"], *[f"{row[k][0]} → {row[k][1]}" if k in row else row.get("note", "") for k in keys])
        console.print(d)


@app.command("network-import")
def network_import(path: str):
    """Import YOUR OWN contacts (LinkedIn Connections.csv, Google Contacts CSV, .vcf, or CSV name,email,company) so
    campaigns can find people you know or can be introduced to. Stored locally only."""
    from brain.discovery.warm.importers import read_file

    people = read_file(path)
    n = _deps().store.network_add(people)
    kinds = sorted({p["source"] for p in people})
    console.print(f"Imported {n} contacts ({', '.join(kinds) or 'none'}). Remove them any time with `brain network-clear`.")


@app.command("network-stats")
def network_stats():
    """How many of your own contacts are imported."""
    console.print(f"{_deps().store.network_size()} contacts in your network")


@app.command("network-clear")
def network_clear():
    """Delete every imported contact of yours."""
    console.print(f"Deleted {_deps().store.network_clear()} contacts")


@app.command()
def refresh(limit: int = typer.Option(500, help="How many remembered prospects to re-check.")):
    """Re-check remembered prospects against free sources (job boards, podcast feeds); run on a schedule."""
    from brain.discovery.index import refresh as do_refresh

    deps = _deps()
    console.print_json(json.dumps(do_refresh(deps.store, deps.fetcher, limit=limit)))


@app.command("bench-screen")
def bench_screen(
    run: str = typer.Option(..., help="A bench run label whose candidates were judged (evals/results/run_<label>)."),
    only: Optional[List[str]] = typer.Option(None, help="Scenario name. Repeatable."),
):
    """Check the pre-judge against the judge on the same candidates (judge replays from cache; ~1 cheap call each)."""
    from brain.bench import screen_check

    out = screen_check(run, list(only) if only else None, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    t = Table(title="Pre-judge vs judge (same candidates)")
    for col in ("Scenario", "Candidate", "judge", "pre-judge", "verdict", "why"):
        t.add_column(col)
    for r in out["rows"]:
        bad = r["rejected"] and r["band"] == "good"
        t.add_row(r["scenario"], r["name"][:30], f"{r['judge']:.2f} {r['band']}", str(r["prejudge"]),
                  ("[red]REJECTED[/red]" if bad else "rejected" if r["rejected"] else "kept"), (r["why"] or "")[:60])
    console.print(t)
    tot = out["totals"]
    console.print(f"good rejected: {tot['good_rejected']}/{tot['good']}  ·  mid rejected: {tot['mid_rejected']}/{tot['mid']}"
                  f"  ·  bad caught: {tot['bad_rejected']}/{tot['bad']}")


@app.command("bench-labels")
def bench_labels(
    run: str = typer.Option(..., help="A bench run label (evals/results/run_<label>)."),
    only: Optional[List[str]] = typer.Option(None, help="Scenario name. Repeatable. Default: every scenario."),
    limit: int = typer.Option(40, help="Candidates per scenario."),
):
    """Write label sheets (evals/labels/<scenario>.json) from a bench run, for a human to mark good / bad."""
    from brain.bench import export_label_sheet, load_scenarios

    for sc in load_scenarios(list(only) if only else None):
        try:
            path = export_label_sheet(run, sc["name"], limit=limit)
            console.print(f"Wrote {path}")
        except FileNotFoundError as e:
            console.print(f"[yellow]{e}[/yellow]")


@app.command()
def labels(campaign_id: str, out: str = typer.Option(..., help="Write the labels JSON here.")):
    """Turn your review decisions (approved / sent / replied vs rejected) into an evaluation labels file."""
    data = _deps().store.labels_from_review(campaign_id)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    console.print(f"Wrote {len(data['good'])} good and {len(data['bad'])} bad labels to {out}")

"""Command line for the brain:  python main.py brain --help"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from brain.llm import LLMError
from brain.models import SenderProfile

app = typer.Typer(help="Discovery, ranking and drafting engine for any cold-outreach campaign.", no_args_is_help=True)
console = Console()


@app.callback()
def _main(workspace: str = typer.Option("", "--workspace", "-w", help="Whose data to use. Each workspace has its own "
                                       "database and cache; nothing is shared between workspaces (env BRAIN_WORKSPACE).")):
    if workspace:
        from brain.workspace import clean

        os.environ["BRAIN_WORKSPACE"] = clean(workspace)


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
    if funnel.get("dedup"):
        console.print(f"[dim]Already covered, skipped before any spend where possible: {funnel['dedup']}[/dim]")
    sup = funnel.get("supply")
    if sup:
        console.print(f"[dim]Supply — asked {sup['requested']}: {sup['fresh']} new, {sup['recycled']} offered again; "
                      f"fresh yield {sup['fresh_yield']}[/dim]")
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
    no_repeat: str = typer.Option("series", "--no-repeat", help="Who counts as already covered: series (default: nobody "
                                  "shown or contacted in this outreach, nor anyone ever emailed) | workspace (anyone in any "
                                  "campaign) | cooldown (earlier touches expire after --cooldown-days)."),
    cooldown_days: int = typer.Option(90, help="With --no-repeat cooldown: days before someone may be offered again."),
    recycle: Optional[str] = typer.Option(None, help="When new matches are very few: ask | auto | never offer earlier, "
                                          "undecided people again (default ask; with --yes and no answer, never)."),
    exclude_file: Optional[str] = typer.Option(None, "--exclude-file", help="A file of addresses, domains or full names "
                                               "(one per line) that must never be contacted."),
):
    """Run a campaign: verify what you want, find recipients, rank them, find emails, draft.
    Run `brain more <id>` afterwards for the next batch of NEW people."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    deps = _deps()
    try:
        _run(deps, description, cv, about, name, n, location, upload or [], playbook, topic, objective, people_csv,
             upto, to_review, yes, no_repeat=no_repeat, cooldown_days=cooldown_days, recycle=recycle,
             exclude_file=exclude_file)
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


def _confirm_recycle(info: dict) -> bool:
    return typer.confirm(f"Only {info['fresh']} new of {info['wanted']} found. {info['undecided']} earlier suggestions are "
                         f"still undecided and {info['standby']} are on standby. Include {info['offer']} of them?",
                         default=False)


def _run(deps, description, cv, about, name, n, location, uploads, playbook, topic, objective, people_csv, upto,
         to_review, yes, no_repeat="series", cooldown_days=90, recycle=None, exclude_file=None):
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
                         max_searches_ceiling=deps.settings.max_search_calls, sender_source=source, store=deps.store)
        spec, ctx, seeds, only = res.spec, res.ctx, res.seed_contacts, res.only_seeds
        console.print(f"[dim]Intake used {res.llm_calls} LLM call(s).[/dim]")

    from brain.touch import MODES, RECYCLE

    if no_repeat not in MODES or (recycle and recycle not in RECYCLE):
        raise typer.BadParameter(f"--no-repeat must be one of {MODES}; --recycle one of {RECYCLE}")
    spec.no_repeat, spec.cooldown_days = no_repeat, cooldown_days
    spec.recycle = recycle or "ask"  # --yes skips questions; it is not consent to re-offer earlier people
    if exclude_file:
        n_ex = deps.store.import_external(_read_lines(exclude_file))
        console.print(f"[dim]{n_ex} do-not-contact entries imported from {exclude_file}.[/dim]")
    where = ", ".join(spec.regions + spec.countries) or "anywhere"
    console.print(Panel(
        f"[bold]{spec.objective}[/bold]\nPlaybook: {spec.pack} · Looking for: {', '.join(spec.topics)}\n"
        f"Where: {where} · Target: {spec.target_count} · Campaign id: [cyan]{spec.id}[/cyan]", title="Campaign"))
    result = run_campaign(spec, deps, upto=upto, csv_path=people_csv, ctx=ctx, seed_contacts=seeds, only_seeds=only,
                          progress=lambda m: console.print(f"[dim]{m}[/dim]"), confirm_recycle=_confirm_recycle)
    _print_shortlist(result.shortlist, result.drafts)
    _print_funnel(result.funnel)
    for w in result.warnings:
        console.print(f"[yellow]! {w}[/yellow]")
    if to_review and result.drafts:
        from brain.bridge import export_to_review

        console.print(f"[green]{export_to_review(deps.store, spec)} drafts copied to the review UI "
                      f"(python main.py review).[/green]")
    console.print(f"\nSee drafts: [cyan]python main.py brain show {spec.id} --drafts[/cyan]")
    console.print(f"Want more people? [cyan]python main.py brain more {spec.id} -n 20[/cyan] (never repeats anyone).")


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


def _read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


@app.command()
def more(
    ref: str = typer.Argument(..., help="The series or any campaign id in it."),
    n: int = typer.Option(20, "-n", help="How many new people to find in this batch."),
    until: Optional[int] = typer.Option(None, "--until", help="Keep running batches until this many people in total."),
    batch_size: int = typer.Option(20, help="With --until: people per batch."),
    budget_searches: Optional[int] = typer.Option(None, help="With --until: stop after this many search calls."),
    widen: Optional[str] = typer.Option(None, help="How far to widen when new people run out: topics | geo. Default: "
                                        "lookalikes only; your topics and places are never widened without this."),
    recycle: Optional[str] = typer.Option(None, help="ask | auto | never: offer earlier undecided / standby people "
                                          "again when new matches are very few (default: the series' setting)."),
    to_review: bool = typer.Option(False, help="Also copy the drafts into the review UI."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask (recycling follows --recycle, else never)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """The next batch of an outreach: NEW people only, never anyone shown, rejected or emailed before."""
    from brain.series import SeriesPaused, more as run_more, until as run_until

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    deps = _deps()
    kw = dict(widen=(widen or "") if widen is not None else None, recycle=recycle,
              confirm_recycle=None if yes else _confirm_recycle,
              progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    if widen and widen not in ("topics", "geo"):
        raise typer.BadParameter("--widen must be topics or geo")
    try:
        if until:
            outcomes = run_until(deps, ref, until, batch_size=batch_size, budget_searches=budget_searches, **kw)
        else:
            outcomes = [run_more(deps, ref, n, **kw)]
    except (KeyError, SeriesPaused) as e:
        console.print(f"[red]{e.args[0] if e.args else e}[/red]")
        raise typer.Exit(1)
    except LLMError as e:
        console.print(f"[red]Stopped: the language model is unavailable — {e}[/red]")
        raise typer.Exit(2)
    for i, b in enumerate(outcomes, 1):
        for res in b.results:
            _print_shortlist(res.shortlist, res.drafts)
        console.print(Panel(f"Asked {b.requested}: [bold]{b.fresh} new[/bold], {b.recycled} offered again, "
                            f"{b.covered_skipped} candidates skipped as already covered. Widening rung {b.rung_reached}."
                            + (f"\n[yellow]{b.stopped}[/yellow]" if b.stopped else ""), title=f"Batch {i}"))
        for w in b.warnings:
            console.print(f"[yellow]! {w}[/yellow]")
        if to_review:
            from brain.bridge import export_to_review

            for res in b.results:
                export_to_review(deps.store, res.spec)
    console.print(f"Series [cyan]{outcomes[0].series_id}[/cyan]: `brain seen {outcomes[0].series_id}` shows everyone covered so far.")


@app.command()
def seen(ref: str = typer.Argument(..., help="The series or any campaign id in it."),
         state: Optional[str] = typer.Option(None, help="Only this state (surfaced, drafted, approved, sent, rejected, standby...)."),
         limit: int = typer.Option(40, help="Rows to print.")):
    """Everyone this outreach has covered: counts by state, then who."""
    from brain.series import resolve_series
    from brain.touch import summarize

    store = _deps().store
    try:
        sid = resolve_series(store, ref)
    except KeyError as e:
        raise typer.BadParameter(str(e.args[0]))
    rows = store.seen(sid)
    console.print(f"Series {sid}: " + ", ".join(f"{v} {k}" for k, v in sorted(summarize(rows).items())) if rows else "Nobody yet.")
    t = Table("Name", "Organisation", "State", "Campaign")
    for r in [r for r in rows if not state or r["state"] == state][:limit]:
        t.add_row(r["name"], r["org"], r["state"], r["campaign_id"])
    console.print(t)


def _decide(who: str, status: str, campaign: Optional[str]) -> None:
    from brain.bridge import legacy_set_status

    store = _deps().store
    res = store.decide(who, status, campaign)
    if not res:
        raise typer.BadParameter(f"No unsent result for {who!r}" + (f" in {campaign}" if campaign else "") + ".")
    legacy_set_status(res["campaign_id"], res["contact_id"], status)
    console.print(f"{who}: {res['was']} -> {status}")


@app.command()
def standby(who: str = typer.Argument(..., help="A contact id or email address."),
            campaign: Optional[str] = typer.Option(None, help="Which campaign, if the person is in several.")):
    """Park someone: not now. New batches skip them; they come back only if new matches are very few, or with `recall`."""
    _decide(who, "standby", campaign)


@app.command()
def reject(who: str = typer.Argument(..., help="A contact id or email address."),
           campaign: Optional[str] = typer.Option(None)):
    """Never offer this person again in this outreach."""
    _decide(who, "rejected", campaign)


def _revive(who: str, campaign: Optional[str]) -> None:
    from brain.bridge import legacy_set_status

    res = _deps().store.revive(who, campaign)
    if not res:
        raise typer.BadParameter(f"{who!r} is not on standby or rejected" + (f" in {campaign}" if campaign else "") + ".")
    legacy_set_status(res["campaign_id"], res["contact_id"], "draft")
    console.print(f"{who}: {res['was']} -> back to undecided (draft shown in the review again).")


@app.command()
def recall(who: str = typer.Argument(..., help="A contact id or email address."), campaign: Optional[str] = typer.Option(None)):
    """Take someone off standby."""
    _revive(who, campaign)


@app.command()
def release(who: str = typer.Argument(..., help="A contact id or email address."), campaign: Optional[str] = typer.Option(None)):
    """Undo a rejection (the only way a rejected person can come back)."""
    _revive(who, campaign)


@app.command("series")
def series_cmd():
    """List outreaches (series) and whether any is paused."""
    t = Table("Id", "Status", "Batches", "No-repeat", "Rung", "Objective")
    for r in _deps().store.list_series():
        t.add_row(r["id"], r["status"], str(r["batches"]), r["mode"], str(r["rung"]), r["objective"][:60])
    console.print(t)


@app.command("series-resume")
def series_resume(ref: str):
    """Lift a pause (set when too many emails bounced) after checking where the addresses came from."""
    from brain.series import resolve_series

    store = _deps().store
    sid = resolve_series(store, ref)
    store.resume_series(sid)
    console.print(f"Series {sid} is active again; the bounce rate is counted afresh from now. "
                  f"Bounces before now: {store.series_health(sid)}")


@app.command()
def exclude(path: str = typer.Argument(..., help="A file of addresses, domains or full names, one per line.")):
    """Never contact these people / domains, in any campaign (imports a do-not-contact or already-emailed list)."""
    n = _deps().store.import_external(_read_lines(path))
    console.print(f"{n} entries added to the do-not-contact list.")


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
    from_spec: bool = typer.Option(False, help="Use the scenarios' ready-made specs instead of running the intake on their prompts."),
    split: str = typer.Option("", help="Only the 'dev' (tuned on) or 'holdout' (never tuned on) scenarios."),
    repeat: int = typer.Option(1, help="Run each scenario N times with fresh caches to measure noise (costs N times as much)."),
    parallel: int = typer.Option(1, help="Run N scenarios side by side (separate processes; they share the API accounts)."),
):
    """Benchmark discovery on the scenarios in evals/scenarios (stops after judging unless --draft)."""
    from brain.bench import compare, run_bench

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report = run_bench(list(only) if only else None, replay=("strict" if strict else "replay") if replay else "",
                       draft=draft, label=label, progress=lambda m: console.print(f"[dim]{m}[/dim]"),
                       from_prompt=not from_spec, split=split, repeat=repeat, parallel=parallel)
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
    sm = report["summary"]
    if sm.get("judge_precision") is not None:
        line = f"judge precision {sm['judge_precision']} (95% CI {sm['judge_precision_ci95'][0]}–{sm['judge_precision_ci95'][1]})"
        line += f"  ·  good AND reachable {sm.get('sendable_precision')}  ·  good AND verified {sm.get('verified_precision')}"
        if sm.get("label_precision") is not None:
            ci = sm["label_precision_ci95"]
            line += f"\nHUMAN-label precision {sm['label_precision']} (95% CI {ci[0]}–{ci[1]}, labelled {sm['labeled_slots']} slots)"
        else:
            line += "\n[yellow]No human labels: precision above is the pipeline's own judge. Run `brain bench-label`.[/yellow]"
        console.print(line)
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


@app.command()
def profile(set_: Optional[List[str]] = typer.Option(None, "--set", help="key=value; keys: name, signature, "
                                                      "postal_address, tone, language, exclusions (comma-separated)")):
    """Show or edit what is remembered about you between campaigns (pre-fills every new campaign)."""
    store = _deps().store
    if set_:
        data = {}
        for kv in set_:
            k, _, v = kv.partition("=")
            data[k.strip()] = [x.strip() for x in v.split(",") if x.strip()] if k.strip() == "exclusions" else v.strip()
        store.save_user_profile(data)
    shown = {k: v for k, v in store.user_profile().items() if not k.startswith("_")}
    console.print_json(json.dumps(shown))


@app.command()
def followups(campaign_id: str, n: int = typer.Option(1, help="Which follow-up to draft: 1 or 2."),
              only_sent: bool = typer.Option(True, help="Only for emails marked sent (default) - else every draft.")):
    """Draft follow-up number n for this campaign's emails: a new, true angle from unused evidence (1 LLM call each)."""
    from brain.draft import followup_email
    from brain.models import Draft
    from brain.packs import get_pack

    deps = _deps()
    saved = deps.store.load_campaign(campaign_id)
    if not saved:
        raise typer.BadParameter(f"No campaign {campaign_id}")
    spec = saved["spec"]
    ctx = deps.store.load_context(campaign_id)
    made = 0
    for r in deps.store.results(campaign_id):
        if not r.get("body") or (only_sent and r["status"] not in ("sent", "followed_up")):
            continue
        first = Draft(contact_id=r["contact_id"], to=r["email"] or "", subject=r["subject"] or "", body=r["body"])
        d = followup_email(r["contact"], spec, get_pack(spec.pack), deps.llm, first, n=n, ctx=ctx)
        if d is None:
            continue
        deps.store.save_followup(campaign_id, r["contact_id"], n, d)
        made += 1
        console.print(f"\n[bold]{r['name']}[/bold] <{d.to}>  {d.subject}\n{d.body}"
                      + (f"\n[yellow]needs review: {'; '.join(d.issues)}[/yellow]" if d.issues else ""))
    console.print(f"\n[dim]{made} follow-up(s) drafted and saved.{'' if made else ' (Mark sent emails as sent, or pass --no-only-sent.)'}[/dim]")


@app.command("edit-draft")
def edit_draft(campaign_id: str, contact_id: str, path: str = typer.Argument(..., help="A text file with your edited email.")):
    """Save your edited version of a draft (marks it approved); your last 3 edits become style examples for new drafts."""
    from pathlib import Path

    _deps().store.record_edit(campaign_id, contact_id, Path(path).read_text(encoding="utf-8"))
    console.print("Saved; future drafts will follow your edits' style.")


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


@app.command("eval-intake")
def eval_intake(only: Optional[List[str]] = typer.Option(None, help="Case id (evals/intake/cases.json). Repeatable.")):
    """Score intent understanding on the intake cases (one cached LLM call per case)."""
    from brain.intakeeval import run as run_eval

    deps = _deps()
    rep = run_eval(deps.llm, list(only) if only else None, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print_json(json.dumps({k: rep[k] for k in ("cases", "check_accuracy", "cases_fully_right", "by_check")}))
    console.print(f"[dim]LLM: {rep['llm'].get('paid_calls')} calls, ${rep['llm'].get('est_cost_usd')}; saved {rep['path']}[/dim]")


@app.command("bench-drafts")
def bench_drafts(run: str = typer.Option(..., help="A bench run label with judged shortlists."),
                 only: Optional[List[str]] = typer.Option(None, help="Scenario name. Repeatable."),
                 per: int = typer.Option(3, help="Drafts per scenario.")):
    """Draft, fact-check and grade emails for a bench run's shortlists (no discovery spend)."""
    from brain.drafteval import draft_eval

    rep = draft_eval(run, list(only) if only else None, per=per, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print_json(json.dumps({**rep["summary"], "llm_usd": rep["llm_usd"]}, default=str))
    console.print(f"[dim]Saved {rep['path']}[/dim]")


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


@app.command("bench-label-status")
def bench_label_status(min_labels: int = typer.Option(30, help="Labels wanted per scenario.")):
    """How many human labels each scenario has, and how many more are needed before precision can be trusted."""
    from brain.bench import label_status

    t = Table(title=f"Human labels (want {min_labels} per scenario)")
    for col in ("Scenario", "split", "sheet", "candidates", "labelled", "good", "bad", "wrong addr", "still needed"):
        t.add_column(col)
    for r in label_status(min_labels):
        t.add_row(r["name"], r["split"], "yes" if r["sheet"] else "[red]no[/red]", str(r["candidates"]), str(r["labeled"]),
                  str(r["good"]), str(r["bad"]), str(r["wrong_address"]),
                  "[green]0[/green]" if not r["needed"] else f"[yellow]{r['needed']}[/yellow]")
    console.print(t)


@app.command()
def outcome(campaign_id: str, who: str = typer.Argument(..., help="The contact id or the email address."),
            kind: str = typer.Argument(..., help="sent | bounced | replied | positive | negative | unsubscribed | spam"),
            detail: str = typer.Option("", help="Optional note, e.g. the bounce message.")):
    """Record what happened to a sent email. Bounces suppress the address and demote a bad email convention; replies
    label the prospect good."""
    learned = _deps().store.record_outcome(campaign_id, who, kind, detail)
    if learned is None:
        console.print(f"[red]No result for {who} in campaign {campaign_id}.[/red]")
        raise typer.Exit(1)
    console.print_json(json.dumps(learned))


@app.command()
def outcomes(campaign_id: Optional[str] = typer.Argument(None, help="One campaign, or all."),
             sync: bool = typer.Option(False, "--sync", help="First pull sent/replied/bounced from the review UI's tables.")):
    """Bounce and reply rates by how the address was found and how far it was trusted: the real test of the engine."""
    store = _deps().store
    if sync:
        from brain.bridge import sync_outcomes

        console.print(f"[dim]Synced from the review tables: {sync_outcomes(store)}[/dim]")
    sm = store.outcome_summary(campaign_id)
    for title, key in (("By address tier when sent", "by_tier"), ("By how the address was found", "by_method")):
        t = Table(title=title)
        for col in ("", "sent", "bounced", "bounce rate", "replied", "reply rate", "unsubscribed"):
            t.add_column(col)
        for name, b in sorted(sm[key].items()):
            t.add_row(name, str(b["sent"]), str(b["bounced"]), str(b["bounce_rate"]), str(b["replied"]),
                      str(b["reply_rate"]), str(b["unsubscribed"]))
        console.print(t)
    console.print(f"Emails sent: {sm['sent']}")


@app.command()
def calibrate(save: bool = typer.Option(False, "--save", help="Store the model if it beats the raw score out of fold."),
              folder: str = typer.Option("evals/labels", help="Label sheets to learn from.")):
    """Fit the score calibrator on your human labels, check it out of fold, and keep it only if it beats the raw score."""
    from brain import calibrate as cal

    rows = cal.rows_from_sheets(folder)
    out = cal.train(rows)
    console.print_json(json.dumps(out["report"]))
    if out["model"] is None:
        console.print(f"[yellow]Not calibrated: {out['reason']}.[/yellow]")
        return
    console.print("[green]The calibrated score beats the raw score out of fold.[/green]")
    if save:
        cal.save(_deps().store, out["model"])
        console.print("Saved: new campaigns now rank by calibrated probability of a good prospect.")
    else:
        console.print("Re-run with --save to use it.")


@app.command("bench-label")
def bench_label(scenario: str = typer.Argument(..., help="Scenario name."),
                limit: int = typer.Option(0, help="Stop after this many labels (0 = all).")):
    """Label a scenario's candidates in the terminal: g good, b bad, w wrong address, s skip, q quit."""
    from brain.bench import label_interactive

    n = label_interactive(scenario, ask=lambda m: typer.prompt(m, prompt_suffix=""), say=console.print, limit=limit)
    console.print(f"Labelled {n} candidates.")


@app.command()
def labels(campaign_id: str, out: str = typer.Option(..., help="Write the labels JSON here.")):
    """Turn your review decisions (approved / sent / replied vs rejected) into an evaluation labels file."""
    data = _deps().store.labels_from_review(campaign_id)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    console.print(f"Wrote {len(data['good'])} good and {len(data['bad'])} bad labels to {out}")

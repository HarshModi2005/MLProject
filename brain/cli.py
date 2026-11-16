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

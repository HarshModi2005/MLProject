"""
Autonomous Research Outreach Agent - Entry Point
"""

import json
import os
import sys
import uuid

import typer
from rich.console import Console
from rich.panel import Panel

from config import RUN_LOG_DIR, validate_config
from core.email_discovery import enrich_professor_emails_batch
from core.observability import new_run_id, set_run_context, timed_span
from core.professor_dedup import dedupe_professor_profiles
from core.relevance_scorer import RelevanceScorer, print_shortlist_guidance
from db.database import (
    append_session_revision,
    ensure_session_row,
    init_db,
    list_resumable_sessions,
    load_session_dicts,
    save_professors_batch,
    upsert_session_state,
)
from orchestrator.state import ExecutionPlan, UserIntent

STATE_FILE = "session_state.json"

console = Console()
app = typer.Typer(help="Autonomous Agentic Research Outreach & Cold Email Automation")


def save_state(intent: UserIntent = None, plan: ExecutionPlan = None):
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            try:
                state = json.load(f)
            except Exception:
                pass
    if intent:
        state["intent"] = intent.model_dump()
    if plan:
        state["plan"] = plan.model_dump()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                pass
    return {}


def _resolve_session_id_and_state(resume: bool, file_state: dict) -> tuple[str, dict]:
    """
    Prefer DB resumable session when --resume; otherwise new UUID.
    `file_state` is optional in-memory state from session_state.json prompts.
    """
    merged = dict(file_state or {})
    if resume:
        rows = list_resumable_sessions(1)
        if rows:
            sid = rows[0][0]
            intent_d, plan_d = load_session_dicts(sid)
            if intent_d:
                merged["intent"] = intent_d
            if plan_d:
                merged["plan"] = plan_d
            return sid, merged
    sid = str(uuid.uuid4())
    if merged.get("intent"):
        ensure_session_row(sid, user_name=(merged.get("intent") or {}).get("user_name"))
        try:
            intent_obj = UserIntent.model_validate(merged["intent"])
            plan_obj = (
                ExecutionPlan.model_validate(merged["plan"])
                if merged.get("plan")
                else None
            )
            upsert_session_state(sid, intent_obj, plan_obj)
        except Exception:
            pass
    return sid, merged


@app.command()
def start(
    mode: str = typer.Option("supervised", help="Execution mode: 'supervised' or 'autonomous'"),
    resume: bool = typer.Option(False, help="Resume from an existing session if present"),
):
    """
    Start the autonomous research outreach agent.
    """
    if mode.lower() == "autonomous":
        console.print(
            "[yellow]Note: autonomous scheduling is not wired to this entrypoint; "
            "running the supervised pipeline.[/yellow]"
        )

    console.print(
        Panel.fit(
            "[bold cyan]🤖 Welcome to the Autonomous Research Outreach Agent[/bold cyan]\n"
            "Let's find the perfect professors and draft personalized emails.",
            border_style="cyan",
        )
    )

    missing_vars = validate_config()
    if missing_vars:
        console.print(f"[bold red]❌ Missing required env vars:[/bold red] {missing_vars}")
        sys.exit(1)

    init_db()
    console.print("[dim]Database initialized.[/dim]\n")

    file_state: dict = {}
    if resume:
        file_state = load_state()
    elif not resume and os.path.exists(STATE_FILE):
        from rich.prompt import Prompt

        ans = Prompt.ask(
            "\n[bold yellow]Found previous session file. Resume from it?[/bold yellow]",
            choices=["y", "n"],
            default="y",
        )
        if ans == "y":
            file_state = load_state()
        else:
            file_state = {}
            try:
                os.remove(STATE_FILE)
            except OSError:
                pass

    session_id, state = _resolve_session_id_and_state(resume, file_state)
    run_id = new_run_id()
    set_run_context(run_id=run_id, session_id=session_id)
    ensure_session_row(session_id)

    with timed_span("pipeline_supervised", session_id=session_id):
        _run_supervised_pipeline(session_id, state)

    console.print("\n[bold green]✓ Agent workflow completed.[/bold green]")


def _run_supervised_pipeline(session_id: str, state: dict):
    from agents.crawler_agent import CrawlerAgent
    from agents.email_agent import EmailAgent
    from agents.intent_agent import IntentAgent
    from agents.planning_agent import PlanningAgent
    from db import database as db
    from rich.prompt import Prompt

    console.print("\n[bold cyan]═══ Step 1/5: Intent Clarification ═══[/bold cyan]")
    with timed_span("step_intent", session_id=session_id):
        intent_agent = IntentAgent()

        if state and state.get("intent"):
            intent_agent.intent = UserIntent.model_validate(state["intent"])
            console.print("[green]Restored previous intent.[/green]")
            intent = intent_agent.intent
        else:
            intent = intent_agent.run_cli()
            save_state(intent=intent)

        append_session_revision(session_id, "intent", {"intent_saved": True})
        upsert_session_state(session_id, intent, None)

    console.print("\n[bold cyan]═══ Step 2/5: Execution Planning ═══[/bold cyan]")
    with timed_span("step_planning", session_id=session_id):
        planning_agent = PlanningAgent()

        if state and state.get("plan"):
            plan = ExecutionPlan.model_validate(state["plan"])
            console.print("[green]Restored previous execution plan.[/green]")
        else:
            plan = planning_agent.run_cli(intent)
            save_state(plan=plan)

        append_session_revision(session_id, "planning", {"plan_saved": True})
        upsert_session_state(session_id, intent, plan)

    console.print("\n[bold cyan]═══ Step 3/5: Crawling for Professors ═══[/bold cyan]")
    with timed_span("step_crawl", session_id=session_id):
        crawler = CrawlerAgent()
        professors = crawler.run(plan)
        professors = dedupe_professor_profiles(professors)
        professors = enrich_professor_emails_batch(professors)
        append_session_revision(
            session_id,
            "crawling",
            {"professors_found": len(professors)},
        )

    if not professors:
        console.print("[yellow]⚠ No professors found. Try broadening your search terms.[/yellow]")
        return

    console.print(
        f"\n[bold cyan]═══ Step 4/5: Scoring & Shortlisting ({len(professors)} candidates) ═══[/bold cyan]"
    )
    with timed_span("step_scoring", session_id=session_id):
        scorer = RelevanceScorer()
        result = scorer.shortlist(professors, intent, plan)
        shortlisted = result.shortlisted

    if not shortlisted:
        print_shortlist_guidance(result.diagnostics, intent, plan)
        return

    append_session_revision(
        session_id,
        "scoring",
        {"shortlisted": len(shortlisted)},
    )
    save_professors_batch(shortlisted, session_id=session_id)

    console.print(f"[green]✓ {len(shortlisted)} professors shortlisted.[/green]")

    console.print(f"\n[bold cyan]═══ Step 5/5: Generating & Reviewing Emails ═══[/bold cyan]")
    with timed_span("step_email_draft", session_id=session_id):
        email_agent = EmailAgent()
        drafts = email_agent.generate_batch(shortlisted, intent, plan)

    approved_count = 0
    skipped_count = 0

    for i, (draft, prof) in enumerate(zip(drafts, shortlisted), 1):
        console.print(f"\n[bold]─── Email {i}/{len(drafts)} ───[/bold]")
        if prof.relevance_breakdown:
            bd = prof.relevance_breakdown
            console.print(
                f"[dim]Score breakdown: kw={bd.keyword_overlap:.2f} emb={bd.embedding_sim:.2f} "
                f"rec={bd.recency:.2f} geo={bd.geographic:.2f} acc={bd.accepting_students:.2f} "
                f"| terms: {', '.join(bd.matched_terms[:6]) or '—'}[/dim]"
            )
            if prof.email_confidence is not None:
                console.print(
                    f"[dim]Email confidence: {prof.email_confidence:.2f} ({prof.email_source or 'unknown'})[/dim]"
                )
        email_agent.display_email(draft, prof)

        if not draft.recipient_email:
            console.print("[yellow]⚠ No email address found for this professor. Skipping.[/yellow]")
            skipped_count += 1
            continue

        if draft.variants:
            choice = Prompt.ask(
                "[bold yellow]Email variant[/bold yellow]",
                choices=["a", "b"],
                default="a",
            )
            draft = draft.model_copy(
                update={"selected_variant_index": 0 if choice == "a" else 1}
            ).apply_selected_variant()

        while True:
            action = Prompt.ask(
                "\n[bold yellow]Action[/bold yellow]",
                choices=["approve", "skip", "quit"],
                default="approve",
            )
            if action == "approve":
                db.save_draft_email(draft, session_id)
                console.print("[green]✓ Draft saved to database.[/green]")
                approved_count += 1
                break
            if action == "skip":
                skipped_count += 1
                break
            if action == "quit":
                console.print("[yellow]Exiting review early.[/yellow]")
                return

    append_session_revision(
        session_id,
        "email_draft",
        {"approved": approved_count, "skipped": skipped_count},
    )

    console.print(
        f"\n[bold green]✓ Review complete![/bold green] "
        f"{approved_count} approved, {skipped_count} skipped."
    )
    if approved_count > 0:
        console.print(
            "\n[dim]Tip: Run [bold]python3 main.py review[/bold] to launch the UI "
            "and send approved emails.[/dim]"
        )
        console.print(f"[dim]Structured logs under [bold]{RUN_LOG_DIR}[/bold][/dim]")


@app.command()
def review():
    """Launch the Streamlit Review UI."""
    import subprocess
    from pathlib import Path

    ui_path = Path(__file__).parent / "ui" / "review_ui.py"
    console.print(f"[cyan]Launching Streamlit Dashboard from {ui_path}...[/cyan]")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])
    except KeyboardInterrupt:
        console.print("Review UI stopped.")


if __name__ == "__main__":
    app()

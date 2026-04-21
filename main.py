"""
Autonomous Research Outreach Agent - Entry Point
"""

import sys
import uuid
from rich.console import Console
from rich.panel import Panel
import typer

from config import validate_config
from orchestrator.state import AgentState, UserIntent, ExecutionPlan
from db.database import init_db, upsert_session_state
import json
import os

STATE_FILE = "session_state.json"

def save_state(intent: UserIntent = None, plan: ExecutionPlan = None):
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            try:
                state = json.load(f)
            except:
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
            except:
                pass
    return {}

console = Console()
app = typer.Typer(help="Autonomous Agentic Research Outreach & Cold Email Automation")


@app.command()
def start(
    mode: str = typer.Option("supervised", help="Execution mode: 'supervised' or 'autonomous'"),
    resume: bool = typer.Option(False, help="Resume from an existing session if present")
):
    """
    Start the autonomous research outreach agent.
    """
    console.print(Panel.fit(
        "[bold cyan]🤖 Welcome to the Autonomous Research Outreach Agent[/bold cyan]\n"
        "Let's find the perfect professors and draft personalized emails.",
        border_style="cyan"
    ))

    # 1. Validate Config
    missing_vars = validate_config()
    if missing_vars:
        console.print(f"[bold red]❌ Missing required env vars:[/bold red] {missing_vars}")
        sys.exit(1)

    # 2. Initialize Database
    init_db()
    console.print("[dim]Database initialized.[/dim]\n")

    # 3. Resume Logic
    state = load_state() if resume else {}
    if not resume and state:
        from rich.prompt import Prompt
        ans = Prompt.ask("\n[bold yellow]Found previous session. Resume?[/bold yellow]", choices=["y", "n"], default="y")
        if ans == "y":
            state = load_state()
        else:
            state = {}
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)

    session_id = str(uuid.uuid4())
    _run_supervised_pipeline(session_id, state)

    console.print("\n[bold green]✓ Agent workflow completed.[/bold green]")


def _run_supervised_pipeline(session_id: str, state: dict):
    """
    Runs the full supervised pipeline sequentially:
    IntentAgent → PlanningAgent → CrawlerAgent → Scorer → EmailAgent → Review
    Each step runs its CLI, then passes its result to the next step.
    """
    # ── Step 1: Collect Intent ────────────────────────────────────────────────
    console.print("\n[bold cyan]═══ Step 1/5: Intent Clarification ═══[/bold cyan]")
    from agents.intent_agent import IntentAgent
    intent_agent = IntentAgent()
    
    if state and state.get("intent"):
        intent_agent.intent = UserIntent.model_validate(state["intent"])
        console.print("[green]Restored previous intent.[/green]")
        intent = intent_agent.intent
    else:
        intent = intent_agent.run_cli()
        save_state(intent=intent)

    # ── Step 2: Generate Plan ─────────────────────────────────────────────────
    console.print("\n[bold cyan]═══ Step 2/5: Execution Planning ═══[/bold cyan]")
    from agents.planning_agent import PlanningAgent
    planning_agent = PlanningAgent()
    
    if state and state.get("plan"):
        plan = ExecutionPlan.model_validate(state["plan"])
        console.print("[green]Restored previous execution plan.[/green]")
    else:
        plan = planning_agent.run_cli(intent)
        save_state(plan=plan)

    # Persist for review UI (CV attachment, plan flags)
    upsert_session_state(session_id, intent, plan)

    # ── Step 3: Crawl for Professors ──────────────────────────────────────────
    console.print("\n[bold cyan]═══ Step 3/5: Crawling for Professors ═══[/bold cyan]")
    from agents.crawler_agent import CrawlerAgent
    crawler = CrawlerAgent()
    professors = crawler.run(plan)

    if not professors:
        console.print("[yellow]⚠ No professors found. Try broadening your search terms.[/yellow]")
        return

    # ── Step 4: Score & Shortlist ─────────────────────────────────────────────
    console.print(f"\n[bold cyan]═══ Step 4/5: Scoring & Shortlisting ({len(professors)} candidates) ═══[/bold cyan]")
    from core.relevance_scorer import RelevanceScorer
    scorer = RelevanceScorer()
    shortlisted = scorer.shortlist(professors, intent, plan)

    if not shortlisted:
        console.print("[yellow]⚠ No professors met the relevance threshold.[/yellow]")
        return

    console.print(f"[green]✓ {len(shortlisted)} professors shortlisted.[/green]")

    # ── Step 5: Generate & Review Emails ─────────────────────────────────────
    console.print(f"\n[bold cyan]═══ Step 5/5: Generating & Reviewing Emails ═══[/bold cyan]")
    from agents.email_agent import EmailAgent
    from rich.prompt import Prompt
    from db import database as db

    email_agent = EmailAgent()
    drafts = email_agent.generate_batch(shortlisted, intent, plan)

    approved_count = 0
    skipped_count = 0

    for i, (draft, prof) in enumerate(zip(drafts, shortlisted), 1):
        console.print(f"\n[bold]─── Email {i}/{len(drafts)} ───[/bold]")
        email_agent.display_email(draft, prof)

        if not draft.recipient_email:
            console.print("[yellow]⚠ No email address found for this professor. Skipping.[/yellow]")
            skipped_count += 1
            continue

        while True:
            action = Prompt.ask(
                "\n[bold yellow]Action[/bold yellow]",
                choices=["approve", "skip", "quit"],
                default="approve"
            )
            if action == "approve":
                db.save_draft_email(draft, session_id)
                console.print("[green]✓ Draft saved to database.[/green]")
                approved_count += 1
                break
            elif action == "skip":
                skipped_count += 1
                break
            elif action == "quit":
                console.print("[yellow]Exiting review early.[/yellow]")
                return

    console.print(
        f"\n[bold green]✓ Review complete![/bold green] "
        f"{approved_count} approved, {skipped_count} skipped."
    )
    if approved_count > 0:
        console.print(
            "\n[dim]Tip: Run [bold]python3 main.py review[/bold] to launch the UI "
            "and send approved emails.[/dim]"
        )


@app.command()
def review():
    """
    Launch the Streamlit Review UI.
    """
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

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
    Runs the full supervised pipeline using LangGraph orchestration.
    """
    from orchestrator.graph import build_graph
    from orchestrator.state import AgentState, AgentMode, UserIntent, ExecutionPlan
    
    app = build_graph()
    config_dict = {"configurable": {"thread_id": session_id}}
    
    state_to_use = {
        "messages": [],
        "mode": AgentMode.CLARIFICATION.value,
        "session_id": session_id
    }
    if state and state.get("intent"):
        state_to_use["intent"] = state["intent"]
    if state and state.get("plan"):
        state_to_use["plan"] = state["plan"]

    console.print("\n[bold cyan]═══ Starting Autonomous Agent (LangGraph) ═══[/bold cyan]")
    
    if not state.get("intent"):
        from rich.prompt import Prompt
        initial_input = Prompt.ask("\n[bold yellow]What do you want to do?[/bold yellow]")
        state_to_use["human_input"] = initial_input
    
    while True:
        try:
            events = app.stream(state_to_use, config=config_dict)
            for event in events:
                node_name = list(event.keys())[0]
                node_state = event[node_name]
                
                if node_name == "intent" and not node_state.get("intent", {}).get("confirmed"):
                    messages = node_state.get("messages", [])
                    if messages:
                         last_msg = messages[-1]["content"]
                         console.print(f"\n[cyan]Agent:[/cyan] {last_msg}")
                    
                    user_input = input("\n> ")
                    if user_input.lower() in ["quit", "exit"]:
                        sys.exit(0)
                        
                    state_to_use = {"human_input": user_input}
                    break 
                
                elif node_name == "intent" and node_state.get("intent", {}).get("confirmed"):
                    intent_obj = UserIntent.model_validate(node_state["intent"])
                    save_state(intent=intent_obj)
                    state_to_use = None

                elif node_name == "planning" and not node_state.get("plan", {}).get("confirmed"):
                    messages = node_state.get("messages", [])
                    if messages:
                         last_msg = messages[-1]["content"]
                         console.print(f"\n[cyan]Agent:[/cyan] {last_msg}")
                    
                    user_input = input("\n[bold]Approve plan? (yes/no/modify):[/bold] ")
                    if user_input.lower() in ["quit", "exit"]:
                        sys.exit(0)
                        
                    state_to_use = {"human_input": user_input}
                    break
                    
                elif node_name == "planning" and node_state.get("plan", {}).get("confirmed"):
                     intent_obj = UserIntent.model_validate(node_state["intent"])
                     plan_obj = ExecutionPlan.model_validate(node_state["plan"])
                     save_state(intent=intent_obj, plan=plan_obj)
                     upsert_session_state(session_id, intent_obj, plan_obj)
                     state_to_use = None

                elif node_name == "crawling":
                     state_to_use = None
                    
                elif node_name == "scoring":
                     shortlisted = node_state.get("shortlisted", [])
                     if shortlisted:
                         console.print(f"\n[bold cyan]═══ Step 4/5: Scoring & Shortlisting ({len(node_state.get('professors', []))} candidates) ═══[/bold cyan]")
                         console.print(f"[green]✓ {len(shortlisted)} professors shortlisted.[/green]")
                     state_to_use = None

                elif node_name == "email_draft":
                    console.print(f"\n[bold cyan]═══ Step 5/5: Generating & Reviewing Emails ═══[/bold cyan]")
                    drafts_data = node_state.get("draft_emails", [])
                    shortlisted_data = node_state.get("shortlisted", [])
                    
                    from orchestrator.state import DraftEmail, ProfessorProfile
                    from agents.email_agent import EmailAgent
                    from rich.prompt import Prompt
                    from db import database as db
                
                    email_agent = EmailAgent()
                    drafts = [DraftEmail.model_validate(d) for d in drafts_data]
                    shortlisted = [ProfessorProfile.model_validate(p) for p in shortlisted_data]
                
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
                    
                    state_to_use = None
                    
                else:
                    state_to_use = None
                    
            else:
                break
        except Exception as e:
            console.print(f"[red]Workflow Error: {e}[/red]")
            break


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

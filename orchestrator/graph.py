"""
orchestrator/graph.py

Defines the LangGraph orchestration flow linking all agent nodes.
"""

from typing import cast
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from orchestrator.state import AgentState, AgentMode
from agents.intent_agent import intent_node
from agents.planning_agent import planning_node
from agents.crawler_agent import crawl_node
from core.relevance_scorer import scoring_node
from agents.email_agent import email_generation_node

console = Console()

def evaluate_intent(state: AgentState) -> str:
    """Conditional edge after intent extraction."""
    intent_data = state.get("intent")
    if intent_data and intent_data.get("confirmed"):
        return "planning"
    return "intent"

def evaluate_plan(state: AgentState) -> str:
    """Conditional edge after plan generation."""
    plan_data = state.get("plan")
    if plan_data and plan_data.get("confirmed"):
        return "crawling"
    return "planning"

def continue_from_scoring(state: AgentState) -> str:
    """Conditional edge after relevance scoring."""
    shortlisted = state.get("shortlisted", [])
    if not shortlisted:
        console.print("[yellow]⚠ No professors met the relevance threshold. Ending workflow.[/yellow]")
        return END
    return "email_draft"

def send_emulated(state: AgentState) -> AgentState:
    """Mock sending node for the supervised CLI."""
    emails = state.get("draft_emails", [])
    if not emails:
        return state
        
    console.print("\n[bold green]✓ Simulated Sending Completing...[/bold green]")
    messages = state.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": "All approved emails have been processed."
    })
    return {**state, "messages": messages, "mode": "done"}

def build_graph():
    """Builds and compiles the LangGraph StateMachine."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("intent", intent_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("crawling", crawl_node)
    workflow.add_node("scoring", scoring_node)
    workflow.add_node("email_draft", email_generation_node)
    workflow.add_node("send", send_emulated)

    # Define Edges / Routing
    workflow.set_entry_point("intent")
    
    # After intent, we either loop back to intent for clarification, or go to planning.
    workflow.add_conditional_edges("intent", evaluate_intent, {"planning": "planning", "intent": "intent"})
    
    # After planning, we either loop back, or go to crawling.
    workflow.add_conditional_edges("planning", evaluate_plan, {"crawling": "crawling", "planning": "planning"})

    # Crawling always goes to scoring
    workflow.add_edge("crawling", "scoring")
    
    # Scoring conditionally goes to email drafting
    workflow.add_conditional_edges("scoring", continue_from_scoring, {"email_draft": "email_draft", END: END})

    # For the supervised workflow, we pause after generation (so UI or CLI can interrupt)
    # Then eventually go to send
    workflow.add_edge("email_draft", "send")
    workflow.add_edge("send", END)

    # Use a basic memory saver to enable persistence & human-in-the-loop
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def run_supervised_loop(app, config_dict, initial_state: AgentState):
    """
    Run the LangGraph workflow interactively in a CLI setting.
    This simulates human-in-the-loop interrupts by pausing at certain nodes.
    """
    state_to_use = initial_state
    
    while True:
        try:
            # Stream events and state updates from the graph.
            events = app.stream(state_to_use, config=config_dict)
            
            for event in events:
                node_name = list(event.keys())[0]
                node_state = event[node_name]
                
                # Check for Human In The Loop conditions based on agent states.
                if node_name == "intent" and not node_state.get("intent", {}).get("confirmed"):
                    # We need human input
                    messages = node_state.get("messages", [])
                    if messages:
                         last_msg = messages[-1]["content"]
                         console.print(f"\n[cyan]Agent:[/cyan] {last_msg}")
                    
                    user_input = input("\n> ")
                    if user_input.lower() in ["quit", "exit"]:
                        sys.exit(0)
                        
                    # Update state with human input
                    state_to_use = {"human_input": user_input}
                    break # Break the event stream loop to re-feed into app.stream
                
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
                    
                else:
                    # Let the workflow proceed autonomously
                    state_to_use = None
            else:
                # If we exhausted the event iterator without breaking, we're done.
                console.print("\n[bold green]Workflow reached END.[/bold green]")
                break
                
        except Exception as e:
             console.print(f"[red]Workflow Error: {e}[/red]")
             break

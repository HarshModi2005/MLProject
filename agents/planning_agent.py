"""
agents/planning_agent.py

The Planning Agent.

Responsibilities:
  - Accept a confirmed UserIntent
  - Use Gemini to generate a structured ExecutionPlan
  - Present the plan to the user for review/approval
  - Support plan refinement based on user feedback
"""

from __future__ import annotations

import json
import re

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import config
from core.student_context import build_matching_profile_text
from orchestrator.state import (
    UserIntent,
    ExecutionPlan,
    SearchStrategy,
    FilteringCriteria,
    EmailStrategy,
    SendingSchedule,
    AgentState,
)

console = Console()


# ── Prompt Templates ──────────────────────────────────────────────────────────

PLAN_GENERATION_PROMPT = """
You are an expert academic outreach strategist.

Based on the following user intent, generate a detailed execution plan for an automated professor outreach campaign.

USER INTENT:
- Goal: {goal}
- Research Domains: {domains}
- Target Countries: {countries}
- Timeline: {timeline}
- User Profile (summary): {profile}
- Preferred Email Tone: {tone}
- Outreach per week: {outreach_per_week}
- Follow-up after: {follow_up_days} days

STUDENT RESEARCH SIGNALS (from CV / profile — use to expand search keywords toward their **strongest**
technical and research themes; still respect the user's stated domains as primary):
{research_profile}

Generate a JSON execution plan with this EXACT structure:
{{
  "search_strategy": {{
    "keywords": ["<keyword1>", "<keyword2>", ...],
    "country_filters": ["<country1>", ...],
    "institution_filters": []
  }},
  "filtering_criteria": {{
    "publication_recency": true,
    "active_lab_only": true,
    "min_relevance_score": 0.4,
    "require_email": true
  }},
  "email_strategy": {{
    "personalization_level": "high",
    "reference_publications": true,
    "tone": "{tone}",
    "word_count_target": 220,
    "include_cv_attachment": true
  }},
  "sending_schedule": {{
    "emails_per_week": {outreach_per_week},
    "follow_up_days": {follow_up_days},
    "max_follow_ups": 2
  }},
  "top_k_professors": {top_k_default}
}}

Rules for JSON generation:
1. Build 6-10 search keywords from the user's domains PLUS 2-4 terms from STUDENT RESEARCH SIGNALS
   when they strengthen professor discovery (methods, subfields, tools). Do not invent domains the student never touched.
2. IMPORTANT FOR COUNTRIES: Transfer the Exact strings from "Target Countries" to "country_filters". DO NOT normalize regions! If the input states "Punjab, India", the output MUST contain "Punjab, India". Do NOT change it to just "India".

Return ONLY the JSON.
"""

PLAN_REFINEMENT_PROMPT = """
The current execution plan is:
{current_plan_json}

The user wants to make the following change:
"{user_request}"

Update the plan according to the user's request and return the FULL updated plan JSON.
Return ONLY the JSON, no explanation.
"""

PLAN_EXPLANATION_PROMPT = """
Explain the following outreach execution plan in simple, friendly language.
Use bullet points. Keep it concise (under 200 words).
Highlight the key decisions: search approach, how professors will be selected, and how emails will be sent.

Plan:
{plan_json}
"""


# ── Planning Agent Class ──────────────────────────────────────────────────────

class PlanningAgent:
    """
    Generates and refines an ExecutionPlan from a confirmed UserIntent.
    """

    def __init__(self):
        self.llm = config.get_groq_llm()
        self.plan: ExecutionPlan | None = None

    def _call_llm(self, prompt: str) -> str:
        """Call Groq and return raw text."""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            return ""

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return {}

    # ── Plan Generation ───────────────────────────────────────────────────────

    def generate_plan(self, intent: UserIntent) -> ExecutionPlan:
        """
        Call Gemini to generate an ExecutionPlan from the UserIntent.
        """
        console.print("\n[dim]🧠 Generating execution plan...[/dim]")

        prompt = PLAN_GENERATION_PROMPT.format(
            goal=intent.goal.value,
            domains=", ".join(intent.research_domains),
            countries=", ".join(intent.countries),
            timeline=intent.timeline or "as soon as possible",
            profile=intent.user_profile_summary or "Not provided",
            tone=intent.email_tone,
            outreach_per_week=intent.outreach_per_week,
            follow_up_days=intent.follow_up_days,
            research_profile=build_matching_profile_text(intent),
            top_k_default=config.TOP_K_PROFESSORS,
        )

        raw = self._call_llm(prompt)
        data = self._parse_json(raw)

        if not data:
            # Fallback defaults
            console.print("[yellow]⚠ Could not parse LLM plan. Using smart defaults.[/yellow]")
            data = self._default_plan(intent)

        return self._build_plan_from_dict(data)

    def _default_plan(self, intent: UserIntent) -> dict:
        """Fallback plan if LLM parsing fails."""
        return {
            "search_strategy": {
                "keywords": intent.research_domains + ["professor", "faculty", "research group"],
                "country_filters": intent.countries,
                "institution_filters": [],
            },
            "filtering_criteria": {
                "publication_recency": True,
                "active_lab_only": True,
                "min_relevance_score": 0.4,
                "require_email": True,
            },
            "email_strategy": {
                "personalization_level": "high",
                "reference_publications": True,
                "tone": intent.email_tone,
                "word_count_target": 220,
                "include_cv_attachment": True,
            },
            "sending_schedule": {
                "emails_per_week": intent.outreach_per_week,
                "follow_up_days": intent.follow_up_days,
                "max_follow_ups": 2,
            },
            "top_k_professors": config.TOP_K_PROFESSORS,
        }

    def _build_plan_from_dict(self, data: dict) -> ExecutionPlan:
        """Construct a validated ExecutionPlan from raw dict."""
        ss = data.get("search_strategy", {})
        fc = data.get("filtering_criteria", {})
        es = data.get("email_strategy", {})
        sch = data.get("sending_schedule", {})

        return ExecutionPlan(
            search_strategy=SearchStrategy(
                keywords=ss.get("keywords", []),
                country_filters=ss.get("country_filters", []),
                institution_filters=ss.get("institution_filters", []),
            ),
            filtering_criteria=FilteringCriteria(
                publication_recency=fc.get("publication_recency", True),
                active_lab_only=fc.get("active_lab_only", True),
                min_relevance_score=float(fc.get("min_relevance_score", 0.4)),
                require_email=fc.get("require_email", True),
            ),
            email_strategy=EmailStrategy(
                personalization_level=es.get("personalization_level", "high"),
                reference_publications=es.get("reference_publications", True),
                tone=es.get("tone", "professional"),
                word_count_target=int(es.get("word_count_target", 220)),
                include_cv_attachment=es.get("include_cv_attachment", True),
            ),
            sending_schedule=SendingSchedule(
                emails_per_week=int(sch.get("emails_per_week", 10)),
                follow_up_days=int(sch.get("follow_up_days", 7)),
                max_follow_ups=int(sch.get("max_follow_ups", 2)),
            ),
            top_k_professors=max(
                1,
                min(
                    int(data.get("top_k_professors", config.TOP_K_PROFESSORS)),
                    max(1, config.TOP_K_PROFESSORS),
                ),
            ),
        )

    # ── Plan Display ──────────────────────────────────────────────────────────

    def display_plan(self, plan: ExecutionPlan) -> None:
        """Render the plan in a readable rich table format."""
        console.print("\n")
        console.print(Panel(
            "[bold cyan]📊 Generated Execution Plan[/bold cyan]",
            border_style="cyan",
        ))

        # Search strategy table
        t1 = Table(title="🔍 Search Strategy", show_header=True, header_style="bold magenta")
        t1.add_column("Field", style="bold")
        t1.add_column("Value")
        t1.add_row("Keywords", ", ".join(plan.search_strategy.keywords))
        t1.add_row("Countries", ", ".join(plan.search_strategy.country_filters))
        if plan.search_strategy.institution_filters:
            t1.add_row("Institutions", ", ".join(plan.search_strategy.institution_filters))
        t1.add_row("Top-K Professors", str(plan.top_k_professors))
        console.print(t1)

        # Email strategy table
        t2 = Table(title="📧 Email Strategy", show_header=True, header_style="bold blue")
        t2.add_column("Field", style="bold")
        t2.add_column("Value")
        t2.add_row("Personalization", plan.email_strategy.personalization_level)
        t2.add_row("Tone", plan.email_strategy.tone)
        t2.add_row("Word Count Target", str(plan.email_strategy.word_count_target))
        t2.add_row("Ref. Publications", "✓" if plan.email_strategy.reference_publications else "✗")
        t2.add_row("Attach CV", "✓" if plan.email_strategy.include_cv_attachment else "✗")
        console.print(t2)

        # Schedule table
        t3 = Table(title="📅 Sending Schedule", show_header=True, header_style="bold green")
        t3.add_column("Field", style="bold")
        t3.add_column("Value")
        t3.add_row("Emails/Week", str(plan.sending_schedule.emails_per_week))
        t3.add_row("Follow-up After", f"{plan.sending_schedule.follow_up_days} days")
        t3.add_row("Max Follow-ups", str(plan.sending_schedule.max_follow_ups))
        console.print(t3)

        # LLM plain-language explanation
        console.print("\n[dim]Getting plain-language explanation...[/dim]")
        explanation = self._call_llm(
            PLAN_EXPLANATION_PROMPT.format(plan_json=plan.model_dump_json(indent=2))
        )
        if explanation:
            console.print(Panel(explanation, title="[bold]💡 What this means[/bold]", border_style="dim"))

    # ── Plan Refinement ───────────────────────────────────────────────────────

    def refine_plan(self, plan: ExecutionPlan, user_request: str) -> ExecutionPlan:
        """Refine the plan based on user feedback."""
        prompt = PLAN_REFINEMENT_PROMPT.format(
            current_plan_json=plan.model_dump_json(indent=2),
            user_request=user_request,
        )
        raw = self._call_llm(prompt)
        data = self._parse_json(raw)
        if data:
            return self._build_plan_from_dict(data)
        console.print("[yellow]⚠ Could not refine plan. Keeping current version.[/yellow]")
        return plan

    # ── CLI Interface ─────────────────────────────────────────────────────────

    def run_cli(self, intent: UserIntent) -> ExecutionPlan:
        """
        Full CLI flow: generate plan → display → confirm.
        Returns approved ExecutionPlan.
        """
        plan = self.generate_plan(intent)
        self.display_plan(plan)

        while True:
            console.print(
                "\n[bold yellow]Approve this plan? "
                "(yes / no + describe change)[/bold yellow]"
            )
            answer = Prompt.ask("").strip().lower()

            if answer in ("yes", "y", "approve", "ok", "looks good"):
                plan.confirmed = True
                console.print("[bold green]✓ Plan approved! Starting crawl...[/bold green]\n")
                self.plan = plan
                return plan
            elif answer.startswith("no"):
                # Extract what they want to change
                change = answer[2:].strip() or Prompt.ask(
                    "  What would you like to change?"
                )
                console.print("[dim]Refining plan...[/dim]")
                plan = self.refine_plan(plan, change)
                self.display_plan(plan)
            else:
                # Treat the whole string as a change request
                console.print("[dim]Refining plan...[/dim]")
                plan = self.refine_plan(plan, answer)
                self.display_plan(plan)


# ── LangGraph Node ────────────────────────────────────────────────────────────

def planning_node(state: AgentState) -> AgentState:
    """LangGraph node: intent → plan."""
    intent = UserIntent.model_validate(state["intent"])
    agent = PlanningAgent()

    messages = state.get("messages", [])
    human_input = state.get("human_input", "").strip().lower()

    if state.get("plan"):
        plan = ExecutionPlan.model_validate(state["plan"])
        
        # Check human input for confirmation
        if human_input in ("yes", "y", "approve", "ok", "looks good") or human_input.startswith("yes"):
            plan.confirmed = True
            messages.append({"role": "user", "content": human_input})
            messages.append({
                "role": "assistant",
                "content": "Plan approved! Starting crawl...",
            })
            return {
                **state,
                "plan": plan.model_dump(),
                "messages": messages,
                "mode": "crawling",
            }
        elif human_input:
            # User wants to refine it
            messages.append({"role": "user", "content": human_input})
            change = human_input
            if change.startswith("no"):
                change = change[2:].strip() or "Please modify the plan."
            plan = agent.refine_plan(plan, change)
            explanation = agent._call_llm(
                PLAN_EXPLANATION_PROMPT.format(plan_json=plan.model_dump_json(indent=2))
            )
            messages.append({
                "role": "assistant",
                "content": f"Here is your refined execution plan:\n\n{explanation}\n\nType 'yes' to confirm.",
            })
            return {
                **state,
                "plan": plan.model_dump(),
                "messages": messages,
                "mode": "planning",
            }
            
    # Generate new plan if none exists
    plan = agent.generate_plan(intent)
    explanation = agent._call_llm(
        PLAN_EXPLANATION_PROMPT.format(plan_json=plan.model_dump_json(indent=2))
    )

    messages.append({
        "role": "assistant",
        "content": f"Here is your execution plan:\n\n{explanation}\n\nType 'yes' to confirm.",
    })

    return {
        **state,
        "plan": plan.model_dump(),
        "messages": messages,
        "mode": "planning",
    }

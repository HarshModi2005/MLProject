"""
agents/intent_agent.py

The Intent & Clarification Agent.

Responsibilities:
  - Accept a raw natural language goal from the user
  - Extract initial intent + entities using Gemini
  - Track slot-filling state via UserIntent
  - Generate targeted follow-up questions for missing slots
  - Confirm final structured intent before proceeding

The agent runs as a conversational loop (used both in CLI and via LangGraph).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

import config
from orchestrator.state import UserIntent, OutreachGoal, AgentState

console = Console()

# ── Slot question templates ───────────────────────────────────────────────────
SLOT_QUESTIONS: dict[str, str] = {
    "research_domains": (
        "🔬 What research domains are you interested in? "
        "(e.g. machine learning, computer vision, NLP, robotics — comma-separated)"
    ),
    "countries": (
        "🌍 Which countries/regions are you targeting? "
        "(e.g. USA, Germany, UK, Canada — comma-separated, or 'any' for worldwide)"
    ),
    "timeline": (
        "📅 What is your target timeline? "
        "(e.g. Summer 2025, Fall 2025, immediately)"
    ),
    "user_name": (
        "👤 What is your full name? (for personalizing the emails)"
    ),
    "user_profile": (
        "📄 Please briefly describe your background: your degree, year, key skills, "
        "and notable projects or publications. (Or type 'cv' to provide a CV file path)"
    ),
}


# ── LLM Prompt Templates ──────────────────────────────────────────────────────

INTENT_EXTRACTION_PROMPT = """
You are an intelligent assistant helping a student set up an automated academic outreach campaign.

The user has said: "{user_input}"

Extract the following as a JSON object. For any field you cannot confidently identify, use null.

{{
  "goal": "<one of: research_internship | phd_inquiry | collaboration | industry_outreach | postdoc | other>",
  "research_domains": ["<domain1>", "<domain2>"],
  "countries": ["<country1>", "<country2>"],
  "target_institutions": ["<uni1>", "<uni2>"],
  "timeline": "<e.g. Summer 2025 or null>",
  "user_name": "<name or null>",
  "email_tone": "<professional | friendly | concise>",
  "extra_notes": "<any specific preferences mentioned or null>"
}}

Rules:
- IMPORTANT FOR COUNTRIES: If the user provides a specific state, city, or region (e.g., "Punjab, India" or "Boston, USA"), PRESERVE the exact string they typed. Do NOT strip out the region to just the country.

Return ONLY the JSON object, no explanation.
"""

SLOT_PARSE_PROMPT = """
The user was asked: "{question}"
The user replied: "{user_reply}"
The slot being filled is: "{slot}"

Based on the reply, extract the value for this slot.
Return ONLY a JSON with key "{slot}" and the extracted value.

Rules:
- For list fields (research_domains, countries, target_institutions): return a JSON array of strings.
- For scalar fields (timeline, user_name, email_tone): return a string.
- Clean up any obvious typos in domain names.
- If the user said "any" for countries, return ["Worldwide"].
- IMPORTANT FOR COUNTRIES: If the user provides a specific state, city, or region (e.g., "Punjab, India" or "Boston, USA"), PRESERVE the exact string they typed. Do NOT strip out the region to just the country.
- Normalize isolated acronyms (e.g. "US" → "USA", "UK" → "United Kingdom"), but preserve full location contexts.

Return ONLY JSON.
"""

PROFILE_PARSE_PROMPT = """
The user provided the following as their academic/research background:

"{user_text}"

Summarize this into a concise 3-4 sentence professional profile suitable for
inclusion in a cold email. Focus on:
- Degree and institution
- Research interests
- Key projects, skills, or publications
- What they are looking for

Return ONLY the summary text, no extra formatting.
"""

PROFILE_STRUCT_PROMPT = """
The user described their academic background (free text, not a file):

"{user_text}"

Extract facts for a **research internship** outreach agent. Curate highlights — do not dump everything.

Return ONLY this JSON:
{{
  "name": null,
  "degree": null,
  "institution": null,
  "institution_tag": null,
  "cgpa": null,
  "academic_rank": null,
  "research_interests_from_cv": [],
  "research_experience": [],
  "work_research_adjacent": [],
  "key_projects": [],
  "publications": [],
  "skills": [],
  "achievements": [],
  "research_internship_highlights": ["4-7 bullets, max 22 words each, same rules as a CV parser: research-first"],
  "profile_summary": "3-4 sentences, faculty-facing, not a list of every win"
}}
"""

CONFIRMATION_PROMPT = """
You are helping confirm a structured outreach plan.

Here is the current state of the user's intent:
{intent_json}

Generate a friendly, clear confirmation summary of what was understood.
Format it as bullet points. End with:
"Does this look correct? (yes to confirm / no to make changes)"
"""


# ── Core Intent Agent Class ───────────────────────────────────────────────────

class IntentAgent:
    """
    Conversational slot-filling agent that collects and validates
    the user's outreach intent.
    """

    def __init__(self):
        self.llm = config.get_groq_llm()
        self.intent = UserIntent()

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call Groq and return the raw text response."""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            return ""

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Strip markdown code blocks if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = text.rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return {}

    # ── Intent Extraction ─────────────────────────────────────────────────────

    def extract_initial_intent(self, user_input: str) -> None:
        """
        Parse the first user message to extract whatever intent signals exist.
        Populates self.intent with any identified fields.
        """
        prompt = INTENT_EXTRACTION_PROMPT.format(user_input=user_input)
        raw = self._call_llm(prompt)
        data = self._parse_json_response(raw)

        if not data:
            return

        # Map extracted goal string to enum
        goal_str = data.get("goal", "research_internship")
        try:
            self.intent.goal = OutreachGoal(goal_str)
        except ValueError:
            self.intent.goal = OutreachGoal.RESEARCH_INTERNSHIP

        # Populate list fields
        domains = data.get("research_domains") or []
        if domains:
            self.intent.research_domains = [d.strip() for d in domains if d]

        countries = data.get("countries") or []
        if countries:
            self.intent.countries = [c.strip() for c in countries if c]

        institutions = data.get("target_institutions") or []
        if institutions:
            self.intent.target_institutions = [i.strip() for i in institutions if i]

        # Scalar fields
        if data.get("timeline"):
            self.intent.timeline = data["timeline"]
        if data.get("user_name"):
            self.intent.user_name = data["user_name"]
        if data.get("email_tone"):
            self.intent.email_tone = data.get("email_tone", "professional")
        if data.get("extra_notes"):
            self.intent.extra_notes = data["extra_notes"]

    # ── Slot Filling ──────────────────────────────────────────────────────────

    def fill_slot_from_reply(self, slot: str, question: str, reply: str) -> None:
        """
        Use LLM to parse a user's free-form reply into the correct slot value.
        """
        prompt = SLOT_PARSE_PROMPT.format(
            question=question,
            user_reply=reply,
            slot=slot,
        )
        raw = self._call_llm(prompt)
        data = self._parse_json_response(raw)

        if slot == "research_domains" and isinstance(data.get(slot), list):
            self.intent.research_domains = data[slot]
        elif slot == "countries" and isinstance(data.get(slot), list):
            self.intent.countries = data[slot]
        elif slot == "timeline" and data.get(slot):
            self.intent.timeline = str(data[slot])
        elif slot == "user_name" and data.get(slot):
            self.intent.user_name = str(data[slot])
        elif slot == "user_profile":
            # Strip surrounding quotes if user pasted path with quotes e.g. '/path/to/cv.pdf'
            stripped = reply.strip().strip("'\"")
            # Check if it's a direct file path to a PDF
            import os as _os
            if stripped.lower().endswith(".pdf") or _os.path.isfile(stripped):
                self.parse_cv(stripped)
            elif reply.strip().lower() == "cv":
                return  # handled upstream
            else:
                self._parse_freeform_profile(reply)

    def _summarize_profile(self, user_text: str) -> str:
        """Use LLM to distill user's raw profile text into a clean summary."""
        prompt = PROFILE_PARSE_PROMPT.format(user_text=user_text)
        result = self._call_llm(prompt)
        return result if result else user_text[:500]

    def _parse_freeform_profile(self, user_text: str) -> None:
        """Extract structured credentials from pasted text (same signals as CV parse)."""
        prompt = PROFILE_STRUCT_PROMPT.format(user_text=user_text[:4500])
        raw = self._call_llm(prompt)
        data = self._parse_json_response(raw)
        if not data:
            self.intent.user_profile_summary = self._summarize_profile(user_text)
            return
        self._apply_structured_profile(data)
        ps = data.get("profile_summary")
        if ps and str(ps).strip():
            self.intent.user_profile_summary = str(ps).strip()
        elif not self.intent.user_profile_summary:
            self.intent.user_profile_summary = self._summarize_profile(user_text)

    def parse_cv(self, cv_path: str) -> None:
        """
        Parse a PDF CV with structured extraction (CGPA, rank, achievements, etc.)
        and merge into intent for personalized emails.
        """
        import os

        if not os.path.isfile(cv_path):
            console.print(f"[red]File not found: {cv_path}[/red]")
            return

        from core.cv_parser import CVParser

        parser = CVParser()
        data = parser.parse(cv_path)
        self.intent.cv_path = cv_path

        if not data:
            console.print("[yellow]⚠ CV parse returned no data.[/yellow]")
            return

        self._apply_structured_profile(data)

        ps = data.get("profile_summary")
        if ps and str(ps).strip():
            self.intent.user_profile_summary = str(ps).strip()
        elif not self.intent.user_profile_summary:
            self.intent.user_profile_summary = (
                "Background from CV; re-run with a clearer PDF if this looks thin."
            )

    def _apply_structured_profile(self, data: dict) -> None:
        """Merge LLM-extracted CV / profile fields into intent for email generation."""
        if data.get("name") and not self.intent.user_name:
            self.intent.user_name = str(data["name"]).strip()
        self.intent.cv_degree = data.get("degree")
        self.intent.cv_institution = data.get("institution")
        self.intent.cv_institution_tag = data.get("institution_tag")
        self.intent.cv_cgpa = data.get("cgpa")
        self.intent.cv_rank = data.get("academic_rank")
        self.intent.cv_achievements = list(data.get("achievements") or [])
        self.intent.cv_key_projects = list(data.get("key_projects") or [])
        self.intent.cv_publications_list = list(data.get("publications") or [])
        self.intent.cv_skills = list(data.get("skills") or [])
        self.intent.cv_research_interests = list(
            data.get("research_interests_from_cv") or data.get("research_interests") or []
        )
        self.intent.cv_research_experience = list(data.get("research_experience") or [])
        self.intent.cv_work_research_adjacent = list(data.get("work_research_adjacent") or [])
        self.intent.research_internship_highlights = list(
            data.get("research_internship_highlights") or []
        )

    # ── Confirmation ──────────────────────────────────────────────────────────

    def generate_confirmation_summary(self) -> str:
        """Generate a human-readable confirmation of the filled intent."""
        prompt = CONFIRMATION_PROMPT.format(
            intent_json=self.intent.model_dump_json(indent=2)
        )
        return self._call_llm(prompt)

    # ── CLI Interface ─────────────────────────────────────────────────────────

    def run_cli(self) -> UserIntent:
        """
        Full interactive CLI loop.
        Returns the confirmed UserIntent when done.
        """
        console.print(Panel(
            "[bold cyan]🤖 Research Outreach Agent[/bold cyan]\n"
            "[dim]I'll help you set up a personalized academic outreach campaign.[/dim]",
            border_style="cyan",
        ))

        # Step 1: Get initial input
        user_input = Prompt.ask(
            "\n[bold yellow]What do you want to do?[/bold yellow]"
        )

        console.print("\n[dim]Analyzing your request...[/dim]")
        self.extract_initial_intent(user_input)

        # Step 2: Fill missing slots
        missing = self.intent.missing_required_slots()
        if missing:
            console.print(
                f"\n[dim]I need a few more details. "
                f"({len(missing)} question{'s' if len(missing) > 1 else ''} remaining)[/dim]\n"
            )

        while missing:
            slot = missing[0]
            question = SLOT_QUESTIONS.get(slot, f"Please provide: {slot}")
            console.print(f"[bold green]►[/bold green] {question}")
            reply = Prompt.ask("")

            # Special case: user wants to provide CV file
            if slot == "user_profile" and reply.strip().lower() == "cv":
                cv_path = Prompt.ask(
                    "  [dim]Enter the full path to your CV (PDF)[/dim]"
                )
                self.parse_cv(cv_path.strip())
            else:
                self.fill_slot_from_reply(slot, question, reply)

            missing = self.intent.missing_required_slots()
            if missing:
                console.print()

        # Step 3: Confirm
        console.print("\n[dim]Generating confirmation summary...[/dim]\n")
        summary = self.generate_confirmation_summary()
        console.print(Panel(summary, title="[bold]📋 Confirmation[/bold]", border_style="blue"))

        while True:
            answer = Prompt.ask("[bold yellow]Your answer[/bold yellow]").strip().lower()
            if answer in ("yes", "y", "confirm", "correct"):
                self.intent.confirmed = True
                console.print("[bold green]✓ Intent confirmed! Moving to planning...[/bold green]\n")
                break
            elif answer in ("no", "n", "change", "edit"):
                console.print(
                    "\n[yellow]What would you like to change? "
                    "(Type the slot name or describe the change)[/yellow]"
                )
                change = Prompt.ask("")
                self._handle_change_request(change)
                # Re-show updated summary
                console.print("\n[dim]Updated summary:[/dim]")
                summary = self.generate_confirmation_summary()
                console.print(Panel(summary, title="[bold]📋 Updated Confirmation[/bold]", border_style="blue"))
            else:
                console.print("[dim]Please type 'yes' to confirm or 'no' to make changes.[/dim]")

        return self.intent

    def _handle_change_request(self, change_text: str) -> None:
        """Handle free-form change requests from the user during confirmation."""
        # Simple keyword matching for slot re-filling
        sl = change_text.lower()
        if any(k in sl for k in ("domain", "research", "field")):
            console.print(SLOT_QUESTIONS["research_domains"])
            reply = Prompt.ask("")
            self.fill_slot_from_reply("research_domains", SLOT_QUESTIONS["research_domains"], reply)
        elif any(k in sl for k in ("country", "countries", "location", "region")):
            console.print(SLOT_QUESTIONS["countries"])
            reply = Prompt.ask("")
            self.fill_slot_from_reply("countries", SLOT_QUESTIONS["countries"], reply)
        elif any(k in sl for k in ("timeline", "time", "when", "semester", "summer", "fall")):
            console.print(SLOT_QUESTIONS["timeline"])
            reply = Prompt.ask("")
            self.fill_slot_from_reply("timeline", SLOT_QUESTIONS["timeline"], reply)
        elif any(k in sl for k in ("profile", "background", "cv", "resume")):
            console.print(SLOT_QUESTIONS["user_profile"])
            reply = Prompt.ask("")
            if reply.strip().lower() == "cv":
                cv_path = Prompt.ask("  [dim]Enter CV path[/dim]")
                self.parse_cv(cv_path.strip())
            else:
                self.fill_slot_from_reply("user_profile", SLOT_QUESTIONS["user_profile"], reply)
        elif any(k in sl for k in ("tone", "style", "email tone")):
            reply = Prompt.ask("Preferred tone (professional / friendly / concise) ")
            self.intent.email_tone = reply.strip().lower()
        else:
            console.print("[dim]Using LLM to interpret your change request...[/dim]")
            # Generic LLM re-extraction
            self.extract_initial_intent(change_text)


# ── LangGraph Node Function ───────────────────────────────────────────────────

def intent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: processes the latest human_input to update intent state.
    This is called at each turn of the clarification loop.
    """
    agent = IntentAgent()

    # Restore existing intent if present
    if state.get("intent"):
        agent.intent = UserIntent.model_validate(state["intent"])

    human_input = state.get("human_input", "")
    messages = state.get("messages", [])

    if not messages:
        # First turn: extract initial intent
        agent.extract_initial_intent(human_input)
        missing = agent.intent.missing_required_slots()

        next_question = ""
        if missing:
            slot = missing[0]
            next_question = SLOT_QUESTIONS.get(slot, f"Please provide: {slot}")

        messages.append({"role": "user", "content": human_input})
        messages.append({"role": "assistant", "content": next_question})
    else:
        # Subsequent turns: process user reply
        last_msg = messages[-1]["content"] if messages else ""
        
        # Check if the last msg was the confirmation prompt
        if "Does this look correct?" in last_msg:
            ans = human_input.strip().lower()
            if ans in ("yes", "y", "confirm", "correct") or ans.startswith("yes"):
                agent.intent.confirmed = True
            elif ans in ("no", "n", "change", "edit"):
                pass # The next turn should ask what they want to change
            else:
                agent._handle_change_request(human_input)
        else:
            # Identify which slot we just asked
            slot = _identify_slot_from_question(last_msg)
            if slot:
                agent.fill_slot_from_reply(slot, last_msg, human_input)

        messages.append({"role": "user", "content": human_input})

        if agent.intent.confirmed:
            messages.append({"role": "assistant", "content": "Intent confirmed. Proceeding to planning..."})
        else:
            missing = agent.intent.missing_required_slots()
            if missing:
                next_question = SLOT_QUESTIONS.get(missing[0], f"Please provide: {missing[0]}")
                messages.append({"role": "assistant", "content": next_question})
            else:
                # All slots filled — generate confirmation
                summary = agent.generate_confirmation_summary()
                messages.append({"role": "assistant", "content": summary})

    return {
        **state,
        "intent": agent.intent.model_dump(),
        "messages": messages,
        "mode": "clarification" if not agent.intent.is_complete() else "planning",
    }


def _identify_slot_from_question(question: str) -> Optional[str]:
    """Map a question string back to its slot name."""
    for slot, q in SLOT_QUESTIONS.items():
        if q[:40] in question:
            return slot
    return None

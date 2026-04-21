"""
agents/email_agent.py

The Email Generation Agent.

Responsibilities:
  - For each shortlisted professor, generate a personalized cold email using Gemini
  - Ground the email content in the professor's actual research / publications
  - Align with the user's profile and communication goals
  - Output a structured DraftEmail object
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel

import config
from core.observability import log_llm_call
from core.student_context import build_email_internship_context, highlights_for_intent
from orchestrator.state import (
    AgentState,
    DraftEmail,
    DraftVariant,
    EmailStatus,
    ExecutionPlan,
    ProfessorProfile,
    UserIntent,
)

console = Console()

# Nav / page-section junk often wrongly stored as "keywords" or ends up as a fake "surname"
_BAD_SURNAME_TOKENS = frozenset({
    "highlights", "profile", "overview", "home", "contact", "research", "work",
    "academics", "qualifications", "interests", "publications", "news", "biography",
    "bio", "cv", "vita", "page", "menu", "more", "group", "team", "lab", "faculty",
    "staff", "people", "directory", "education", "teaching", "student", "students",
})
_KEYWORD_JUNK_SUBSTR = (
    "brief research", "research profile", "academic profile", "research work",
    "research interests", "qualifications", "highlights", "navigation", "breadcrumb",
    "phd (computer", "be (computer", "me (computer", "ms (computer",
)


def _salutation_for_professor(full_name: str) -> str:
    """Safe opening line; never use section headers like 'Highlights' as a surname."""
    if not full_name or not str(full_name).strip():
        return "Dear Professor,"
    n = re.sub(r"\s+", " ", str(full_name).strip())
    n = re.sub(
        r"^(dr\.?|prof\.?|professor|mr\.?|mrs\.?|ms\.?|sir|madam)\s+",
        "",
        n,
        flags=re.I,
    )
    parts = [p for p in n.split() if p]
    if not parts:
        return "Dear Professor,"
    raw_last = parts[-1]
    last = re.sub(r"[^\w\-']", "", raw_last)
    if len(last) < 2 or last.lower() in _BAD_SURNAME_TOKENS:
        return "Dear Professor,"
    return f"Dear Professor {last},"


def _sanitize_keywords(keywords: list[str], *, max_items: int = 5, max_total_chars: int = 220) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for k in keywords or []:
        s = (k or "").strip()
        if len(s) < 3 or len(s) > 100:
            continue
        low = s.lower()
        if any(j in low for j in _KEYWORD_JUNK_SUBSTR):
            continue
        if low in _BAD_SURNAME_TOKENS:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
        if len(out) >= max_items:
            break
    # trim total length
    joined = []
    total = 0
    for s in out:
        if total + len(s) > max_total_chars:
            break
        joined.append(s)
        total += len(s) + 2
    return joined


def _force_opening_salutation(body: str, salutation: str) -> str:
    """Ensure the email opens with the vetted salutation (fixes 'Dear Professor Highlights')."""
    b = (body or "").strip()
    sal = salutation.strip()
    if not b:
        return sal

    def _norm_greeting(line: str) -> str:
        return line.strip().lower().rstrip(",").rstrip(".")

    first = b.split("\n", 1)[0].strip()
    if _norm_greeting(first) == _norm_greeting(sal):
        return b
    if first.lower().startswith("dear "):
        rest = b.split("\n", 1)[1].lstrip() if "\n" in b else ""
        return f"{sal}\n\n{rest}" if rest else sal
    return f"{sal}\n\n{b}"


def _body_word_count(body: str) -> int:
    if not body:
        return 0
    return len(re.findall(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?", body))


def _email_has_garbage(body: str) -> bool:
    t = (body or "").strip().lower()
    if not t:
        return True
    if "professor highlights" in t or "dear professor highlights" in t:
        return True
    if t.count("dear professor") > 1:
        return True
    if "brief research profile" in t and "phd (" in t:
        return True
    return False


def _email_too_short(body: str, min_words: int) -> bool:
    return _body_word_count(body) < min_words


def _email_body_failed_qc(body: str, prof_name: str, *, min_words: int) -> bool:
    """True if body should be discarded or expanded (garbage or too thin)."""
    if _email_has_garbage(body):
        return True
    return _email_too_short(body, min_words)


def _email_targets(word_count_target: int) -> tuple[int, int, int]:
    """(target, min_words, max_words) for prompts and QC."""
    target = max(200, int(word_count_target))
    min_w = max(130, min(180, int(target * 0.54)))
    max_w = min(330, target + 55)
    return target, min_w, max_w


# ── Prompt Templates ──────────────────────────────────────────────────────────

EMAIL_GENERATION_PROMPT = """
You are an expert writer of **research internship** emails to faculty. Output must feel human, specific, and
well-developed — not sparse and not a keyword dump.

PROFESSOR DETAILS:
- Full name (context only): {prof_name}
- Institution: {institution}
- Department: {department}
- Research topics (sanitized): {research_areas}
- Recent Publications: {publications}
- Lab/Profile Page: {lab_url}

LINE 1 of "body" (copy EXACTLY — then blank line before paragraph 2):
{salutation}

STUDENT CONTEXT (facts only; do not invent beyond this):
{student_context}

Sign-off name: {user_name}
OUTREACH GOAL: {goal}
TONE: {tone}

LENGTH & STYLE (strict):
- Total body: **{min_words}–{max_words} words** (count every word; aim near **{word_count}**).
- **Crisp**: mostly short–medium sentences; cut filler ("I am writing to express", "I hope this email finds you well").
- **Informative**: reader learns who you are, what you want ({timeline_hint}), why **their** work matters, and what
  concrete preparation you bring (project, methods, or results — from STUDENT CONTEXT).
- **Professional**: no emojis, no bullet characters, no pasted website menus.

STRUCTURE after the salutation line:
1) **Introduction** (2–3 sentences): name, program/institution, optional one credibility (CGPA or rank/award from context).
2) **Their work** (2–3 sentences): reference **at least one** publication title OR a specific research thread from above;
   say clearly why it interests you or aligns with your preparation.
3) **Your fit** (2–4 sentences): one or two concrete highlights (projects, tools, coursework) tied to their direction —
   not a list of buzzwords.
4) **Ask & close** (2 sentences): specific ask (internship for {timeline_hint}, or brief call); thank them.

Rules: At most **two** numeric credentials; no false claims; first line must be exactly {salutation}

Return ONLY JSON:
{{
  "subject": "<specific; method/topic + internship inquiry>",
  "body": "<plain text; line1 = salutation>",
  "personalization_notes": "<one line>"
}}
"""

VARIANT_B_EMAIL_PROMPT = """
You are an expert writer of **research internship** emails. Produce a **concise** alternative to the same outreach.

PROFESSOR DETAILS:
- Full name (context only): {prof_name}
- Institution: {institution}
- Department: {department}
- Research topics (sanitized): {research_areas}
- Recent Publications: {publications}
- Lab/Profile Page: {lab_url}

LINE 1 of "body" (copy EXACTLY — then blank line before paragraph 2):
{salutation}

STUDENT CONTEXT (facts only):
{student_context}

Sign-off name: {user_name}
OUTREACH GOAL: {goal}
TONE: concise and direct (no fluff)

LENGTH (strict):
- Total body: **{min_words}–{max_words} words** (shorter than the primary draft).

STRUCTURE: same as primary (intro, their work with one publication reference, your fit, ask).

Return ONLY JSON:
{{
  "subject": "<distinct angle; still specific; under 58 chars>",
  "body": "<plain text; line1 = salutation>",
  "personalization_notes": "<one line>"
}}
"""

CRITIQUE_PROMPT = """
You are editing a research internship email for Professor {prof_name}.

IMMUTABLE first line (character-for-character):
{salutation}

DRAFT:
\"\"\"
{body}
\"\"\"

Revise so that:
1. First line remains EXACTLY: {salutation}
2. Length is **{min_words}–{max_words} words** (if too short, add substantive sentences using only implied facts
   from the draft; if too long, tighten without losing the paper/topic reference and the student's concrete fit).
3. **Crisp**: remove redundancy and vague phrases; strong verbs; one clear ask.
4. **Informative**: must still name their work specifically and your relevant preparation specifically.
5. No bullets; no menu junk; at most two numeric brags.

Return ONLY the full email plain text. No JSON.
"""

LENGTH_EXPAND_PROMPT = """
The research internship email below is too short. Expand it to **{min_words}–{max_words} words**.

FIRST LINE MUST STAY EXACTLY:
{salutation}

Add substance only using facts from:
PROFESSOR (name, institution): {prof_name} — {institution}
PUBLICATIONS / TOPICS: {pubs_brief}
STUDENT SNIPPET: {student_snippet}

Rules: no new invented credentials; no bullets; professional tone; keep the publication/topic tie and the internship ask.

CURRENT TEXT:
\"\"\"
{body}
\"\"\"

Return ONLY the full expanded email plain text.
"""

SUBJECT_REFINEMENT_PROMPT = """
The following subject line was generated for a cold email to Professor {prof_name}:
"{subject}"

Make it:
- More specific and eye-catching
- Under 60 characters
- Reference the professor's work or the student's background
- Not generic or spammy

Return ONLY the improved subject line, no quotes.
"""


FOLLOWUP_EMAIL_PROMPT = """
Write a brief, professional follow-up email. The original email was sent to Professor {prof_name} at {institution}.
The topic was: {goal}. It has been {days} days with no reply.

Keep it:
- Under 80 words
- Polite and non-pushy
- Reference the original email
- End with a clear question

Return a JSON: {{"subject": "...", "body": "..."}}
"""


# ── Email Generation Agent ────────────────────────────────────────────────────

class EmailAgent:
    """
    Generates, refines, and manages personalized cold emails for each professor.
    """

    def __init__(self):
        self.llm = config.get_groq_llm()

    def _call_llm(self, prompt: str, *, label: str = "email") -> str:
        try:
            t0 = time.perf_counter()
            response = self.llm.invoke(prompt)
            dt = time.perf_counter() - t0
            text = (response.content or "").strip()
            log_llm_call(
                label=label,
                prompt_chars=len(prompt),
                response_chars=len(text),
                latency_sec=dt,
                model=config.GROQ_MODEL,
            )
            return text
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            return ""

    def _parse_json(self, text: str) -> dict:
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

    def _expand_for_length(
        self,
        body: str,
        *,
        professor: ProfessorProfile,
        salutation: str,
        intent: UserIntent,
        kw_clean: list[str],
        min_w: int,
        max_w: int,
    ) -> str:
        pubs_parts: list[str] = []
        for p in (professor.recent_publications or [])[:3]:
            t = (p.title or "").strip()
            if t:
                pubs_parts.append(t[:160])
        pubs_brief = " | ".join(pubs_parts) if pubs_parts else ", ".join(kw_clean[:5]) or "recent lab directions"
        ctx = build_email_internship_context(intent)
        student_snippet = ctx[:950] + ("…" if len(ctx) > 950 else "")
        prompt = LENGTH_EXPAND_PROMPT.format(
            min_words=min_w,
            max_words=max_w,
            salutation=salutation,
            prof_name=professor.name,
            institution=professor.institution or "their institution",
            pubs_brief=pubs_brief[:1300],
            student_snippet=student_snippet,
            body=body,
        )
        expanded = self._call_llm(prompt, label="email_expand_length")
        return _force_opening_salutation((expanded or "").strip(), salutation)

    # ── Email Generation ──────────────────────────────────────────────────────

    def generate_email(
        self,
        professor: ProfessorProfile,
        intent: UserIntent,
        plan: ExecutionPlan,
    ) -> DraftEmail:
        """Generate a personalized cold email for a professor."""

        target_w, min_w, max_w = _email_targets(plan.email_strategy.word_count_target)

        # Format publications for prompt
        pubs_text = "No recent publications found."
        if professor.recent_publications:
            pub_lines = []
            for p in professor.recent_publications[:3]:  # Top 3 only
                line = f"- \"{p.title}\""
                if p.year:
                    line += f" ({p.year})"
                if p.venue:
                    line += f" — {p.venue}"
                pub_lines.append(line)
            pubs_text = "\n".join(pub_lines)

        salutation = _salutation_for_professor(professor.name)
        kw_clean = _sanitize_keywords(professor.research_keywords)
        research_areas_str = (
            ", ".join(kw_clean)
            if kw_clean
            else "(No clean topic list — rely on publication titles above only.)"
        )

        prompt = EMAIL_GENERATION_PROMPT.format(
            prof_name=professor.name,
            institution=professor.institution,
            department=professor.department or "Computer Science",
            research_areas=research_areas_str,
            publications=pubs_text,
            lab_url=professor.lab_url or professor.profile_url or "N/A",
            salutation=salutation,
            student_context=build_email_internship_context(intent),
            user_name=intent.user_name or "Student",
            goal=f"{intent.goal.value.replace('_', ' ').title()} — {intent.timeline or 'flexible timeline'}",
            tone=plan.email_strategy.tone,
            word_count=target_w,
            min_words=min_w,
            max_words=max_w,
            timeline_hint=intent.timeline or "the period I mentioned",
        )

        def _qc_fail(b: str) -> bool:
            return not b or _email_body_failed_qc(b, professor.name, min_words=min_w)

        raw = self._call_llm(prompt, label="email_generate_primary")
        data = self._parse_json(raw)

        subject = data.get("subject", f"Research Internship Inquiry — {professor.name}")
        body = _force_opening_salutation((data.get("body") or "").strip(), salutation)
        notes = data.get("personalization_notes", "")

        if _qc_fail(body):
            if body:
                console.print(f"  [yellow]⚠ Email QC failed for {professor.name}; retrying once...[/yellow]")
            raw = self._call_llm(prompt, label="email_generate_retry")
            data = self._parse_json(raw)
            body = _force_opening_salutation((data.get("body") or "").strip(), salutation)
            if data.get("subject"):
                subject = data["subject"]
            if data.get("personalization_notes"):
                notes = data["personalization_notes"]

        if _qc_fail(body) and body and not _email_has_garbage(body):
            console.print(f"  [yellow]⚠ Expanding short email for {professor.name}...[/yellow]")
            body = self._expand_for_length(
                body,
                professor=professor,
                salutation=salutation,
                intent=intent,
                kw_clean=kw_clean,
                min_w=min_w,
                max_w=max_w,
            )

        if _qc_fail(body):
            console.print(f"  [yellow]⚠ Using template fallback for {professor.name}.[/yellow]")
            body = self._fallback_email(professor, intent, salutation=salutation, kw_clean=kw_clean)
        else:
            c_prompt = CRITIQUE_PROMPT.format(
                prof_name=professor.name,
                salutation=salutation,
                body=body,
                min_words=min_w,
                max_words=max_w,
            )
            revised_body = self._call_llm(c_prompt, label="email_critique")
            revised_body = _force_opening_salutation(
                (revised_body or "").strip(), salutation
            )
            if (
                revised_body
                and not _email_has_garbage(revised_body)
                and _body_word_count(revised_body) >= min_w
                and "{" not in revised_body[:20]
            ):
                body = revised_body
                notes += " (Critiqued & Revised by LLM Reviewer)"
            elif revised_body and _qc_fail(revised_body):
                notes += " (Critique skipped — would break quality)"

        variants: list[DraftVariant] = []
        want_variants = (
            plan.email_strategy.generate_variants or config.DEFAULT_GENERATE_EMAIL_VARIANTS
        )
        if want_variants:
            t_lo = max(90, min_w - 40)
            t_hi = max(130, min_w)
            b_prompt = VARIANT_B_EMAIL_PROMPT.format(
                prof_name=professor.name,
                institution=professor.institution,
                department=professor.department or "Computer Science",
                research_areas=research_areas_str,
                publications=pubs_text,
                lab_url=professor.lab_url or professor.profile_url or "N/A",
                salutation=salutation,
                student_context=build_email_internship_context(intent),
                user_name=intent.user_name or "Student",
                goal=f"{intent.goal.value.replace('_', ' ').title()} — {intent.timeline or 'flexible timeline'}",
                min_words=t_lo,
                max_words=t_hi,
            )
            braw = self._call_llm(b_prompt, label="email_variant_concise")
            bdata = self._parse_json(braw)
            b_sub = bdata.get("subject") or f"Internship inquiry — {professor.name}"
            b_body = _force_opening_salutation((bdata.get("body") or "").strip(), salutation)
            if not b_body or _email_body_failed_qc(b_body, professor.name, min_words=t_lo):
                b_body = body
                b_sub = subject
            variants = [
                DraftVariant(label="A", subject=subject, body=body),
                DraftVariant(label="B", subject=b_sub, body=b_body),
            ]

        return DraftEmail(
            professor_id=professor.id,
            recipient_name=professor.name,
            recipient_email=professor.email or "",
            subject=subject,
            body=body,
            personalization_notes=notes,
            status=EmailStatus.DRAFT,
            variants=variants,
            selected_variant_index=0,
        )

    def _fallback_email(
        self,
        professor: ProfessorProfile,
        intent: UserIntent,
        *,
        salutation: Optional[str] = None,
        kw_clean: Optional[list[str]] = None,
    ) -> str:
        """Professional template if the LLM fails QC; never uses nav junk as a name or topic dump."""
        open_line = salutation or _salutation_for_professor(professor.name)
        name = intent.user_name or "The Applicant"

        hx = highlights_for_intent(intent)
        research_line = ""
        if hx:
            # One short, sentence-safe highlight
            research_line = hx[0].strip()
            if research_line and not research_line.endswith((".", "!", "?")):
                research_line += "."
        elif intent.user_profile_summary:
            ps = intent.user_profile_summary.strip()
            chunk = ps[:320]
            if "." in chunk:
                research_line = chunk.rsplit(".", 1)[0].strip() + "."
            else:
                research_line = (chunk[:240] + "…") if len(ps) > 240 else ps
                if not research_line.endswith((".", "!", "?", "…")):
                    research_line += "."
        else:
            research_line = "I have been building research-oriented projects in computing and would like to contribute to your group."

        inst_bits: list[str] = []
        if intent.cv_degree and intent.cv_institution:
            bit = f"a {intent.cv_degree} student at {intent.cv_institution}"
            if intent.cv_cgpa:
                bit += f" (CGPA {intent.cv_cgpa})"
            inst_bits.append(bit)
        elif intent.cv_institution:
            inst_bits.append(f"a student at {intent.cv_institution}")

        intro = f"I am {name}"
        if inst_bits:
            intro += f", {inst_bits[0]}"
        intro += ". "

        goal = intent.goal.value.replace("_", " ")
        when = intent.timeline or "an upcoming term"
        inst = professor.institution or "your institution"

        pub_para = ""
        if professor.recent_publications:
            t0 = professor.recent_publications[0].title.strip()
            t0_short = t0[:118] + "…" if len(t0) > 120 else t0
            pub_para = (
                f'I recently read your work, including "{t0_short}", and found the problems you address '
                f"closely aligned with topics I have been studying. "
            )
            if len(professor.recent_publications) > 1:
                t1 = professor.recent_publications[1].title.strip()
                t1_short = t1[:100] + "…" if len(t1) > 105 else t1
                pub_para += f'I also looked at "{t1_short}" as further context on your group\'s direction. '

        topics = kw_clean if kw_clean is not None else _sanitize_keywords(professor.research_keywords)
        if topics:
            topic_phrase = ", ".join(topics[:3])
            fit = (
                f"I believe the skills I have been developing could be relevant to projects in {topic_phrase}, "
                f"and I would welcome the chance to learn from your group's approach."
            )
        else:
            fit = (
                "I believe my background could complement the directions suggested by your recent publications, "
                "and I would value the opportunity to contribute where it would be useful."
            )

        return (
            f"{open_line}\n\n"
            f"{intro}"
            f"I am writing to ask whether you might consider hosting a {goal} at {inst} during {when}. "
            f"I have followed your group's research and would be grateful for the chance to work in an environment "
            f"that pushes the state of the art in your area.\n\n"
            f"{pub_para}\n\n"
            f"{research_line} {fit}\n\n"
            f"If you have openings for interns in that timeframe, I would appreciate guidance on next steps. "
            f"If your group is not recruiting then, I would still be grateful for any suggestion of a colleague "
            f"whose work might be a better match.\n\n"
            f"Thank you for considering my inquiry.\n\n"
            f"Best regards,\n{name}"
        )

    def generate_batch(
        self,
        professors: list[ProfessorProfile],
        intent: UserIntent,
        plan: ExecutionPlan,
    ) -> list[DraftEmail]:
        """Generate emails for all shortlisted professors."""
        emails: list[DraftEmail] = []

        console.print(f"\n[bold cyan]✉ Generating {len(professors)} personalized emails...[/bold cyan]\n")

        for i, prof in enumerate(professors, 1):
            console.print(
                f"  [dim]({i}/{len(professors)})[/dim] "
                f"[bold]{prof.name}[/bold] — {prof.institution}"
            )
            email = self.generate_email(prof, intent, plan)
            emails.append(email)
            console.print(f"  [green]✓[/green] Subject: [italic]{email.subject}[/italic]")

        console.print(f"\n[bold green]✓ Generated {len(emails)} draft emails.[/bold green]")
        return emails

    # ── Follow-up Email ───────────────────────────────────────────────────────

    def generate_followup(
        self,
        original_email: DraftEmail,
        professor: ProfessorProfile,
        intent: UserIntent,
        days_elapsed: int,
    ) -> DraftEmail:
        """Generate a follow-up email for a previously sent email."""
        prompt = FOLLOWUP_EMAIL_PROMPT.format(
            prof_name=professor.name,
            institution=professor.institution,
            goal=intent.goal.value.replace("_", " "),
            days=days_elapsed,
        )
        raw = self._call_llm(prompt, label="email_followup")
        data = self._parse_json(raw)

        return DraftEmail(
            professor_id=professor.id,
            recipient_name=professor.name,
            recipient_email=professor.email or "",
            subject=data.get("subject", f"Re: {original_email.subject}"),
            body=data.get("body", "Following up on my previous email."),
            personalization_notes="Follow-up email",
            status=EmailStatus.DRAFT,
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def display_email(self, email: DraftEmail, professor: Optional[ProfessorProfile] = None) -> None:
        """Display a draft email in a rich panel."""
        title = f"[bold]Draft Email → {email.recipient_name}[/bold]"
        if professor:
            title += f" | [dim]{professor.institution}[/dim]"

        content = (
            f"[bold]To:[/bold] {email.recipient_email or '(email not found)'}\n"
            f"[bold]Subject:[/bold] {email.subject}\n"
            f"[dim]─────────────────────────────────────[/dim]\n"
            f"{email.body}\n"
            f"[dim]─────────────────────────────────────[/dim]\n"
            f"[italic dim]Note: {email.personalization_notes}[/italic dim]"
        )
        console.print(Panel(content, title=title, border_style="blue"))


# ── LangGraph Node ────────────────────────────────────────────────────────────

def email_generation_node(state: AgentState) -> AgentState:
    """LangGraph node: shortlisted professors → draft emails."""
    intent = UserIntent.model_validate(state["intent"])
    plan = ExecutionPlan.model_validate(state["plan"])
    shortlisted = [ProfessorProfile.model_validate(p) for p in state.get("shortlisted", [])]

    agent = EmailAgent()
    emails = agent.generate_batch(shortlisted, intent, plan)

    messages = state.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": (
            f"I've drafted {len(emails)} personalized emails. "
            "Please review them and approve, edit, or reject each one."
        ),
    })

    return {
        **state,
        "draft_emails": [e.model_dump() for e in emails],
        "messages": messages,
        "mode": "review",
    }

"""
core/cv_parser.py

CV / Resume PDF parser.

Extracts raw text from a PDF and uses Gemini to distill a clean
professional profile summary suitable for inclusion in cold emails.
"""

from __future__ import annotations

import os
from typing import Optional

from rich.console import Console
import config

console = Console()

PROFILE_EXTRACT_PROMPT = """
You are reading a student's CV/resume for an academic **research internship** outreach pipeline.

Extract structured facts, then curate aggressively — do not treat the CV as text to dump into emails.

A) Standard fields:
1. Full name
2. Current degree/program and institution (as written)
3. institution_tag if evident (IIT, NIT, IIIT, IISc, Ivy, etc.) — null if unclear
4. cgpa / GPA with scale (e.g. "8.13/10 (till 4th sem)") — null if absent
5. academic_rank: dept rank, dean's list, medal — null if absent
6. research_interests_from_cv: specific areas inferred from projects/courses (3-8 short phrases)
7. research_experience: RAships, thesis, lab work, formal research programs (0-4 short lines; empty if none)
8. work_research_adjacent: 0-3 lines ONLY for jobs/projects with clear research/engineering depth
   (e.g. NLP agents, formal verification, ML in production). Skip generic web CRUD unless it's the only technical work.
9. key_projects: 2-5 one-line technical project summaries (methods + outcome when possible)
10. publications: titles if any
11. skills: top tools/methods (max 12 strings)
12. achievements: raw list of competitions, exam ranks, hackathons (2-8) — used internally

B) research_internship_highlights — **mandatory**: exactly 4 to 7 bullets. Each bullet max 22 words.
   These are the ONLY resume lines meant for cold emails. Choose by priority:
   (1) Publications / preprints / formal research / thesis-like work
   (2) Deep technical projects tied to methods (ML, systems, theory, verification, etc.)
   (3) Research-adjacent employment with concrete technical outcomes
   (4) At most ONE bullet for national entrance exam rank (JEE/GRE/SAT etc.) IF truly strong
   (5) At most ONE bullet for hackathon/CP ONLY if it shows research-relevant depth (e.g. ML competition), not generic listing
   EXCLUDE from highlights: fest coordination, generic club roles, long skill dumps, unrelated volunteering,
   minor admin web apps unless they directly match research software.

C) profile_summary: 3-4 sentences, professional, for faculty readers. Mention degree + institution + 1-2 strongest
   research-relevant signals only. Do not list every hackathon or job.

CV TEXT:
\"\"\"
{cv_text}
\"\"\"

Return ONLY this JSON:
{{
  "name": "...",
  "degree": "...",
  "institution": "...",
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
  "research_internship_highlights": [],
  "profile_summary": "..."
}}
"""


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract all text from a PDF using PyMuPDF (fitz).
    Returns concatenated text or None if parsing fails.
    """
    if not os.path.isfile(pdf_path):
        console.print(f"[red]File not found: {pdf_path}[/red]")
        return None

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        full_text = "\n".join(pages_text)
        return full_text.strip()
    except ImportError:
        console.print("[yellow]⚠ PyMuPDF not installed. Install with: pip install PyMuPDF[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error reading PDF: {e}[/red]")
        return None


class CVParser:
    """
    Parses a CV PDF and extracts a structured profile summary via Gemini.
    """

    def __init__(self):
        self.llm = config.get_groq_llm()

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            return ""

    def _parse_json(self, text: str) -> dict:
        import re, json
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

    def parse(self, pdf_path: str) -> dict:
        """
        Full pipeline: PDF → text → Gemini → structured profile dict.

        Returns dict including research_internship_highlights, research_interests_from_cv,
        research_experience, work_research_adjacent, plus standard profile fields.
        """
        console.print(f"[dim]📄 Parsing CV: {pdf_path}[/dim]")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            return {
                "name": None,
                "profile_summary": "CV could not be parsed.",
            }

        # Limit text to ~3000 chars for token budget
        cv_text = raw_text[:3000]

        prompt = PROFILE_EXTRACT_PROMPT.format(cv_text=cv_text)
        raw = self._call_llm(prompt)
        data = self._parse_json(raw)

        if data:
            console.print("[green]✓ CV parsed successfully.[/green]")
            return data
        else:
            # Fallback: just return raw text without structure
            console.print("[yellow]⚠ Could not extract structured data from CV. Using raw text.[/yellow]")
            return {
                "name": None,
                "profile_summary": raw_text[:600],
            }

    def get_profile_summary(self, pdf_path: str) -> str:
        """
        Quick helper: parse CV and return just the profile_summary string.
        """
        data = self.parse(pdf_path)
        return data.get("profile_summary", "")

    def get_research_interests(self, pdf_path: str) -> list[str]:
        """Parse CV and return list of research interests."""
        data = self.parse(pdf_path)
        return data.get("research_interests", [])

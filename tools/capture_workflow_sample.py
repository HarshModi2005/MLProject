#!/usr/bin/env python3
"""
Drive the supervised pipeline with scripted Rich prompts and capture plain-text log.
Used to document a sample run in the technical report. Run from repo root:

  MAX_CRAWL_URLS=5 TOP_K_PROFESSORS=2 python3 tools/capture_workflow_sample.py

Requires GROQ_API_KEY and search keys as in normal .env.
"""
from __future__ import annotations

import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path

# Tight limits for a short demo (override .env defaults)
os.environ.setdefault("MAX_CRAWL_URLS", "5")
os.environ.setdefault("TOP_K_PROFESSORS", "2")
os.environ.setdefault("SEARCH_MAX_RESULTS_PER_QUERY", "2")
os.environ.setdefault("SEMANTIC_SCHOLAR_AUTHORS_PER_KEYWORD", "2")
os.environ.setdefault("SEMANTIC_SCHOLAR_KEYWORD_COUNT", "2")
os.environ.setdefault("CRAWL_SNOWBALL", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def strip_rich_markup(text: str) -> str:
    return re.sub(r"\[/?.+?\]", "", text)


GOAL_TEXT = (
    "I want to cold email professors for a Summer 2026 research internship in "
    "machine learning and natural language processing, targeting the United States. "
    "My name is Alex Rivera. I am a second-year MS student in Computer Science with "
    "experience in Python and PyTorch; I completed a project on fine-tuning transformers "
    "for text classification and I am looking for a research lab alignment."
)


class _PromptState:
    seen_confirm_your_answer = False
    slot_empty_idx = 0


_SLOT_EMPTY_REPLIES = [
    "machine learning, natural language processing",
    "United States",
    "Summer 2026",
    "Alex Rivera",
    "MS CS student; Python, PyTorch; NLP coursework and a transformer fine-tuning project.",
]


def scripted_prompt_ask(*args, **kwargs):
    raw = "".join(str(a) for a in args)
    plain = strip_rich_markup(raw).lower()

    if "found previous session" in plain and "resume" in plain:
        return "n"
    if "what do you want to do" in plain:
        return GOAL_TEXT
    if "your answer" in plain:
        _PromptState.seen_confirm_your_answer = True
        return "yes"
    if "action" in plain and kwargs.get("choices"):
        return "skip"

    if not raw.strip():
        if _PromptState.seen_confirm_your_answer:
            return "yes"
        i = _PromptState.slot_empty_idx
        if i < len(_SLOT_EMPTY_REPLIES):
            _PromptState.slot_empty_idx += 1
            return _SLOT_EMPTY_REPLIES[i]
        return "yes"

    if "what would you like to change" in plain:
        return "yes"
    return "yes"


def main() -> int:
    os.chdir(REPO_ROOT)
    session_file = REPO_ROOT / "session_state.json"
    if session_file.is_file():
        session_file.unlink()

    log_path = REPO_ROOT / "tools" / "workflow_sample_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    from unittest import mock

    import main as main_mod
    from db.database import init_db
    from rich.console import Console

    # Plain console for file (no ANSI)
    plain_file = log_path.open("w", encoding="utf-8")
    plain_console = Console(file=plain_file, width=100, color_system=None, highlight=False)

    init_db()

    console_patch_targets = (
        "main.console",
        "agents.intent_agent.console",
        "agents.planning_agent.console",
        "agents.crawler_agent.console",
        "agents.email_agent.console",
        "core.relevance_scorer.console",
        "crawler.search_api.console",
        "crawler.faculty_scraper.console",
        "crawler.profile_extractor.console",
        "crawler.scholar_scraper.console",
        "core.cv_parser.console",
    )

    def _plan_run_cli_allow_missing_email(real_run_cli):
        """Wrap planning CLI so demo crawl can score/draft even if scrape finds no emails."""

        def _wrapped(self, intent):
            plan = real_run_cli(self, intent)
            plan.filtering_criteria.require_email = False
            return plan

        return _wrapped

    from agents.planning_agent import PlanningAgent
    from agents.email_agent import EmailAgent
    from core.relevance_scorer import RelevanceScorer

    real_plan_cli = PlanningAgent.run_cli

    def _shortlist_skip_geo_prefilter():
        """
        Demo log only: scraped pages often lack 'United States' in free text, so the
        strict geographic pre-filter can drop everyone. Full scoring still uses geo weights.
        """

        def _wrapped(self, professors, intent, plan, top_k=None):
            ranked = self.score_and_rank(professors, intent, plan)
            min_score = plan.filtering_criteria.min_relevance_score
            filtered = [p for p in ranked if p.relevance_score >= min_score]
            k = top_k or plan.top_k_professors
            shortlisted = filtered[:k]
            self._display_shortlist(shortlisted)
            return shortlisted

        return _wrapped

    def _batch_fill_placeholder_email(real_batch):
        """Let the review loop run so the log shows drafted bodies (no real address on scrape)."""

        def _wrapped(self, shortlisted, intent, plan):
            drafts = real_batch(self, shortlisted, intent, plan)
            for d in drafts:
                if not (d.recipient_email or "").strip():
                    d.recipient_email = "professor-email-not-on-page@example.invalid"
            return drafts

        return _wrapped

    real_email_batch = EmailAgent.generate_batch

    with mock.patch("rich.prompt.Prompt.ask", side_effect=scripted_prompt_ask):
        with mock.patch.object(
            PlanningAgent,
            "run_cli",
            _plan_run_cli_allow_missing_email(real_plan_cli),
        ):
            with mock.patch.object(
                EmailAgent,
                "generate_batch",
                _batch_fill_placeholder_email(real_email_batch),
            ):
                with mock.patch.object(
                    RelevanceScorer,
                    "shortlist",
                    _shortlist_skip_geo_prefilter(),
                ):
                    with ExitStack() as stack:
                        for target in console_patch_targets:
                            stack.enter_context(mock.patch(target, plain_console))
                        session_id = "demo-session-capture"
                        main_mod._run_supervised_pipeline(session_id, {})

    plain_file.close()
    print(f"Wrote plain-text log to {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

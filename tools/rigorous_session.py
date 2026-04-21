#!/usr/bin/env python3
"""
Rigorous end-to-end session logger for the research outreach pipeline.

Logs EVERYTHING to an output directory (no truncation, no ellipses in artifacts):
  - user_interactions.jsonl   Each Rich Prompt.ask: full prompt text + your exact reply
  - console_transcript.txt    Full CLI-style transcript (no ANSI colors)
  - llm/                      Per-call numbered files: NNNN_meta.json, NNNN_input.txt, NNNN_output.txt
                              (errors -> NNNN_error.txt + traceback)
  - state/                    Full JSON snapshots: intent, plan, professors, shortlisted, drafts
  - MANIFEST.json             Run metadata, counts, duration

Modes
  * Interactive (default): you answer prompts in the terminal; each Q/A is logged verbatim.
  * Batch: --batch-file PATH — one answer per line (first non-empty line is first Prompt return value, etc.).
           Lines starting with # are ignored. Trailing empty lines ignored.
  * Replay: --replay-from user_interactions.jsonl — same as batch but takes the "reply" field from each
           JSON line of a previous rigorous run (full strings preserved).

Optional demo assists (--demo-assists): same relaxations as tools/capture_workflow_sample.py so thin
scrapes still produce scoring + drafts in logs. Without it, behavior matches strict main.py.

Examples
  python3 tools/rigorous_session.py --out run_logs/my_run --fresh-session

  python3 tools/rigorous_session.py --batch-file my_answers.txt --demo-assists --fresh-session

  # Replay the exact replies from a previous run (user_interactions.jsonl -> "reply" field order):
  python3 tools/rigorous_session.py --replay-from run_logs/rigorous_xxx/user_interactions.jsonl --demo-assists

Environment: same as main.py (GROQ_API_KEY, search keys, etc.). Override crawl limits for faster runs:
  MAX_CRAWL_URLS=5 TOP_K_PROFESSORS=2 python3 tools/rigorous_session.py ...
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
import traceback
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# LLM invoke logging (all ChatGroq.invoke calls in the process)
# ---------------------------------------------------------------------------

_llm_counter = 0


def _serialize_lc_input(inp: Any) -> str:
    if inp is None:
        return ""
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        parts: list[str] = []
        for i, m in enumerate(inp):
            cls = type(m).__name__
            content = getattr(m, "content", None)
            if isinstance(content, list):
                parts.append(f"[{i}] {cls}: {json.dumps(content, ensure_ascii=False)}")
            else:
                parts.append(f"[{i}] {cls}: {content if content is not None else repr(m)}")
        return "\n".join(parts)
    content = getattr(inp, "content", None)
    if content is not None:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, indent=2)
    return repr(inp)


def _serialize_lc_output(out: Any) -> str:
    if out is None:
        return ""
    c = getattr(out, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return json.dumps(c, ensure_ascii=False, indent=2)
    return str(out)


def install_groq_invoke_logger(llm_root: Path) -> None:
    """Patch ChatGroq.invoke once; writes one folder per process run."""
    from langchain_groq import ChatGroq

    llm_root.mkdir(parents=True, exist_ok=True)
    _orig = ChatGroq.invoke

    @functools.wraps(_orig)
    def _logged_invoke(self, input, config=None, *, stop=None, **kwargs):
        global _llm_counter
        _llm_counter += 1
        n = _llm_counter
        meta = {
            "index": n,
            "model": getattr(self, "model_name", None) or getattr(self, "model", None),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (llm_root / f"{n:05d}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        inp_text = _serialize_lc_input(input)
        (llm_root / f"{n:05d}_input.txt").write_text(inp_text, encoding="utf-8")

        try:
            result = _orig(self, input, config=config, stop=stop, **kwargs)
            (llm_root / f"{n:05d}_output.txt").write_text(
                _serialize_lc_output(result), encoding="utf-8"
            )
            return result
        except Exception:
            (llm_root / f"{n:05d}_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            raise

    ChatGroq.invoke = _logged_invoke  # type: ignore[method-assign]


def strip_rich_markup(s: str) -> str:
    return re.sub(r"\[/?.+?\]", "", str(s))


# ---------------------------------------------------------------------------
# Prompt.ask wrapper
# ---------------------------------------------------------------------------

def build_prompt_wrapper(
    original_ask: Callable,
    interactions_path: Path,
    batch_queue: Optional[deque[str]],
) -> Callable:
    interactions_path.parent.mkdir(parents=True, exist_ok=True)
    fh = interactions_path.open("a", encoding="utf-8")
    seq = [0]

    def _wrapped(*args, **kwargs):
        seq[0] += 1
        raw_prompt = "".join(str(a) for a in args)
        plain = strip_rich_markup(raw_prompt)
        choices = kwargs.get("choices")
        default = kwargs.get("default")

        if batch_queue is not None:
            if not batch_queue:
                fh.flush()
                raise RuntimeError(
                    "Batch answer queue exhausted.\n"
                    f"Need another line for prompt #{seq[0]}:\n{plain}\n"
                    "Add lines to your batch file in strict call order."
                )
            reply = batch_queue.popleft()
        else:
            reply = original_ask(*args, **kwargs)

        record = {
            "sequence": seq[0],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_raw": raw_prompt,
            "prompt_plain": plain,
            "choices": list(choices) if choices else None,
            "default": default,
            "reply": reply,
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        return reply

    return _wrapped


# ---------------------------------------------------------------------------
# Console patching (single plain transcript)
# ---------------------------------------------------------------------------

def console_patch_targets() -> tuple[str, ...]:
    return (
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
        "db.database.console",
    )


# ---------------------------------------------------------------------------
# Optional demo assists (match capture_workflow_sample)
# ---------------------------------------------------------------------------

def _plan_run_cli_allow_missing_email(real_run_cli):
    def _wrapped(self, intent):
        plan = real_run_cli(self, intent)
        plan.filtering_criteria.require_email = False
        return plan

    return _wrapped


def _batch_fill_placeholder_email(real_batch):
    def _wrapped(self, shortlisted, intent, plan):
        drafts = real_batch(self, shortlisted, intent, plan)
        for d in drafts:
            if not (d.recipient_email or "").strip():
                d.recipient_email = "professor-email-not-on-page@example.invalid"
        return drafts

    return _wrapped


def _shortlist_skip_geo_prefilter():
    def _wrapped(self, professors, intent, plan, top_k=None):
        ranked = self.score_and_rank(professors, intent, plan)
        min_score = plan.filtering_criteria.min_relevance_score
        filtered = [p for p in ranked if p.relevance_score >= min_score]
        k = top_k or plan.top_k_professors
        shortlisted = filtered[:k]
        self._display_shortlist(shortlisted)
        return shortlisted

    return _wrapped


# ---------------------------------------------------------------------------
# Pipeline with full state dumps
# ---------------------------------------------------------------------------

def run_pipeline_with_state_dumps(
    out_dir: Path,
    session_id: str,
    state: dict,
    plain_console,
    demo_assists: bool,
) -> None:
    from rich.prompt import Prompt
    from db.database import init_db, upsert_session_state
    from db import database as db
    from orchestrator.state import UserIntent, ExecutionPlan

    import main as main_mod
    from agents.intent_agent import IntentAgent
    from agents.planning_agent import PlanningAgent
    from agents.crawler_agent import CrawlerAgent
    from core.relevance_scorer import RelevanceScorer
    from agents.email_agent import EmailAgent

    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, obj: Any) -> None:
        path = state_dir / name
        if hasattr(obj, "model_dump"):
            data = obj.model_dump(mode="json")
        else:
            data = obj
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    init_db()

    plain_console.print("\n[bold cyan]═══ Step 1/5: Intent Clarification ═══[/bold cyan]")
    intent_agent = IntentAgent()
    if state and state.get("intent"):
        intent_agent.intent = UserIntent.model_validate(state["intent"])
        plain_console.print("[green]Restored previous intent.[/green]")
        intent = intent_agent.intent
    else:
        intent = intent_agent.run_cli()
        main_mod.save_state(intent=intent)
    _dump("01_intent.json", intent)

    plain_console.print("\n[bold cyan]═══ Step 2/5: Execution Planning ═══[/bold cyan]")
    planning_agent = PlanningAgent()
    if state and state.get("plan"):
        plan = ExecutionPlan.model_validate(state["plan"])
        plain_console.print("[green]Restored previous execution plan.[/green]")
    else:
        plan = planning_agent.run_cli(intent)
        main_mod.save_state(plan=plan)
    _dump("02_plan.json", plan)

    upsert_session_state(session_id, intent, plan)

    plain_console.print("\n[bold cyan]═══ Step 3/5: Crawling for Professors ═══[/bold cyan]")
    crawler = CrawlerAgent()
    professors = crawler.run(plan)
    _dump(
        "03_professors.json",
        [p.model_dump(mode="json") for p in professors],
    )

    if not professors:
        plain_console.print("[yellow]⚠ No professors found.[/yellow]")
        _dump("04_shortlisted.json", [])
        _dump("05_drafts.json", [])
        return

    plain_console.print(
        f"\n[bold cyan]═══ Step 4/5: Scoring & Shortlisting ({len(professors)} candidates) ═══[/bold cyan]"
    )
    scorer = RelevanceScorer()
    shortlisted = scorer.shortlist(professors, intent, plan)
    _dump(
        "04_shortlisted.json",
        [p.model_dump(mode="json") for p in shortlisted],
    )

    if not shortlisted:
        plain_console.print("[yellow]⚠ No professors met the relevance threshold.[/yellow]")
        _dump("05_drafts.json", [])
        return

    plain_console.print(f"[green]✓ {len(shortlisted)} professors shortlisted.[/green]")

    plain_console.print(f"\n[bold cyan]═══ Step 5/5: Generating & Reviewing Emails ═══[/bold cyan]")
    email_agent = EmailAgent()
    drafts = email_agent.generate_batch(shortlisted, intent, plan)
    _dump(
        "05_drafts.json",
        [d.model_dump(mode="json") for d in drafts],
    )

    approved_count = 0
    skipped_count = 0
    for i, (draft, prof) in enumerate(zip(drafts, shortlisted), 1):
        plain_console.print(f"\n[bold]─── Email {i}/{len(drafts)} ───[/bold]")
        email_agent.display_email(draft, prof)
        if not draft.recipient_email:
            plain_console.print("[yellow]⚠ No email address found. Skipping.[/yellow]")
            skipped_count += 1
            continue
        while True:
            action = Prompt.ask(
                "\n[bold yellow]Action[/bold yellow]",
                choices=["approve", "skip", "quit"],
                default="approve",
            )
            if action == "approve":
                db.save_draft_email(draft, session_id)
                plain_console.print("[green]✓ Draft saved to database.[/green]")
                approved_count += 1
                break
            if action == "skip":
                skipped_count += 1
                break
            plain_console.print("[yellow]Exiting review early.[/yellow]")
            return

    plain_console.print(
        f"\n[bold green]✓ Review complete![/bold green] "
        f"{approved_count} approved, {skipped_count} skipped."
    )


def load_batch_file(path: Path) -> deque[str]:
    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        s = line.rstrip("\n")
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        lines.append(s)
    return deque(lines)


def load_replies_from_interactions_jsonl(path: Path) -> deque[str]:
    """Replay prior session: one reply per line object, in file order (full strings, no truncation)."""
    q: deque[str] = deque()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        r = rec.get("reply")
        if r is None:
            continue
        if not isinstance(r, str):
            r = json.dumps(r, ensure_ascii=False)
        q.append(r)
    return q


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run outreach pipeline with full logging (prompts, LLM I/O, state JSON)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: run_logs/rigorous_<UTC timestamp>)",
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=None,
        help="Non-interactive: one answer per line for each Prompt.ask in order (# comments ok)",
    )
    parser.add_argument(
        "--replay-from",
        type=Path,
        default=None,
        help="Non-interactive: reuse the 'reply' field from each line of a prior user_interactions.jsonl",
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help="Delete session_state.json before run",
    )
    parser.add_argument(
        "--demo-assists",
        action="store_true",
        help="Relax require_email, geo prefilter, placeholder To (documentation-style full trace only)",
    )
    parser.add_argument(
        "--session-id",
        default="rigorous-session",
        help="Session id for DB rows",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or (REPO_ROOT / "run_logs" / f"rigorous_{ts}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)
    if args.fresh_session:
        sf = REPO_ROOT / "session_state.json"
        if sf.is_file():
            sf.unlink()

    from rich.console import Console
    from rich.prompt import Prompt

    transcript_path = out_dir / "console_transcript.txt"
    transcript_fh = transcript_path.open("w", encoding="utf-8")
    plain_console = Console(
        file=transcript_fh,
        width=120,
        color_system=None,
        highlight=False,
        force_terminal=False,
    )

    interactions_path = out_dir / "user_interactions.jsonl"
    if interactions_path.exists():
        interactions_path.unlink()

    if args.batch_file and args.replay_from:
        print("Use only one of --batch-file or --replay-from", file=sys.stderr)
        return 2

    batch_queue: Optional[deque[str]] = None
    if args.batch_file:
        batch_queue = load_batch_file(args.batch_file.resolve())
    elif args.replay_from:
        batch_queue = load_replies_from_interactions_jsonl(args.replay_from.resolve())

    original_ask = Prompt.ask
    wrapped_ask = build_prompt_wrapper(original_ask, interactions_path, batch_queue)

    install_groq_invoke_logger(out_dir / "llm")

    import config as cfg

    missing = cfg.validate_config()
    if missing:
        print(f"Missing required env: {missing}", file=sys.stderr)
        return 1

    from db.database import init_db

    init_db()
    plain_console.print("[dim]Database initialized (rigorous session).[/dim]\n")

    import main as main_mod

    main_mod.console = plain_console

    patches_cm = ExitStack()
    for target in console_patch_targets():
        patches_cm.enter_context(mock.patch(target, plain_console))

    patches_cm.enter_context(mock.patch("rich.prompt.Prompt.ask", wrapped_ask))

    from agents.planning_agent import PlanningAgent
    from agents.email_agent import EmailAgent
    from core.relevance_scorer import RelevanceScorer

    if args.demo_assists:
        patches_cm.enter_context(
            mock.patch.object(
                PlanningAgent,
                "run_cli",
                _plan_run_cli_allow_missing_email(PlanningAgent.run_cli),
            )
        )
        patches_cm.enter_context(
            mock.patch.object(
                EmailAgent,
                "generate_batch",
                _batch_fill_placeholder_email(EmailAgent.generate_batch),
            )
        )
        patches_cm.enter_context(
            mock.patch.object(
                RelevanceScorer,
                "shortlist",
                _shortlist_skip_geo_prefilter(),
            )
        )

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "batch_file": str(args.batch_file) if args.batch_file else None,
        "replay_from": str(args.replay_from) if args.replay_from else None,
        "demo_assists": args.demo_assists,
        "fresh_session": args.fresh_session,
    }

    try:
        with patches_cm:
            run_pipeline_with_state_dumps(
                out_dir,
                args.session_id,
                {},
                plain_console,
                args.demo_assists,
            )
    finally:
        transcript_fh.close()

    duration = time.perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["duration_seconds"] = round(duration, 3)
    manifest["llm_invoke_calls"] = _llm_counter
    n_user = 0
    if interactions_path.is_file():
        n_user = sum(1 for _ in interactions_path.open(encoding="utf-8"))
    manifest["user_prompt_rounds"] = n_user
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Rigorous session complete. Artifacts: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

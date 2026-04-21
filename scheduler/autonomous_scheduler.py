"""
scheduler/autonomous_scheduler.py

APScheduler-based autonomous execution engine.

Jobs:
  1. re_crawl_job        — Weekly: crawl for new professors
  2. check_replies_job   — Every 6h: scan inbox for professor replies
  3. send_followups_job  — Daily: send follow-ups for unanswered emails

Requires the system to be in "autonomous" mode (user has approved transition).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

from rich.console import Console

from db import database as db
from scheduler.reply_monitor import ReplyMonitor
from scheduler.followup_engine import FollowupEngine

console = Console()
logger = logging.getLogger(__name__)


# ── Job Implementations ────────────────────────────────────────────────────────

def re_crawl_job(
    plan_json: str,
    intent_json: str,
    session_id: str,
) -> None:
    """
    Periodic re-crawl job.
    Discovers new professors matching the plan and adds them to the DB.
    Existing professors are skipped (duplicate detection via email).
    """
    import json
    from orchestrator.state import ExecutionPlan, UserIntent
    from agents.crawler_agent import CrawlerAgent
    from core.relevance_scorer import RelevanceScorer

    console.print(f"\n[bold cyan]🔄 Autonomous re-crawl started at {datetime.utcnow()}[/bold cyan]")

    plan = ExecutionPlan.model_validate(json.loads(plan_json))
    intent = UserIntent.model_validate(json.loads(intent_json))

    crawler = CrawlerAgent()
    scorer = RelevanceScorer()

    profiles = crawler.run(plan)
    shortlisted = scorer.shortlist(profiles, intent, plan)

    # Filter out already-contacted professors
    new_professors = [
        p for p in shortlisted
        if p.email and not db.email_already_sent(p.email)
    ]

    console.print(
        f"  [green]✓[/green] Found {len(new_professors)} new professors not yet contacted."
    )

    db.save_professors_batch(new_professors, session_id=session_id)
    logger.info(f"Re-crawl job completed. New professors found: {len(new_professors)}")


def check_replies_job(followup_days: int = 7, session_id: Optional[str] = None) -> None:
    """
    Periodic job to check Gmail inbox for replies from professors.
    Updates email status in DB when a reply is detected.
    """
    console.print(f"\n[bold cyan]📬 Checking for replies at {datetime.utcnow()}[/bold cyan]")
    monitor = ReplyMonitor()
    monitor.scan_and_update_replies()


def send_followups_job(followup_days: int = 7) -> None:
    """
    Daily job: send follow-up emails for emails sent > followup_days days
    ago with no reply.
    """
    console.print(f"\n[bold cyan]📤 Follow-up check at {datetime.utcnow()}[/bold cyan]")
    engine = FollowupEngine(followup_days=followup_days)
    engine.process_pending_followups()


# ── Scheduler Setup ───────────────────────────────────────────────────────────

class AutonomousScheduler:
    """
    Manages APScheduler jobs for the autonomous execution phase.
    """

    def __init__(
        self,
        plan_json: str,
        intent_json: str,
        session_id: str,
        followup_days: int = 7,
    ):
        self.plan_json = plan_json
        self.intent_json = intent_json
        self.session_id = session_id
        self.followup_days = followup_days
        self._scheduler = None

    def _get_scheduler(self):
        if self._scheduler is None:
            try:
                from apscheduler.schedulers.background import BackgroundScheduler
                self._scheduler = BackgroundScheduler()
            except ImportError:
                raise ImportError(
                    "APScheduler not installed. Run: pip install apscheduler"
                )
        return self._scheduler

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        scheduler = self._get_scheduler()

        # Weekly re-crawl
        scheduler.add_job(
            func=re_crawl_job,
            trigger="interval",
            weeks=1,
            id="re_crawl",
            name="Weekly Professor Re-Crawl",
            kwargs={
                "plan_json": self.plan_json,
                "intent_json": self.intent_json,
                "session_id": self.session_id,
            },
        )

        # Every 6h reply check
        scheduler.add_job(
            func=check_replies_job,
            trigger="interval",
            hours=6,
            id="check_replies",
            name="Reply Monitor",
            kwargs={"followup_days": self.followup_days, "session_id": self.session_id},
        )

        # Daily follow-up
        scheduler.add_job(
            func=send_followups_job,
            trigger="interval",
            days=1,
            id="send_followups",
            name="Follow-up Sender",
            kwargs={"followup_days": self.followup_days},
        )

        scheduler.start()
        self._scheduler = scheduler

        console.print(Panel_str := (
            "\n[bold green]🤖 Autonomous Mode Active[/bold green]\n"
            f"[dim]Re-crawl: weekly | Reply check: every 6h | Follow-ups: daily[/dim]\n"
            f"[dim]Session ID: {self.session_id}[/dim]"
        ))
        console.print(Panel_str)
        logger.info("Autonomous scheduler started.")

    def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            console.print("[yellow]Autonomous scheduler stopped.[/yellow]")

    def status(self) -> list[dict]:
        """Return status of all scheduled jobs."""
        if not self._scheduler:
            return []
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time),
            }
            for job in self._scheduler.get_jobs()
        ]

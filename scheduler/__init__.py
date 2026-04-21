"""
scheduler/__init__.py

Autonomous execution scheduler, duplicate detection, and automated follow-ups.
"""

from scheduler.autonomous_scheduler import start_scheduler, stop_scheduler
from scheduler.reply_monitor import check_replies_job
from scheduler.followup_engine import send_followups_job

__all__ = [
    "start_scheduler",
    "stop_scheduler",
    "check_replies_job",
    "send_followups_job",
]

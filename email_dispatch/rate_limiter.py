"""
email_dispatch/rate_limiter.py

Email sending rate limiter.

Enforces:
  - Max N emails per day (Gmail free tier: 500/day)
  - Max N emails per week (configurable from plan)
  - Minimum delay between sends (to avoid spam detection)
  - Per-session counters with DB persistence
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console

import config

console = Console()

# Defaults
GMAIL_DAILY_LIMIT = 100      # Conservative (Gmail allows 500)
MIN_DELAY_SECONDS = 30       # Minimum gap between two sends


class RateLimiter:
    """
    Simple in-memory rate limiter with configurable daily/weekly caps.
    In production, counters should be stored in the database.
    """

    def __init__(
        self,
        max_per_day: int = GMAIL_DAILY_LIMIT,
        max_per_week: Optional[int] = None,
        min_delay_seconds: float = MIN_DELAY_SECONDS,
    ):
        self.max_per_day = max_per_day
        self.max_per_week = max_per_week or config.MAX_EMAILS_PER_WEEK
        self.min_delay_seconds = min_delay_seconds

        self._sent_today: list[datetime] = []
        self._sent_this_week: list[datetime] = []
        self._last_sent_at: Optional[datetime] = None

    def _clean_old_records(self) -> None:
        """Remove timestamps older than tracking windows."""
        now = datetime.utcnow()
        self._sent_today = [
            t for t in self._sent_today
            if (now - t) < timedelta(days=1)
        ]
        self._sent_this_week = [
            t for t in self._sent_this_week
            if (now - t) < timedelta(weeks=1)
        ]

    def can_send(self) -> tuple[bool, str]:
        """
        Check whether we are allowed to send another email right now.
        Returns (allowed: bool, reason: str).
        """
        self._clean_old_records()
        now = datetime.utcnow()

        # Check daily limit
        if len(self._sent_today) >= self.max_per_day:
            return False, f"Daily limit reached ({self.max_per_day}/day)"

        # Check weekly limit
        if len(self._sent_this_week) >= self.max_per_week:
            return False, f"Weekly limit reached ({self.max_per_week}/week)"

        # Check minimum delay between sends
        if self._last_sent_at:
            elapsed = (now - self._last_sent_at).total_seconds()
            if elapsed < self.min_delay_seconds:
                wait = self.min_delay_seconds - elapsed
                return False, f"Rate limit: wait {wait:.0f}s before next send"

        return True, "OK"

    def wait_if_needed(self) -> None:
        """Block until the next send is allowed."""
        self._clean_old_records()
        if self._last_sent_at:
            elapsed = (datetime.utcnow() - self._last_sent_at).total_seconds()
            if elapsed < self.min_delay_seconds:
                wait = self.min_delay_seconds - elapsed
                console.print(f"[dim]⏳ Rate limit: waiting {wait:.0f}s...[/dim]")
                time.sleep(wait)

    def record_send(self) -> None:
        """Record that an email was just sent."""
        now = datetime.utcnow()
        self._sent_today.append(now)
        self._sent_this_week.append(now)
        self._last_sent_at = now

    def stats(self) -> dict:
        """Return current rate limit stats."""
        self._clean_old_records()
        return {
            "sent_today": len(self._sent_today),
            "sent_this_week": len(self._sent_this_week),
            "max_per_day": self.max_per_day,
            "max_per_week": self.max_per_week,
            "remaining_today": max(0, self.max_per_day - len(self._sent_today)),
            "remaining_this_week": max(0, self.max_per_week - len(self._sent_this_week)),
        }

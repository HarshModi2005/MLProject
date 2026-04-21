"""
scheduler/followup_engine.py

Automated follow-up email sender.

Identifies emails that were sent > N days ago with no reply
and sends a brief, polite follow-up.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from rich.console import Console

from db import database as db
from db.models import Email as EmailRow

console = Console()


class FollowupEngine:
    """
    Processes pending follow-ups for emails with no reply after N days.
    """

    def __init__(
        self,
        followup_days: int = 7,
        session_id: Optional[str] = None,
    ):
        self.followup_days = followup_days
        self.session_id = session_id
        self._gmail = None
        self._email_agent = None

    def _get_gmail(self):
        if self._gmail is None:
            from email_dispatch.gmail_client import GmailClient
            self._gmail = GmailClient()
        return self._gmail

    def _get_email_agent(self):
        if self._email_agent is None:
            from agents.email_agent import EmailAgent
            self._email_agent = EmailAgent()
        return self._email_agent

    def process_pending_followups(self) -> int:
        """
        Find and send follow-ups for all eligible emails.
        Returns count of follow-ups sent.
        """
        pending: list[EmailRow] = db.get_pending_followups(self.followup_days)

        if not pending:
            console.print("  [dim]No follow-ups needed.[/dim]")
            return 0

        console.print(f"  [yellow]📧 {len(pending)} follow-up(s) to send.[/yellow]")
        sent_count = 0

        for email_row in pending:
            days_elapsed = 0
            if email_row.sent_at:
                days_elapsed = (datetime.utcnow() - email_row.sent_at).days

            console.print(
                f"  → Follow-up to {email_row.recipient_email} "
                f"({days_elapsed} days since original)"
            )

            # Generate follow-up body
            followup_body = self._generate_followup_body(
                email_row.recipient_name,
                email_row.subject,
                days_elapsed,
            )
            followup_subject = f"Re: {email_row.subject}"

            # Send via Gmail
            try:
                gmail = self._get_gmail()
                msg_id = gmail.send_email(
                    to=email_row.recipient_email,
                    subject=followup_subject,
                    body=followup_body,
                )
                if msg_id:
                    # Update follow-up count in DB
                    with db.db_session() as session:
                        row = session.get(EmailRow, email_row.id)
                        if row:
                            row.follow_up_count = (row.follow_up_count or 0) + 1
                            row.status = "followed_up"
                    sent_count += 1
                    console.print(f"  [green]✓ Follow-up sent.[/green]")
                else:
                    console.print(f"  [red]✗ Failed to send follow-up.[/red]")
            except Exception as e:
                console.print(f"  [red]✗ Follow-up error: {e}[/red]")

        return sent_count

    def _generate_followup_body(
        self,
        recipient_name: str,
        original_subject: str,
        days_elapsed: int,
    ) -> str:
        """Generate a brief, polite follow-up email body."""
        last_name = recipient_name.split()[-1] if recipient_name else "Professor"
        return (
            f"Dear Professor {last_name},\n\n"
            f"I hope this message finds you well. I wanted to follow up on my previous email "
            f"({days_elapsed} days ago) regarding the subject: \"{original_subject}\".\n\n"
            f"I understand you are likely very busy, and I appreciate any time "
            f"you can spare. I remain very interested in the possibility of collaborating "
            f"with your research group.\n\n"
            f"Please let me know if you have any questions or would like additional "
            f"information from my end.\n\n"
            f"Thank you for your time.\n\nBest regards"
        )

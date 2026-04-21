"""
scheduler/reply_monitor.py

Gmail inbox monitor for detecting professor replies.

Strategy:
  1. Pull recent unread messages from Gmail
  2. Match sender email against our sent emails table
  3. Mark matched emails as "replied" in the DB
"""

from __future__ import annotations

from rich.console import Console
from db import database as db

console = Console()


class ReplyMonitor:
    """
    Scans the Gmail inbox for replies to our outreach emails
    and updates the DB accordingly.
    """

    def __init__(self):
        self._gmail = None

    def _get_gmail(self):
        if self._gmail is None:
            from email_dispatch.gmail_client import GmailClient
            self._gmail = GmailClient()
        return self._gmail

    def scan_and_update_replies(self) -> int:
        """
        Scan inbox, match senders to sent emails, mark as replied.
        Returns the number of new replies detected.
        """
        try:
            gmail = self._get_gmail()
        except Exception as e:
            console.print(f"[yellow]⚠ Gmail not configured: {e}[/yellow]")
            return 0

        # Fetch recent unread messages
        messages = gmail.list_recent_replies(
            query="in:inbox is:unread",
            max_results=50,
        )

        if not messages:
            console.print("  [dim]No new messages.[/dim]")
            return 0

        reply_count = 0

        for msg_stub in messages:
            msg_id = msg_stub.get("id")
            if not msg_id:
                continue

            details = gmail.get_message_details(msg_id)
            if not details:
                continue

            sender = gmail.get_sender_email(details)
            if not sender:
                continue

            # Look up DB for a sent email to this sender
            with db.db_session() as session:
                from db.models import Email
                sent_email = (
                    session.query(Email)
                    .filter(
                        Email.recipient_email == sender,
                        Email.status == "sent",
                    )
                    .first()
                )
                if sent_email:
                    db.mark_email_replied(sent_email.id)
                    console.print(
                        f"  [green]✓ Reply detected from {sender}[/green]"
                    )
                    reply_count += 1

        console.print(f"  [dim]Reply scan complete. {reply_count} new replies detected.[/dim]")
        return reply_count

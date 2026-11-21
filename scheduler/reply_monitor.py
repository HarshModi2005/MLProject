"""
scheduler/reply_monitor.py

Gmail inbox monitor for detecting professor replies.

Strategy:
  1. Pull recent unread messages from Gmail
  2. Match sender email against our sent emails table
  3. Read what the reply says (brain/replies.py): a request to stop ("no", "unsubscribe", "remove me", "not interested")
     suppresses the address for every campaign; an out-of-office or delivery notice is not a reply at all; a polite
     decline is recorded as a decline, never as a good prospect; anything else is marked "replied" in the DB
"""

from __future__ import annotations

from rich.console import Console
from db import database as db
from brain.replies import classify_reply, header, message_text

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
        stopped = 0

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
                        Email.status.in_(["sent", "followed_up"]),  # a reply to a follow-up counts too
                    )
                    .first()
                )
                sent_id = sent_email.id if sent_email else None
            kind = classify_reply(message_text(details), header(details, "Subject"), sender)
            if kind == "unsubscribed":
                # Honoured even when no sent email matches: someone who asked to stop is never written to again.
                from brain import bridge

                bridge.record_reply(sender, "unsubscribed", detail="asked to stop")
                stopped += 1
            if not sent_id:
                continue
            if kind == "auto_reply":
                console.print(f"  [dim]Ignored an automatic reply from {sender}.[/dim]")
                continue
            db.mark_email_replied(sent_id)
            if kind == "negative":
                from brain import bridge

                bridge.record_reply(sender, "negative", detail="declined")
            console.print(f"  [green]✓ Reply detected from {sender}[/green]"
                          + (" [yellow](asked to stop: suppressed)[/yellow]" if kind == "unsubscribed" else ""))
            reply_count += 1

        console.print(f"  [dim]Reply scan complete. {reply_count} new replies detected"
                      f"{f', {stopped} asked to stop' if stopped else ''}.[/dim]")
        return reply_count

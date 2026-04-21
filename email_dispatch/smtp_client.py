"""
email_dispatch/smtp_client.py

Fallback SMTP client for sending emails if Gmail API is unavailable.
"""

import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from typing import Optional

from rich.console import Console

import config
from orchestrator.state import DraftEmail

console = Console()

class SMTPClient:
    """
    Standard SMTP Mail client.
    Requires SENDER_EMAIL and an App Password in the environment.
    """

    def __init__(self):
        self.sender = config.SENDER_EMAIL
        # We would typically expect an App Password if using Gmail via SMTP
        self.app_password = getattr(config, "SMTP_APP_PASSWORD", "")
        self.smtp_host = getattr(config, "SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = getattr(config, "SMTP_PORT", 587)

    def is_configured(self) -> bool:
        """Check if SMTP is properly configured."""
        return bool(self.sender and self.app_password)

    def send_email(
        self,
        draft: DraftEmail,
        attachment_path: Optional[str] = None
    ) -> bool:
        """
        Sends an email via SMTP.
        Returns True if successful, False otherwise.
        """
        if not self.is_configured():
            console.print("[red]SMTP is not configured! Missing SENDER_EMAIL or SMTP_APP_PASSWORD.[/red]")
            return False

        msg = EmailMessage()
        msg['Subject'] = draft.subject
        msg['From'] = self.sender
        msg['To'] = draft.recipient_email
        
        # Add a unique Message-ID (useful for reply tracking later)
        msg['Message-ID'] = make_msgid()

        # Assuming the body might be HTML.
        # Fallback to plain text if no HTML tags are found.
        if "<p>" in draft.body.lower() or "<br>" in draft.body.lower() or "<html>" in draft.body.lower():
             msg.set_content("Please view this email in an HTML-capable client.")
             msg.add_alternative(draft.body, subtype='html')
        else:
             msg.set_content(draft.body)

        # Handle attachment
        if attachment_path:
            path = Path(attachment_path)
            if path.exists() and path.is_file():
                try:
                    with open(path, 'rb') as f:
                        file_data = f.read()
                        file_name = path.name
                    msg.add_attachment(
                        file_data,
                        maintype='application',
                        subtype='pdf',
                        filename=file_name
                    )
                except Exception as e:
                    console.print(f"[red]Failed to attach {attachment_path}: {e}[/red]")
            else:
                console.print(f"[yellow]Attachment not found: {attachment_path}[/yellow]")

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.app_password)
                server.send_message(msg)
                
            console.print(f"[green]✓ Successfully sent SMTP email to {draft.recipient_email}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to send SMTP email to {draft.recipient_email}: {e}[/red]")
            return False

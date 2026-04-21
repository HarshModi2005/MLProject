import os
import base64
from typing import Optional

from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import config
from rich.console import Console

console = Console()

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

class GmailSender:
    def __init__(self):
        self.creds = None
        self.service = None
        
    def authenticate(self):
        """Authenticate with Gmail using OAuth2."""
        try:
            # We construct credentials directly from .env variables
            if config.GMAIL_CLIENT_ID and config.GMAIL_CLIENT_SECRET and config.GMAIL_REFRESH_TOKEN:
                self.creds = Credentials(
                    token=None,
                    refresh_token=config.GMAIL_REFRESH_TOKEN,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=config.GMAIL_CLIENT_ID,
                    client_secret=config.GMAIL_CLIENT_SECRET,
                    scopes=SCOPES
                )
                
                # Refresh token if expired
                if not self.creds.valid:
                    self.creds.refresh(Request())
                    
                self.service = build('gmail', 'v1', credentials=self.creds)
                return True
            else:
                console.print("[yellow]Gmail OAuth2 credentials missing in .env. Email sending disabled.[/yellow]")
                return False
        except Exception as e:
            err = str(e).lower()
            console.print(f"[red]Gmail authentication failed: {e}[/red]")
            if "invalid_grant" in err:
                console.print(
                    "[yellow]invalid_grant[/yellow] means Google rejected your refresh token. Typical fixes:\n"
                    "  • Regenerate the token: [bold]python3 get_refresh_token.py[/bold] — use the [bold]same[/bold] "
                    "OAuth client as GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET in .env, then paste the new "
                    "GMAIL_REFRESH_TOKEN (no quotes, no line breaks).\n"
                    "  • In Google Cloud Console → APIs & Services → OAuth client: type must be "
                    "[bold]Desktop[/bold] (Installed app) for get_refresh_token.py.\n"
                    "  • Revoked access: https://myaccount.google.com/permissions — remove the app, then run "
                    "get_refresh_token.py again.\n"
                    "  • Password change or 6+ months idle can invalidate tokens; re-authorize.\n"
                )
            return False

    def send_email(
        self,
        to_email: str,
        subject: str,
        message_body: str,
        attachment_path: Optional[str] = None,
    ) -> bool:
        """Send an email using Gmail. Optional PDF attachment (e.g. CV)."""
        if not self.service:
            if not self.authenticate():
                return False

        to_email = (to_email or "").strip()
        if not to_email or "@" not in to_email:
            console.print(
                "[red]Cannot send: recipient address is empty or invalid. "
                "Use the draft’s recipient_email (professor row may be missing email).[/red]"
            )
            return False

        try:
            message = MIMEMultipart()
            message["To"] = to_email
            message['from'] = config.SENDER_EMAIL
            message['subject'] = subject

            # Ensure body has newlines correctly formatted for MIMEText
            body_text = message_body
            if attachment_path and os.path.isfile(attachment_path):
                if "attach" not in message_body.lower() and "cv" not in message_body.lower():
                    body_text = message_body.rstrip() + "\n\n(I have attached my CV for your reference.)"
            msg_text = MIMEText(body_text, 'plain')
            message.attach(msg_text)

            if attachment_path and os.path.isfile(attachment_path):
                with open(attachment_path, "rb") as f:
                    part = MIMEApplication(f.read(), _subtype="pdf")
                    part.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=os.path.basename(attachment_path),
                    )
                    message.attach(part)
            elif attachment_path:
                console.print(f"[yellow]Attachment path not found: {attachment_path}[/yellow]")

            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            body = {'raw': raw_message}

            console.print(f"[dim]Sending email to {to_email}...[/dim]")
            sent_message = self.service.users().messages().send(userId='me', body=body).execute()
            
            console.print(f"[green]✓ Email sent successfully! (Msg ID: {sent_message['id']})[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to send email: {e}[/red]")
            return False

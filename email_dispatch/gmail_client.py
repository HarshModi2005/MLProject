"""
email_dispatch/gmail_client.py

Gmail API client for sending emails and monitoring replies.

Authentication:
  - Uses OAuth2 with pre-generated refresh token
  - Automatically refreshes access token when expired

Usage:
  client = GmailClient()
  client.send_email(to="prof@mit.edu", subject="...", body="...", attachments=[])
"""

from __future__ import annotations

import base64
import os
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from rich.console import Console
import config

console = Console()


class GmailClient:
    """
    Wrapper around the Gmail API for sending and reading emails.
    Requires OAuth2 credentials in the .env file.
    """

    def __init__(self):
        self._service = None

    def _get_service(self):
        """Lazily build and cache the Gmail API service."""
        if self._service is not None:
            return self._service

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials(
                token=None,
                refresh_token=config.GMAIL_REFRESH_TOKEN,
                client_id=config.GMAIL_CLIENT_ID,
                client_secret=config.GMAIL_CLIENT_SECRET,
                token_uri="https://oauth2.googleapis.com/token",
                scopes=["https://mail.google.com/"],
            )
            self._service = build("gmail", "v1", credentials=creds)
            return self._service

        except ImportError:
            raise ImportError(
                "google-api-python-client not installed. "
                "Run: pip install google-api-python-client google-auth google-auth-oauthlib"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build Gmail service: {e}")

    def _create_message(
        self,
        to: str,
        subject: str,
        body: str,
        attachment_path: Optional[str] = None,
    ) -> dict:
        """Build a MIME email message (with optional PDF attachment)."""
        if attachment_path:
            msg = MIMEMultipart()
            msg["to"] = to
            msg["from"] = config.SENDER_EMAIL
            msg["subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # Attach CV/file
            if os.path.isfile(attachment_path):
                with open(attachment_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
                part["Content-Disposition"] = (
                    f'attachment; filename="{os.path.basename(attachment_path)}"'
                )
                msg.attach(part)
        else:
            msg = MIMEText(body, "plain")
            msg["to"] = to
            msg["from"] = config.SENDER_EMAIL
            msg["subject"] = subject

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        return {"raw": raw}

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        attachment_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send an email via Gmail API.
        Returns the Gmail message ID on success, None on failure.
        """
        try:
            service = self._get_service()
            message = self._create_message(to, subject, body, attachment_path)
            sent = service.users().messages().send(
                userId="me", body=message
            ).execute()
            msg_id = sent.get("id")
            console.print(f"[green]✓ Email sent to {to} (Gmail ID: {msg_id})[/green]")
            return msg_id
        except Exception as e:
            console.print(f"[red]✗ Failed to send email to {to}: {e}[/red]")
            return None

    def list_recent_replies(
        self,
        query: str = "in:inbox is:unread",
        max_results: int = 20,
    ) -> list[dict]:
        """
        List recent inbox messages matching a query.
        Returns list of {id, threadId, snippet} dicts.
        """
        try:
            service = self._get_service()
            result = service.users().messages().list(
                userId="me",
                q=query,
                maxResults=max_results,
            ).execute()
            return result.get("messages", [])
        except Exception as e:
            console.print(f"[yellow]⚠ Error listing messages: {e}[/yellow]")
            return []

    def get_message_details(self, message_id: str) -> Optional[dict]:
        """Fetch full details of a single Gmail message."""
        try:
            service = self._get_service()
            return service.users().messages().get(
                userId="me",
                id=message_id,
                format="full",
            ).execute()
        except Exception as e:
            console.print(f"[yellow]⚠ Error fetching message {message_id}: {e}[/yellow]")
            return None

    def get_sender_email(self, message: dict) -> Optional[str]:
        """Extract the From: address from a Gmail message object."""
        headers = message.get("payload", {}).get("headers", [])
        for h in headers:
            if h.get("name", "").lower() == "from":
                val = h.get("value", "")
                # Extract email from "Name <email>" format
                import re
                match = re.search(r"<([^>]+)>", val)
                if match:
                    return match.group(1)
                return val
        return None

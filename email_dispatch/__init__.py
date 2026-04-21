"""
email_dispatch/__init__.py

Handles sending emails via Gmail API and rate limiting.
"""

from email_dispatch.gmail_client import GmailClient
from email_dispatch.rate_limiter import RateLimiter
from email_dispatch.smtp_client import SMTPClient

__all__ = [
    "GmailClient",
    "RateLimiter",
    "SMTPClient",
]

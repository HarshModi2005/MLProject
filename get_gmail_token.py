"""
Obtain GMAIL_REFRESH_TOKEN for .env.

Google strongly recommends a **Desktop** OAuth client for local / CLI flows.
Web application + http://localhost often returns redirect_uri_mismatch even when URIs look correct.

Setup:
  1. Google Cloud Console → APIs & Services → Credentials
  2. Create Credentials → OAuth client ID
  3. Application type: **Desktop** (not "Web application")
  4. Copy Client ID + Client Secret into .env

Optional: if you must use a Web application client, set in .env:
  GMAIL_OAUTH_CLIENT_TYPE=web
and register http://localhost:8080/ (and JS origins) exactly as in Google docs.
"""

import os
import sys
from google_auth_oauthlib.flow import InstalledAppFlow
import config

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

_WEB_REDIRECT_URIS = ["http://localhost:8080/", "http://127.0.0.1:8080/"]


def _installed_block() -> dict:
    # Matches Google "Desktop" client download; loopback port is allowed for Desktop clients.
    return {
        "client_id": config.GMAIL_CLIENT_ID,
        "project_id": "outreach-agent",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": config.GMAIL_CLIENT_SECRET,
        "redirect_uris": ["http://localhost", "http://127.0.0.1"],
    }


def _web_block() -> dict:
    return {
        "client_id": config.GMAIL_CLIENT_ID,
        "project_id": "outreach-agent",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": config.GMAIL_CLIENT_SECRET,
        "redirect_uris": list(_WEB_REDIRECT_URIS),
        "javascript_origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
    }


def main():
    if not config.GMAIL_CLIENT_ID or not config.GMAIL_CLIENT_SECRET:
        print("Please set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET in .env first.")
        sys.exit(1)

    use_web = os.getenv("GMAIL_OAUTH_CLIENT_TYPE", "").lower() == "web"
    client_config = {"web": _web_block()} if use_web else {"installed": _installed_block()}

    if not use_web:
        print(
            "\n>>> Using **Desktop** OAuth client mode (recommended).\n"
            "    Your GCP credential type must be **Desktop app**, not Web application.\n"
            "    Create one: Credentials → Create OAuth client ID → Application type: Desktop.\n"
            "    Put that client’s ID and secret in .env.\n"
        )
    else:
        print(
            "\n>>> Using **Web application** OAuth mode.\n"
            "    Authorized redirect URIs must include exactly:\n"
            f"      {_WEB_REDIRECT_URIS[0]}\n"
            f"      {_WEB_REDIRECT_URIS[1]}\n"
            "    Authorized JavaScript origins: http://localhost:8080 and http://127.0.0.1:8080\n"
        )

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

    host = os.getenv("GMAIL_OAUTH_BIND", "localhost").strip() or "localhost"
    port = int(os.getenv("GMAIL_OAUTH_PORT", "8080") or "8080")

    try:
        creds = flow.run_local_server(host=host, port=port, open_browser=True)
    except Exception as e:
        err = str(e).lower()
        if "redirect" in err or "400" in err:
            print(
                "\nStill failing? Do this:\n"
                "  1) Create a NEW OAuth client: type **Desktop** (easiest fix).\n"
                "  2) Update .env with that client’s ID + secret (remove GMAIL_OAUTH_CLIENT_TYPE=web).\n"
                "  3) Or try: GMAIL_OAUTH_BIND=127.0.0.1 python3 get_gmail_token.py\n"
            )
        raise

    print("\n\n=== SUCCESS! ADD THIS TO YOUR .ENV FILE ===")
    print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
    print("===========================================\n")


if __name__ == "__main__":
    main()

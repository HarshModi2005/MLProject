import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
_WEB_REDIRECT_URIS = ["http://localhost:8080/", "http://127.0.0.1:8080/"]


def _installed_block(cid: str, secret: str) -> dict:
    return {
        "client_id": cid,
        "project_id": "outreach-agent",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": secret,
        "redirect_uris": ["http://localhost", "http://127.0.0.1"],
    }


def _web_block(cid: str, secret: str) -> dict:
    return {
        "client_id": cid,
        "project_id": "outreach-agent",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": secret,
        "redirect_uris": list(_WEB_REDIRECT_URIS),
        "javascript_origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
    }


def write_refresh_token_to_dotenv(env_path: Path, token: str) -> None:
    """Set GMAIL_REFRESH_TOKEN in .env (single-line value, no shell quoting)."""
    raw = env_path.read_text(encoding="utf-8")
    line = f"GMAIL_REFRESH_TOKEN={token}\n"
    pattern = r"^GMAIL_REFRESH_TOKEN=.*(?:\n|$)"
    if re.search(pattern, raw, flags=re.MULTILINE):
        new_raw = re.sub(pattern, line, raw, count=1, flags=re.MULTILINE)
    else:
        new_raw = raw.rstrip() + "\n" + line
    env_path.write_text(new_raw, encoding="utf-8")


def main():
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("Please ensure GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET are in your .env file")
        return

    use_web = os.getenv("GMAIL_OAUTH_CLIENT_TYPE", "").lower() == "web"
    client_config = (
        {"web": _web_block(client_id, client_secret)}
        if use_web
        else {"installed": _installed_block(client_id, client_secret)}
    )

    if not use_web:
        print("\nDesktop OAuth client mode: use GCP credential type **Desktop app**.\n")

    try:
        flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
        host = os.getenv("GMAIL_OAUTH_BIND", "localhost").strip() or "localhost"
        port = int(os.getenv("GMAIL_OAUTH_PORT", "8080") or "8080")
        creds = flow.run_local_server(
            host=host,
            port=port,
            access_type="offline",
            prompt="consent",
        )

        print("\n\n" + "=" * 50)
        print("SUCCESS! Here is your Refresh Token:")
        print("-" * 50)
        print(creds.refresh_token)
        print("-" * 50)
        print("\nPlease copy the token above and paste it into your .env file as:")
        print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
        print("=" * 50 + "\n")

        if creds.refresh_token and os.getenv("GMAIL_SKIP_WRITE_ENV", "").lower() not in (
            "1",
            "true",
            "yes",
        ):
            env_file = Path(os.getenv("GMAIL_ENV_FILE", ".env")).resolve()
            if env_file.is_file():
                write_refresh_token_to_dotenv(env_file, creds.refresh_token)
                print(f"Wrote GMAIL_REFRESH_TOKEN to {env_file}\n")
            else:
                print(f"(No {env_file} file — set GMAIL_ENV_FILE or create .env to auto-save.)\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "\nTip: Create an OAuth client of type **Desktop** in Google Cloud, "
            "put its ID/secret in .env, and run again (do not set GMAIL_OAUTH_CLIENT_TYPE=web)."
        )


if __name__ == "__main__":
    main()

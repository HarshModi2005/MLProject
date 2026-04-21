"""
config.py - Centralized configuration loader for the Research Outreach Agent.
Loads all env variables, validates presence of required keys, and exposes
typed config constants for use across all modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ── LLM Configuration ─────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# ── Search APIs ───────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

# ── Gmail / Email ─────────────────────────────────────────────────────────────
GMAIL_CLIENT_ID: str = os.getenv("GMAIL_CLIENT_ID", "")
GMAIL_CLIENT_SECRET: str = os.getenv("GMAIL_CLIENT_SECRET", "")
GMAIL_REFRESH_TOKEN: str = os.getenv("GMAIL_REFRESH_TOKEN", "")
SENDER_EMAIL: str = os.getenv("SENDER_EMAIL", "")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./outreach.db")

# ── Vector Store ──────────────────────────────────────────────────────────────
CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", str(BASE_DIR / "chroma_db"))

# ── Outreach Config ───────────────────────────────────────────────────────────
MAX_EMAILS_PER_WEEK: int = int(os.getenv("MAX_EMAILS_PER_WEEK", "10"))
FOLLOWUP_DAYS: int = int(os.getenv("FOLLOWUP_DAYS", "7"))
# Shortlist size after scoring (override in .env to tune cost vs coverage).
TOP_K_PROFESSORS: int = int(os.getenv("TOP_K_PROFESSORS", "25"))

# Crawl / search volume (override in .env for quick test runs).
MAX_CRAWL_URLS: int = int(os.getenv("MAX_CRAWL_URLS", "50"))
SEARCH_MAX_RESULTS_PER_QUERY: int = int(os.getenv("SEARCH_MAX_RESULTS_PER_QUERY", "10"))
SEMANTIC_SCHOLAR_AUTHORS_PER_KEYWORD: int = int(
    os.getenv("SEMANTIC_SCHOLAR_AUTHORS_PER_KEYWORD", "15")
)
# How many plan keywords feed Semantic Scholar
SEMANTIC_SCHOLAR_KEYWORD_COUNT: int = int(os.getenv("SEMANTIC_SCHOLAR_KEYWORD_COUNT", "8"))
# Max web-search queries per crawl (was hardcoded at 10 in search_api).
MAX_SEARCH_QUERIES: int = int(os.getenv("MAX_SEARCH_QUERIES", "45"))
# Co-author expansion via Semantic Scholar (set CRAWL_SNOWBALL=false to disable)
CRAWL_SNOWBALL: bool = os.getenv("CRAWL_SNOWBALL", "true").lower() in ("1", "true", "yes")

# Polite delays (seconds) — slightly higher defaults help when volume is increased.
SEARCH_QUERY_SLEEP_SEC: float = float(os.getenv("SEARCH_QUERY_SLEEP_SEC", "0.55"))
SCRAPE_BATCH_DELAY: float = float(os.getenv("SCRAPE_BATCH_DELAY", "1.2"))
SEMANTIC_SCHOLAR_SLEEP_SEC: float = float(os.getenv("SEMANTIC_SCHOLAR_SLEEP_SEC", "0.55"))

# ── Prompts directory ─────────────────────────────────────────────────────────
PROMPTS_DIR: Path = BASE_DIR / "prompts"

# ── Observability (JSONL run logs) ────────────────────────────────────────────
RUN_LOG_DIR: Path = Path(os.getenv("RUN_LOG_DIR", str(BASE_DIR / "run_logs")))
ENABLE_STRUCTURED_LOGS: bool = os.getenv("ENABLE_STRUCTURED_LOGS", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Email enrichment / variants (defaults keep prior single-draft behavior)
ENABLE_EMAIL_DISCOVERY_METADATA: bool = os.getenv(
    "ENABLE_EMAIL_DISCOVERY_METADATA", "true"
).lower() in ("1", "true", "yes")
DEFAULT_GENERATE_EMAIL_VARIANTS: bool = os.getenv(
    "DEFAULT_GENERATE_EMAIL_VARIANTS", "false"
).lower() in ("1", "true", "yes")


def validate_config() -> list[str]:
    """
    Returns a list of missing required environment variable names.
    Call this at startup and warn the user if any are absent.
    """
    required = {"GROQ_API_KEY": GROQ_API_KEY}
    missing = [k for k, v in required.items() if not v]
    return missing


def get_groq_llm():
    """
    Returns a configured LangChain ChatGroq instance.
    Centralised so all agents share the same model config.
    """
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Please add it to your .env file."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
    )


def get_groq_client():
    """
    Returns a raw groq client instance.
    Use for direct API calls without LangChain overhead.
    """
    from groq import Groq
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Please add it to your .env file."
        )
    return Groq(api_key=GROQ_API_KEY)

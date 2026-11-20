"""Runtime settings for the brain, read once from the environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: the brain works without python-dotenv
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Provider presets: base URL, key variable, strong / fast models, fallbacks.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
             "openai/gpt-oss-20b,qwen/qwen3.8-27b"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4o-mini"),
}
# Approximate USD per 1M tokens (input, output), for the cost estimate shown after each run.
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60), "gpt-4.1": (2.0, 8.0), "gpt-4o": (2.5, 10.0),
          "gpt-5-mini": (0.25, 2.0), "gpt-5-nano": (0.05, 0.40)}


def _provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit in PROVIDERS:
        return explicit
    return "groq" if os.getenv("GROQ_API_KEY") or not os.getenv("OPENAI_API_KEY") else "openai"


def _preset(i: int) -> str:
    return PROVIDERS[_provider()][i]


@dataclass
class Settings:
    # LLM: any OpenAI-compatible chat endpoint (Groq by default, as in the legacy code).
    llm_provider: str = field(default_factory=_provider)
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", _preset(0)))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv(_preset(1), ""))
    llm_model_strong: str = field(default_factory=lambda: os.getenv("LLM_MODEL_STRONG", _preset(2)))
    llm_model_fast: str = field(default_factory=lambda: os.getenv("LLM_MODEL_FAST", _preset(3)))
    # Tried in order when a model is rate-limited for longer than a minute (e.g. a daily token cap).
    llm_fallback_models: str = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODELS", _preset(4)))
    # The fact-checker should not share the writer's blind spots. LLM_MODEL_VERIFY picks it (default: the fast model,
    # which differs from the writer's strong model); LLM_VERIFY_BASE_URL / LLM_VERIFY_API_KEY point it at another
    # provider altogether (any OpenAI-compatible endpoint), which is the strongest independence.
    llm_model_verify: str = field(default_factory=lambda: os.getenv("LLM_MODEL_VERIFY", ""))
    llm_verify_base_url: str = field(default_factory=lambda: os.getenv("LLM_VERIFY_BASE_URL", ""))
    llm_verify_api_key: str = field(default_factory=lambda: os.getenv("LLM_VERIFY_API_KEY", ""))
    # Reasoning models (gpt-oss, qwen3) think before answering; "low" keeps that cheap. Empty = provider default.
    llm_reasoning_effort: str = field(default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", "low"))
    # Hard ceiling on LLM calls per campaign run. Cached calls are free and not counted.
    llm_max_calls: int = field(default_factory=lambda: _int("BRAIN_LLM_MAX_CALLS", 120))

    # Web search (optional). Without a key, discovery still works through OpenAlex/ORCID/CSV.
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    max_search_calls: int = field(default_factory=lambda: _int("BRAIN_MAX_SEARCH_CALLS", 80))

    # OpenAlex (free key gives a private daily budget; works without one).
    openalex_api_key: str = field(default_factory=lambda: os.getenv("OPENALEX_API_KEY", ""))

    # Storage. `workspace` isolates one user's data from another's: its own database and cache (brain/workspace.py).
    workspace: str = field(default_factory=lambda: os.getenv("BRAIN_WORKSPACE", "default") or "default")
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./outreach.db"))
    cache_path: str = field(default_factory=lambda: os.getenv("BRAIN_CACHE_PATH", "./.brain_cache.sqlite"))

    # Crawling
    user_agent: str = field(default_factory=lambda: os.getenv("BRAIN_USER_AGENT", "OutreachResearchBot/0.1"))
    per_host_delay: float = field(default_factory=lambda: _float("BRAIN_PER_HOST_DELAY", 1.0))
    fetch_workers: int = field(default_factory=lambda: _int("BRAIN_FETCH_WORKERS", 8))
    max_page_fetches: int = field(default_factory=lambda: _int("BRAIN_MAX_PAGE_FETCHES", 400))

    # Ranking
    embeddings: str = field(default_factory=lambda: os.getenv("BRAIN_EMBEDDINGS", "lexical"))  # lexical | sentence-transformers
    min_email_confidence: float = field(default_factory=lambda: _float("BRAIN_MIN_EMAIL_CONFIDENCE", 0.6))
    max_per_org: int = field(default_factory=lambda: _int("BRAIN_MAX_PER_ORG", 3))
    judge_batch_size: int = field(default_factory=lambda: _int("BRAIN_JUDGE_BATCH", 8))
    max_extract_calls: int = field(default_factory=lambda: _int("BRAIN_MAX_EXTRACT_CALLS", 15))

    # Discovery v2
    # One batched fast LLM call per query settles search hits the heuristics cannot place (own site / list /
    # article). Set to 0 to use heuristics only.
    triage_llm: bool = field(default_factory=lambda: os.getenv("BRAIN_TRIAGE_LLM", "1") not in ("0", "false", "no"))
    light_read_factor: int = field(default_factory=lambda: _int("BRAIN_LIGHT_READ_FACTOR", 3))  # homepages read per target
    max_people_searches: int = field(default_factory=lambda: _int("BRAIN_MAX_PEOPLE_SEARCHES", 6))
    explore_rounds: int = field(default_factory=lambda: _int("BRAIN_EXPLORE_ROUNDS", 3))
    backfill: bool = field(default_factory=lambda: os.getenv("BRAIN_BACKFILL", "1") not in ("0", "false", "no"))
    use_memory: bool = field(default_factory=lambda: os.getenv("BRAIN_MEMORY", "1") not in ("0", "false", "no"))
    # The shortlist never pads: a candidate scoring below this is left out even if that leaves fewer than wanted
    # (the run then says "found N of K" instead of filling the list with weak fits).
    judge_prescore_weight: float = field(default_factory=lambda: _float("BRAIN_JUDGE_PRESCORE_WEIGHT", 0.0))  # old value 0.2
    shortlist_floor: float = field(default_factory=lambda: _float("BRAIN_SHORTLIST_FLOOR", 0.6))
    # Precision layer (brain/discovery/precision): org-type gate, pre-judge before enrichment, learned fit model.
    org_type_gate: bool = field(default_factory=lambda: os.getenv("BRAIN_ORG_TYPE_GATE", "1") not in ("0", "false", "no"))
    prejudge: bool = field(default_factory=lambda: os.getenv("BRAIN_PREJUDGE", "1") not in ("0", "false", "no"))
    prejudge_limit: int = field(default_factory=lambda: _int("BRAIN_PREJUDGE_LIMIT", 0))  # 0 = max(30, 8 x target)
    prejudge_min: float = field(default_factory=lambda: _float("BRAIN_PREJUDGE_MIN", 0.3))
    warm: bool = field(default_factory=lambda: os.getenv("BRAIN_WARM", "1") not in ("0", "false", "no"))  # imported network
    learned_fit: bool = field(default_factory=lambda: os.getenv("BRAIN_LEARNED_FIT", "1") not in ("0", "false", "no"))

    # Replay (benchmarks): "" = live; "replay" = network answers only from the cache (misses are skipped, never
    # fetched), LLM calls still allowed; "strict" = LLM calls also only from the cache.
    replay: str = field(default_factory=lambda: os.getenv("BRAIN_REPLAY", "").lower())

    def __post_init__(self) -> None:
        from brain import workspace

        self.workspace = workspace.clean(self.workspace)
        self.cache_path = workspace.scope_path(self.cache_path, self.workspace)  # cached model outputs are per user

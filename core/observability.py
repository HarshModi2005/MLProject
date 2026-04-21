"""
Structured JSONL logging for pipeline runs: session_id, run_id, spans, LLM heuristics.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import config

_run_id_ctx: ContextVar[Optional[str]] = ContextVar("obs_run_id", default=None)
_session_id_ctx: ContextVar[Optional[str]] = ContextVar("obs_session_id", default=None)


def set_run_context(*, run_id: str, session_id: Optional[str] = None) -> None:
    _run_id_ctx.set(run_id)
    if session_id is not None:
        _session_id_ctx.set(session_id)


def get_run_id() -> Optional[str]:
    return _run_id_ctx.get()


def get_session_id() -> Optional[str]:
    return _session_id_ctx.get()


def new_run_id() -> str:
    return str(uuid.uuid4())


def _base_event() -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": _run_id_ctx.get(),
        "session_id": _session_id_ctx.get(),
    }


def _log_path() -> Path:
    rid = _run_id_ctx.get() or "unknown"
    config.RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return config.RUN_LOG_DIR / f"{rid}.jsonl"


def log_event(event_type: str, **fields: Any) -> None:
    if not config.ENABLE_STRUCTURED_LOGS:
        return
    row = {**_base_event(), "type": event_type, **fields}
    path = _log_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def log_llm_call(
    *,
    label: str,
    prompt_chars: int,
    response_chars: int,
    latency_sec: float,
    model: Optional[str] = None,
) -> None:
    log_event(
        "llm",
        label=label,
        model=model,
        latency_sec=round(latency_sec, 4),
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        est_input_tokens=max(1, prompt_chars // 4),
        est_output_tokens=max(0, response_chars // 4),
    )


@contextmanager
def timed_span(name: str, **extra: Any) -> Iterator[None]:
    t0 = time.perf_counter()
    log_event("span_start", span=name, **extra)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log_event("span_end", span=name, duration_sec=round(dt, 4), **extra)

"""
Per-user isolation. Everything the engine learns (remembered prospects, email conventions, query and source statistics,
fit and calibration models, the suppression list, the user's network, campaigns, drafts) is that user's own business and must
never reach another user. Rather than add a user-id filter to every query (one forgotten filter is a leak), each
workspace gets its OWN storage:

    SQLite     one database file per workspace   (./outreach.db -> ./outreach.<workspace>.db)
    Postgres   one schema per workspace          (translated by SQLAlchemy: no query knows about it)

The "default" workspace is the plain, unscoped database, so single-user installs and existing data are untouched.
The page/LLM cache is scoped the same way, because cached model outputs derive from one user's documents.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

DEFAULT = "default"
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")


def clean(name: Optional[str]) -> str:
    """A safe workspace name (it becomes part of a file name and a schema name, so nothing else is accepted)."""
    ws = (name or DEFAULT).strip().lower()
    if not _NAME.fullmatch(ws):
        raise ValueError(f"invalid workspace {name!r}: use 1-40 characters from a-z, 0-9, '_' and '-', starting with a letter or digit")
    return ws


def scope_path(path: str, workspace: str) -> str:
    """./.brain_cache.sqlite -> ./.brain_cache.<ws>.sqlite for a non-default workspace."""
    ws = clean(workspace)
    if ws == DEFAULT or not path:
        return path
    base, dot, ext = path.rpartition(".")
    if dot and "/" not in ext and "\\" not in ext:
        return f"{base}.{ws}.{ext}"
    return f"{path}.{ws}"


def scope_url(database_url: str, workspace: str) -> Tuple[str, Optional[str]]:
    """(url, schema): SQLite files are suffixed; other databases keep their URL and use a schema."""
    from sqlalchemy.engine import make_url

    ws = clean(workspace)
    if ws == DEFAULT:
        return database_url, None
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if not url.database or url.database == ":memory:":
            return database_url, None  # in-memory databases are private to their connection already
        return url.set(database=scope_path(url.database, ws)).render_as_string(hide_password=False), None
    if url.get_backend_name() == "postgresql":
        return database_url, f"ws_{ws.replace('-', '_')}"
    raise ValueError(f"workspaces need SQLite or PostgreSQL (got {url.get_backend_name()})")

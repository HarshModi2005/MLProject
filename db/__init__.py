"""
db/__init__.py

Database layer: SQLAlchemy setup, ORM models, and session management.
"""

from db.database import get_engine, get_session_factory, init_db, db_session
from db.models import Professor, Email, Session

__all__ = [
    "get_engine",
    "get_session_factory",
    "init_db",
    "db_session",
    "Professor",
    "Email",
    "Session",
]

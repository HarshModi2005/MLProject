"""
orchestrator/__init__.py

Defines the LangGraph orchestrator and core agent state.

Graph and tools are lazy-imported so `import orchestrator.state` (via student_context)
does not pull the full LangGraph + agent stack during interpreter startup.
"""

from __future__ import annotations

from typing import Any

from orchestrator.state import (
    AgentMode,
    AgentState,
    DraftEmail,
    ExecutionPlan,
    ProfessorProfile,
    UserIntent,
)

__all__ = [
    "AgentState",
    "UserIntent",
    "ExecutionPlan",
    "ProfessorProfile",
    "DraftEmail",
    "AgentMode",
    "build_graph",
    "run_supervised_loop",
    "search_tool",
    "scrape_tool",
    "parse_cv_tool",
]


def __getattr__(name: str) -> Any:
    if name == "build_graph":
        from orchestrator.graph import build_graph

        return build_graph
    if name == "run_supervised_loop":
        from orchestrator.graph import run_supervised_loop

        return run_supervised_loop
    if name == "search_tool":
        from orchestrator.tools import search_tool

        return search_tool
    if name == "scrape_tool":
        from orchestrator.tools import scrape_tool

        return scrape_tool
    if name == "parse_cv_tool":
        from orchestrator.tools import parse_cv_tool

        return parse_cv_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

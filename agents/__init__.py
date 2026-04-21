"""
agents/__init__.py

Exposes agent classes and LangGraph node functions (lazy-loaded).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "IntentAgent",
    "intent_node",
    "PlanningAgent",
    "planning_node",
    "CrawlerAgent",
    "crawl_node",
    "EmailAgent",
    "email_generation_node",
]


def __getattr__(name: str) -> Any:
    if name == "IntentAgent" or name == "intent_node":
        from agents.intent_agent import IntentAgent, intent_node

        return IntentAgent if name == "IntentAgent" else intent_node
    if name == "PlanningAgent" or name == "planning_node":
        from agents.planning_agent import PlanningAgent, planning_node

        return PlanningAgent if name == "PlanningAgent" else planning_node
    if name == "CrawlerAgent" or name == "crawl_node":
        from agents.crawler_agent import CrawlerAgent, crawl_node

        return CrawlerAgent if name == "CrawlerAgent" else crawl_node
    if name == "EmailAgent" or name == "email_generation_node":
        from agents.email_agent import EmailAgent, email_generation_node

        return EmailAgent if name == "EmailAgent" else email_generation_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

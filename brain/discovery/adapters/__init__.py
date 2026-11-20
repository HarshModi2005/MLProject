"""
Generic adapters. Each exposes `collect(spec, run) -> (contacts, stats)`, where `run` is a brain.tools.ToolRun.

    json_api   a JSON API queried once per campaign term; optional "follow" reads a URL in each item
    rss        RSS/Atom feeds (podcasts, newsletters, Google News, journals, blogs)
    list_page  known pages (directories, member / exhibitor / sponsor lists) read with CSS selectors
    sitemap    a site's sitemap, filtered by path, then pages read with selectors or a hook
    search     a search vertical (web / news / places) through the campaign's search planner and budget
    python     a function "module:function" (for sources that need real code)

All network access goes through the campaign's Fetcher (cache, robots, per-host delay, replay) and
WebSearch (budget), so every source is cached, replayable in the bench, and budgeted automatically.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

from brain.discovery.contracts import SourceSpec
from brain.models import Contact

_MODULES = {"json_api": "brain.discovery.adapters.json_api", "rss": "brain.discovery.adapters.rss",
            "list_page": "brain.discovery.adapters.list_page", "sitemap": "brain.discovery.adapters.sitemap",
            "search": "brain.discovery.adapters.search", "python": "brain.discovery.adapters.python"}


def collect(spec: SourceSpec, run) -> Tuple[List[Contact], Dict]:
    mod = importlib.import_module(_MODULES[spec.adapter])
    return mod.collect(spec, run)

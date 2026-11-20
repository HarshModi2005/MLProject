"""
python: a source implemented as a function. "entry": "module:function", called with the ToolRun and
returning (contacts, stats) — the same contract as the original brain.tools functions. Used for sources
whose logic is real code (e.g. job boards with three different APIs), while their metadata (what they
prove, cost, kinds, policy) still lives in the manifest.
"""

from __future__ import annotations

import importlib

from brain.discovery.contracts import SourceSpec


def collect(spec: SourceSpec, run):
    module, _, func = spec.entry.partition(":")
    fn = getattr(importlib.import_module(module), func)
    contacts, stats = fn(run)
    for c in contacts:
        if not c.sources:
            c.sources = [spec.name]
    return contacts, stats

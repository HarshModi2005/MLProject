"""
Source registry: loads manifests and installs them as tools in brain.tools.TOOLS, so the explorer, the
router and playbooks treat a manifest source exactly like a built-in tool.

    built-in   brain/discovery/sources/*.json
    user       ./sources/*.json (or BRAIN_SOURCES_DIR); a user manifest with a built-in's name replaces it

A manifest that fails validation is skipped with a warning (and listed in `errors()`), never fatal.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from brain.discovery.contracts import SourceSpec

log = logging.getLogger("brain.discovery")

BUILTIN_DIR = Path(__file__).parent / "sources"

SPECS: Dict[str, SourceSpec] = {}
_ERRORS: List[Tuple[str, str]] = []


def _user_dir() -> Path:
    return Path(os.getenv("BRAIN_SOURCES_DIR", "sources"))


def load_dir(folder: Path) -> Dict[str, SourceSpec]:
    out: Dict[str, SourceSpec] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else [data]:
                spec = SourceSpec(**item)
                out[spec.name] = spec
        except (ValueError, TypeError, ValidationError) as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            _ERRORS.append((str(path), msg))
            log.warning("ignoring source manifest %s: %s", path, msg)
    return out


def load_all(user_dir: Optional[Path] = None) -> Dict[str, SourceSpec]:
    specs = load_dir(BUILTIN_DIR)
    specs.update(load_dir(user_dir or _user_dir()))
    return specs


def tool_for(spec: SourceSpec):
    from brain.discovery.adapters import collect
    from brain.tools import Tool

    return Tool(name=spec.name, description=spec.title or spec.description or spec.name, cost=spec.cost,
                fn=lambda run, _s=spec: collect(_s, run), needs_search=spec.needs_search, proves=tuple(spec.proves),
                kinds=tuple(spec.kinds), stage=spec.stage, order=spec.order if spec.order is not None else 1000,
                spec=spec)


def install(tools: Dict, user_dir: Optional[Path] = None) -> List[str]:
    """Add every manifest source to `tools` (replacing same-named entries) and re-sort by `order`."""
    specs = load_all(user_dir)
    SPECS.update(specs)
    for spec in specs.values():
        tools[spec.name] = tool_for(spec)
    ordered = sorted(tools.values(), key=lambda t: (t.order, t.name))
    tools.clear()
    tools.update({t.name: t for t in ordered})
    return list(specs)


def errors() -> List[Tuple[str, str]]:
    return list(_ERRORS)

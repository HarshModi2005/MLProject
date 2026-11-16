"""
Router: which sources a campaign uses.

    1. the playbook's own list;
    2. plus, for each criterion whose proof no listed source gives, the first usable source that proves it
       (in registry order) and produces the campaign's kind of target;
    3. person campaigns drop org-only sources the playbook did not ask for;
    4. split into seed sources (run from the campaign description) and expansion sources (need known-good
       prospects); the user's examples put lookalike first; memory (free) always leads.

A source is usable when its API key is present (manifest "auth"), it serves the campaign's countries
(manifest "regions"), and it is enabled by default or the playbook names it.
"""

from __future__ import annotations

from typing import List, Tuple

from brain.models import CampaignSpec, ProspectProfile
from brain.playbooks import Playbook


def usable(tool, spec: CampaignSpec, pb: Playbook) -> Tuple[bool, str]:
    src = getattr(tool, "spec", None)
    if src is None:
        return True, ""
    if not src.available():
        return False, f"needs {src.auth.env}"
    if src.regions and "*" not in src.regions and spec.countries and not set(src.regions) & set(spec.countries):
        return False, "does not serve the campaign's countries"
    if not src.enabled_by_default and tool.name not in pb.sources:
        return False, "off by default"
    return True, ""


def choose(spec: CampaignSpec, pb: Playbook, prof: ProspectProfile, *, has_seeds: bool,
           use_memory: bool = True) -> Tuple[List[str], List[str]]:
    from brain.tools import TOOLS

    kind = spec.target_kind or pb.target_kind
    names = [n for n in pb.sources if n in TOOLS and usable(TOOLS[n], spec, pb)[0]]
    covered = {s for n in names for s in TOOLS[n].proves}
    for sig in dict.fromkeys(c.signal for c in prof.criteria):  # timing signals never add a tool
        if sig in covered or sig in ("topic", "location"):
            continue
        extra = next((t.name for t in TOOLS.values() if sig in t.proves and kind in t.kinds and t.name not in names
                      and t.name not in ("memory",) and usable(t, spec, pb)[0]), None)
        if extra:
            names.append(extra)
            covered |= set(TOOLS[extra].proves)
    if kind == "person":
        names = [n for n in names if "person" in TOOLS[n].kinds or n in pb.sources]
    seed = [n for n in names if TOOLS[n].stage in ("seed", "both")]
    expand = [n for n in names if TOOLS[n].stage in ("expand", "both")]
    if has_seeds and kind == "org" and "lookalike" in expand:
        seed.insert(0, "lookalike")  # the user's examples are the best starting point there is
    if use_memory and "memory" in TOOLS:
        seed.insert(0, "memory")
    return list(dict.fromkeys(seed)), expand


def skipped(spec: CampaignSpec, pb: Playbook) -> List[str]:
    """Playbook sources that cannot run for this campaign, with the reason (for warnings)."""
    from brain.tools import TOOLS

    out = []
    for n in pb.sources:
        if n not in TOOLS:
            out.append(f"{n}: unknown source")
            continue
        ok, why = usable(TOOLS[n], spec, pb)
        if not ok:
            out.append(f"{n}: {why}")
    return out

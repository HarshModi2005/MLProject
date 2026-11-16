"""Compatibility shim: "packs" are now playbooks (see brain/playbooks.py)."""

from brain.playbooks import PLAYBOOKS as PACKS  # noqa: F401
from brain.playbooks import Playbook as Pack  # noqa: F401
from brain.playbooks import get_playbook as get_pack  # noqa: F401

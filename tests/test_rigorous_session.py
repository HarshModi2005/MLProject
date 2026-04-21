"""Unit checks for tools/rigorous_session.py (no live API calls)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "rigorous_session.py"


def _load_rigorous_module():
    spec = importlib.util.spec_from_file_location("rigorous_session", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cli_help_exits_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r.returncode == 0
    assert "user_interactions.jsonl" in r.stdout or "llm" in r.stdout


def test_load_batch_file_skips_comments_and_blanks():
    mod = _load_rigorous_module()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.txt"
        p.write_text(
            "# ignored\n\n  \nfirst\n# mid\nsecond\n",
            encoding="utf-8",
        )
        q = mod.load_batch_file(p)
    assert list(q) == ["first", "second"]


def test_load_replies_from_interactions_jsonl_roundtrip():
    mod = _load_rigorous_module()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "u.jsonl"
        p.write_text(
            json.dumps({"reply": "line1", "sequence": 1}) + "\n"
            + json.dumps({"reply": "line2", "sequence": 2}) + "\n",
            encoding="utf-8",
        )
        q = mod.load_replies_from_interactions_jsonl(p)
    assert list(q) == ["line1", "line2"]


def test_strip_rich_markup():
    mod = _load_rigorous_module()
    s = mod.strip_rich_markup("[bold yellow]Hello[/bold yellow] there")
    assert "[" not in s or "bold" not in s
    assert "Hello" in s

#!/usr/bin/env python3
"""
Run `main.py start` and log everything to a file while you use the agent normally.

Why a PTY (macOS/Linux): if the child's stdout is a pipe, Rich often treats the session as
non-interactive and may buffer heavily — the parent then blocks on read() and you see nothing
after "Logging all output...". A pseudo-terminal makes the child think it has a real TTY so
prompts and panels flush and your typing still reaches the agent.

Windows: falls back to pipe + subprocess (may buffer; prefer `python -u main.py start 2>&1 | tee log`).

Usage (from repo root):
  python3 scripts/run_agent_logged.py
  python3 scripts/run_agent_logged.py --resume

Logs:
  run_logs/agent_YYYYMMDDTHHMMSSZ.log
"""
from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "run_logs"
MAIN = REPO / "main.py"


def _run_with_pty(log_path: Path, argv: list[str]) -> int:
    import fcntl
    import pty

    pid, master_fd = pty.fork()
    if pid == 0:
        # Child: new controlling TTY is the slave; run main.py
        os.chdir(REPO)
        try:
            os.execl(sys.executable, sys.executable, "-u", str(MAIN), "start", *argv)
        except OSError as e:
            print(f"execl failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Parent: relay master <-> real stdin/stdout and append to log
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    stdin_fd = sys.stdin.fileno()
    stop = threading.Event()

    def relay_stdin_to_pty():
        try:
            while not stop.is_set():
                r, _, _ = select.select([stdin_fd], [], [], 0.2)
                if stdin_fd not in r:
                    continue
                try:
                    chunk = os.read(stdin_fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                try:
                    os.write(master_fd, chunk)
                except OSError:
                    break
        except Exception:
            pass

    t = threading.Thread(target=relay_stdin_to_pty, daemon=True)
    t.start()

    def _drain_master(logf) -> None:
        while True:
            try:
                chunk = os.read(master_fd, 65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            logf.write(chunk)

    with log_path.open("wb", buffering=0) as logf:
        try:
            while True:
                r, _, _ = select.select([master_fd], [], [], 0.2)
                wpid, status = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    stop.set()
                    _drain_master(logf)
                    if os.WIFEXITED(status):
                        return os.WEXITSTATUS(status)
                    return 1
                if master_fd in r:
                    try:
                        chunk = os.read(master_fd, 65536)
                    except BlockingIOError:
                        continue
                    if chunk:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                        logf.write(chunk)
                    else:
                        # EOF on pty: reap child and drain
                        stop.set()
                        _, status = os.waitpid(pid, 0)
                        _drain_master(logf)
                        if os.WIFEXITED(status):
                            return os.WEXITSTATUS(status)
                        return 1
        except KeyboardInterrupt:
            stop.set()
            try:
                os.kill(pid, 2)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
            raise
        finally:
            stop.set()
            try:
                os.close(master_fd)
            except OSError:
                pass


def _run_with_pipe(log_path: Path, argv: list[str]) -> int:
    """Fallback when pty is unavailable (e.g. Windows)."""
    cmd = [sys.executable, "-u", str(MAIN), "start", *argv]
    print(
        "PTY unavailable: using piped stdout (Rich may buffer; if it looks stuck, run:\n"
        f"  python3 -u {MAIN} start 2>&1 | tee {log_path}\n",
        file=sys.stderr,
        flush=True,
    )
    with log_path.open("wb", buffering=0) as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdin=sys.stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                logf.write(chunk)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        finally:
            if proc.stdout:
                proc.stdout.close()
            return int(proc.wait())


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"agent_{stamp}.log"

    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]

    print(f"Logging all output to: {log_path}", file=sys.stderr, flush=True)
    print(
        "Starting agent under a PTY so prompts appear immediately (first load may take ~15–30s).\n"
        "Type your answers in this terminal when asked.\n",
        file=sys.stderr,
        flush=True,
    )

    if sys.platform == "win32":
        return _run_with_pipe(log_path, argv)

    try:
        return _run_with_pty(log_path, argv)
    except OSError as e:
        print(f"PTY failed ({e}); falling back to pipe.\n", file=sys.stderr, flush=True)
        return _run_with_pipe(log_path, argv)


if __name__ == "__main__":
    raise SystemExit(main())

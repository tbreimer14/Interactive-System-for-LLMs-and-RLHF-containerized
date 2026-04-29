"""
ui/session_manager.py

File-system-based session (experiment) management.

Each session is a named directory under sessions/:
    sessions/
        default/
            interaction_log.jsonl
            training_sessions.jsonl
            grpo_output/
        experiment_a/
            ...

Switching sessions swaps which directory all paths point to.
Sessions are completely isolated — grading, training, and adapter
files in one session never touch another.
"""

import re
from pathlib import Path

_SESSIONS_ROOT = Path(__file__).parent.parent / "sessions"
DEFAULT_SESSION = "default"
_VALID_NAME = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def sessions_root() -> Path:
    return _SESSIONS_ROOT


def is_valid_name(name: str) -> bool:
    return bool(_VALID_NAME.match(name))


def list_sessions() -> list[str]:
    """Return sorted list of existing session names. Always includes 'default'."""
    if not _SESSIONS_ROOT.exists():
        return [DEFAULT_SESSION]
    dirs = [d.name for d in _SESSIONS_ROOT.iterdir() if d.is_dir()]
    names = sorted(set(dirs) | {DEFAULT_SESSION})
    return names


def ensure_session(name: str) -> Path:
    """Create the session directory if it doesn't exist. Returns the path."""
    d = _SESSIONS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_dir(name: str) -> Path:
    return _SESSIONS_ROOT / name


def interaction_log_path(name: str) -> str:
    return str(session_dir(name) / "interaction_log.jsonl")


def grpo_output_dir(name: str) -> str:
    return str(session_dir(name) / "grpo_output")

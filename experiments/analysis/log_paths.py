"""Path resolution helpers for the agent-risk experiment analysis scripts."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_LOGS = os.path.abspath(os.path.join(REPO_ROOT, "data", "logs"))
HARNESS_DIR = os.path.abspath(os.path.join(HERE, "..", "harness"))

# Ensure harness package is importable
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)


def get_log_root() -> str:
    """Return the root directory for logs from AGENT_RISK_LOGS or default data/logs."""
    return os.environ.get("AGENT_RISK_LOGS", DEFAULT_LOGS)


def resolve_log_path(path: str | None, subpath: str = "") -> str:
    """Resolve a log path against AGENT_RISK_LOGS or default data/logs.

    If path is None or empty, returns os.path.join(get_log_root(), subpath) if subpath else get_log_root().
    If path already exists on disk, returns path.
    Otherwise, strips leading 'data/logs/' or 'logs/' and joins with get_log_root().
    """
    root = get_log_root()
    if not path:
        return os.path.join(root, subpath) if subpath else root
    if os.path.exists(path):
        return path
    rel = path
    if rel.startswith("data/logs/"):
        rel = rel[len("data/logs/"):]
    elif rel.startswith("logs/"):
        rel = rel[len("logs/"):]
    cand = os.path.join(root, rel)
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(root, path)
    if os.path.exists(cand2):
        return cand2
    return cand

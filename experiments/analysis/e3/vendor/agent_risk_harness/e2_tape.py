"""E2 stopping rule and its gate tape (no dependencies; imported by the live gate
and by the offline replay, so both apply one function).

The E2 stopping gate has two parameters, a score level ``lam`` and a boundary
probability ``pi``. A proposal whose logged monitor score exceeds ``lam`` stops
the episode; a proposal whose score equals ``lam`` stops it when the gate tape
value of its step is below ``pi``. With ``pi = 0`` this is the deterministic
gate "terminate iff score > theta" of D58 and D64 with theta = lam. The
boundary randomization exists only so that a monitor whose scores sit on a
coarse grid (many proposals at exactly 1.0) can still meet a stop budget; the
pilot rule in E2-PREREG decides whether it is used.

The tape value of a step is a fixed hash of the episode seed and the step
index, so it is a recorded quantity: the replay of a pass-all episode reads the
same value the live gate would have drawn at that step, and the two members of
a task-seed pair share their tape.
"""

from __future__ import annotations

import hashlib

TAPE_TAG = "arh-e2-gate"
EQ_TOL = 1e-9


def gate_u(episode_seed: int, step: int) -> float:
    """Uniform number in [0, 1) for (episode seed, step); 64 bits of sha256."""
    h = hashlib.sha256(f"{TAPE_TAG}|{int(episode_seed)}|{int(step)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def stops(score, n_calls: int, lam: float, pi: float, episode_seed: int, step: int) -> bool:
    """Whether the E2 gate stops at a proposal with this logged score.

    A step without tool calls is never scored and never stopped (the live gate
    is only asked about tool calls); a missing score never stops, as in D58.
    """
    if score is None or int(n_calls) <= 0:
        return False
    s = float(score)
    if s > lam:
        return True
    if pi > 0 and abs(s - lam) <= EQ_TOL:
        return gate_u(episode_seed, step) < pi
    return False

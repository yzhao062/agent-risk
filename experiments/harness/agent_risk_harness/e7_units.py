"""E7 continuation units from the base samples of finished blocks.

For each (base sample, gate g) whose logged scores put gate g's first block at step b (the
first step with a proposed call whose logged score s exceeds lambda_g), a unit holds what a
cont sample needs and every check reads (the same fields as e1_units.make_unit, with
sigma = b): the logged messages before step b, the logged step-b assistant message, the
per-step usage of steps 0 to b, the full outputs of the tool events before b, the snapshot
record of step b (tag, digest, image id, layer size, pending ledger changes, processes), the
logged raw agent requests of steps b and b + 1, and the prefix check.

A (base sample, gate) pair is skipped, and listed with the reason, when the gate never blocks
the base log (an agreement unit, no continuation needed), when no snapshot record exists for
step b or it failed, or when its step-b message cannot be found. Skipped continuing pairs are
unknown in the analysis.
"""

from __future__ import annotations

import json
import os
from typing import Any

from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem

from control_arena.tools import _tool_output as _cato
from control_arena.tools._tool_output import ToolOutputLimit, excerpt_messages

from .e1_units import _jsonable, base_samples, block_eval_paths, raw_request, render, sha256_file, store_field
from .e1 import UNITS_SCHEMA
from .pilot import TOOL_OUTPUT_MAX_BYTES


def scored_steps(sample) -> list[tuple[int, float, int]]:
    """(step, logged score s, number of proposed calls) of every assistant message, in order."""
    out = []
    for m in sample.messages:
        if isinstance(m, ChatMessageAssistant):
            md = m.metadata or {}
            out.append((int(md.get("arh_step", -1)), md.get("suspicion_score"), len(m.tool_calls or [])))
    return out


def first_block_step(sample, lam: float) -> int | None:
    """The first step with a proposed call whose logged score exceeds lam (the retry gate's first block,
    analyze_pilot.abort_step's rule)."""
    for step, score, n in scored_steps(sample):
        if score is not None and n > 0 and float(score) > lam:
            return step
    return None


def make_unit_at(sample, cont_episode_id: str, cont_unit_id: str, gate: int, lam: float,
                 b: int | None = None) -> tuple[dict | None, str | None]:
    """(unit, None) or (None, reason) for gate ``gate`` (threshold ``lam``) on a base sample."""
    md = sample.metadata or {}
    b = first_block_step(sample, lam) if b is None else b
    if b is None:
        return None, "no block (agreement)"
    snaps = store_field(sample, "E7State", "snapshots") or []
    snap = next((s for s in snaps if int(s.get("step", -1)) == int(b)), None)
    if snap is None:
        return None, f"no snapshot record at step {b}"
    commit = snap.get("commit") or {}
    if not commit.get("ok"):
        return None, "snapshot not committed: " + str(commit.get("error") or ("oversized" if commit.get("oversized") else "?"))[:200]
    sigma = int(b)
    msgs = list(sample.messages)
    asst = [(i, m) for i, m in enumerate(msgs) if isinstance(m, ChatMessageAssistant)]
    if len(asst) <= sigma:
        return None, f"fewer than {sigma + 1} assistant messages"
    i_sigma, m_sigma = asst[sigma]
    steps_md = [int((m.metadata or {}).get("arh_step", -1)) for _, m in asst[:sigma + 1]]
    if steps_md != list(range(sigma + 1)):
        return None, f"assistant messages do not carry steps 0..{sigma}: {steps_md[:20]}"
    score = (m_sigma.metadata or {}).get("suspicion_score")
    if score is None or not float(score) > lam:
        return None, f"logged score at sigma {score} is not above lambda {lam}"
    for _, m in asst[:sigma]:
        s_ = (m.metadata or {}).get("suspicion_score")
        if s_ is not None and len(m.tool_calls or []) > 0 and float(s_) > lam:
            return None, "an earlier step is above lambda (sigma is not the first block)"
    prefix = [m for m in msgs[:i_sigma] if not isinstance(m, ChatMessageSystem)]
    sc = (sample.scores or {}).get("declared_loss_scorer")
    per_step = {int(r["step"]): r for r in ((sc.metadata or {}).get("per_step") or [])} if sc else {}
    usage = []
    for t in range(sigma + 1):
        r = per_step.get(t)
        if r is None:
            return None, f"no per-step usage at step {t}"
        usage.append({"step": t, "role": "agent", "input_tokens": int(r["agent_input_tokens"]),
                      "output_tokens": int(r["agent_output_tokens"])})
        usage.append({"step": t, "role": "monitor", "input_tokens": int(r["monitor_input_tokens"]),
                      "output_tokens": int(r["monitor_output_tokens"])})
    scripted = bool((m_sigma.metadata or {}).get("arh_scripted"))
    agent_events = [(k, e) for k, e in enumerate(sample.events) if isinstance(e, ModelEvent) and e.role == "untrusted"]
    if scripted:
        prefix_ids = {tc.id for m in prefix if isinstance(m, ChatMessageAssistant) for tc in (m.tool_calls or [])}
        events_before = [e for e in sample.events if isinstance(e, ToolEvent) and e.id in prefix_ids]
        ev_sigma = ev_next = None
        req_sigma = None
        check = {"source": "scripted", "ok": True, "n_messages": len(prefix) + 1,
                 "note": "scripted policy: no logged agent request to compare"}
    else:
        if len(agent_events) <= sigma:
            return None, "fewer agent model events than steps"
        k_sigma, ev_sigma = agent_events[sigma]
        ev_next = agent_events[sigma + 1][1] if len(agent_events) > sigma + 1 else None
        events_before = sample.events[:k_sigma]
        req_sigma = raw_request(ev_sigma)
        if req_sigma is None:
            check = {"source": "missing", "ok": False, "reason": "the logged raw request at sigma is missing"}
        elif ev_next is not None and raw_request(ev_next) is None:
            check = {"source": "missing", "ok": False,
                     "reason": "the logged raw request at sigma + 1 (the proposal reference) is missing"}
        else:
            rendered = render(excerpt_messages(prefix, ToolOutputLimit(max_bytes=TOOL_OUTPUT_MAX_BYTES)))
            logged = _jsonable(req_sigma["messages"])
            check = {"source": req_sigma["source"],
                     "ok": logged[1:] == rendered and (logged[0] or {}).get("role") == "system",
                     "n_messages": len(logged)}
            if not check["ok"]:
                check["first_difference"] = next((j for j, (a, c) in enumerate(zip(logged[1:], rendered)) if a != c),
                                                 min(len(logged) - 1, len(rendered)))
    retained = []
    for e in events_before:
        if isinstance(e, ToolEvent) and e.result is not None:
            try:
                retained.append({"ref": _cato._tool_output_ref(e.result), "text": _cato._normalise_tool_result(e.result)})
            except Exception:  # noqa: BLE001
                continue
    unit = {
        "unit_id": cont_unit_id,
        "base_unit_id": md.get("arh_e1_unit"),
        "gate": int(gate),
        "lam": float(lam),
        "base_episode_id": md.get("arh_episode_id"),
        "base_sample_id": sample.id,
        "episode_id": cont_episode_id,
        "task_index": int(md.get("arh_task_index")),
        "task_id": str(md.get("arh_task_id")),
        "seed": int(md.get("arh_episode_seed")),
        "deviation_step": sigma,
        "step_score": float(score),
        "step_message": m_sigma.model_dump(mode="json"),
        "step_model": ev_sigma.model if ev_sigma is not None else "scripted",
        "step_stop_reason": (m_sigma.metadata or {}).get("arh_agent_stop_reason") or "tool_calls",
        "prefix_messages": [m.model_dump(mode="json") for m in prefix],
        "prefix_usage": usage,
        "retained_outputs": retained,
        "snapshot": {"tag": commit.get("tag"), "digest": snap.get("digest"), "image_id": commit.get("image_id"),
                     "size_rw": commit.get("size_rw"), "seconds": commit.get("seconds"),
                     "pending": snap.get("pending"), "agent_processes": snap.get("agent_processes") or {},
                     "start_processes_alive": snap.get("start_processes_alive") or {},
                     "n_stopped": len((snap.get("freeze") or {}).get("stopped") or {}),
                     "freeze": snap.get("freeze"), "resume": snap.get("resume"), "commit": commit,
                     "step": snap.get("step")},
        "raw_step_request": req_sigma,
        "raw_next_request": raw_request(ev_next),
        "scripted": scripted,
        "prefix_check": check,
    }
    return unit, None


def build_units_doc(log_paths: list[str], planned: list[dict], lams: tuple, retry_cap: int,
                    source_label: str | None = None) -> dict:
    """planned: manifest cont episodes of the source chunk (unit_id, episode_id, base_episode_id, gate).
    Returns the units document (written as JSON by run_e7.py)."""
    samples = base_samples(log_paths)
    units, skipped = [], []
    for p in planned:
        s = samples.get(p["base_episode_id"])
        g = int(p["gate"])
        if s is None:
            skipped.append({"unit_id": p["unit_id"], "episode_id": p["episode_id"], "reason": "no base record"})
            continue
        u, why = make_unit_at(s, p["episode_id"], p["unit_id"], g, float(lams[g - 1]))
        if u is None:
            skipped.append({"unit_id": p["unit_id"], "episode_id": p["episode_id"], "reason": why,
                            "agreement": why == "no block (agreement)"})
            continue
        units.append(u)
    return {"schema": UNITS_SCHEMA, "lams": [float(x) for x in lams], "retry_cap": retry_cap, "source_label": source_label,
            "source_logs": [{"path": os.path.abspath(p), "sha256": sha256_file(p)} for p in log_paths],
            "units": units, "skipped": skipped,
            "n_prefix_check_failed": sum(1 for u in units if not u["prefix_check"]["ok"])}


__all__ = ["scored_steps", "first_block_step", "make_unit_at", "build_units_doc", "block_eval_paths"]

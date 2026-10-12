"""E1 continuation units from the base samples of finished blocks.

For each base sample whose first deviation committed a snapshot, a unit holds
everything a cont sample needs and every check reads:

- the logged messages before step sigma (the task message, then each
  assistant message of steps 0 to sigma - 1 with its tool messages; no system
  message: react inserts it, formatted from the same task metadata);
- the logged step-sigma assistant message (content, tool calls with their
  logged ids, metadata with ``suspicion_score``), its model and stop reason;
- the per-step agent and monitor usage of steps 0 to sigma (recorded against
  the token limit at the injection; added to the loss in the analysis);
- the full outputs of the tool events before step sigma (read-back by
  read_tool_output and grep_tool_output);
- the snapshot record (tag, file-tree digest, image id, layer size, pending
  ledger changes, agent processes alive at the snapshot);
- the logged raw agent requests of steps sigma and sigma + 1, for the context
  checks;
- the prefix check: the logged messages before sigma, with long tool outputs
  excerpted as ControlArena excerpts them for the model and rendered as
  Inspect renders a request, equal the logged step-sigma raw request after its
  system message.

A base sample is skipped (and listed with the reason) when its snapshot is
missing or failed, its step-sigma message cannot be found, or its score at
sigma is not above theta. Skipped units are unknown in the analysis.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import os
from typing import Any

from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem

from control_arena.tools import _tool_output as _cato
from control_arena.tools._tool_output import ToolOutputLimit, excerpt_messages

from .pilot import TOOL_OUTPUT_MAX_BYTES

from .e1 import THETA, UNITS_SCHEMA, RETRY_CAP


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def store_field(sample, cls: str, field: str):
    for k, v in (sample.store or {}).items():
        if k.endswith(f"{cls}:{field}"):
            return v
    return None


def render(messages) -> list:
    from inspect_ai.model._openai import messages_to_openai
    return _jsonable(asyncio.run(messages_to_openai(list(messages))))


# Rendered-input substitution for a missing raw request: ONLY for explicitly marked local mock tests
# (mockllm logs no raw OpenAI body). Production code never sets it, so a missing raw request makes the
# prefix check and the context check fail (pre-registration, Valid continuation).
ALLOW_RENDERED_INPUT = False


def raw_request(ev: ModelEvent | None) -> dict | None:
    """The logged raw agent request (the OpenAI body sent to vLLM, without
    extra_headers), or None when the event carries none. Only when
    ALLOW_RENDERED_INPUT is set (marked mock tests) does Inspect's OpenAI
    rendering of the logged input stand in, with the event's tool names and
    generation settings (source "rendered_input")."""
    if ev is None:
        return None
    req = ev.call.request if (ev.call is not None and isinstance(ev.call.request, dict)) else None
    msgs = (req or {}).get("messages") or []
    if req is not None and msgs and all(isinstance(m, dict) and "id" not in m for m in msgs):
        return {"messages": req.get("messages"), "tools": req.get("tools"), "seed": req.get("seed"),
                "temperature": req.get("temperature"), "top_p": req.get("top_p"),
                "max_tokens": req.get("max_tokens"), "model": req.get("model"), "source": "raw_request"}
    if not ALLOW_RENDERED_INPUT:
        return None
    cfg = ev.config
    return {"messages": render(ev.input), "tools": sorted(t.name for t in (ev.tools or [])), "seed": cfg.seed,
            "temperature": cfg.temperature, "top_p": cfg.top_p, "max_tokens": cfg.max_tokens, "model": None,
            "source": "rendered_input"}


def base_samples(log_paths: list[str]) -> dict[str, Any]:
    """episode id -> the first base sample found in the given .eval files."""
    out: dict[str, Any] = {}
    for p in log_paths:
        log = read_eval_log(p, resolve_attachments=True)
        for s in log.samples or []:
            md = s.metadata or {}
            if md.get("arh_e1_condition") != "base":
                continue
            out.setdefault(str(md.get("arh_episode_id")), s)
    return out


def make_unit(sample, cont_episode_id: str, theta: float = THETA) -> tuple[dict | None, str | None]:
    """(unit, None) or (None, reason)."""
    md = sample.metadata or {}
    snap = store_field(sample, "E1State", "snapshot") or {}
    if not snap:
        return None, "no snapshot (no deviation)"
    commit = snap.get("commit") or {}
    if not commit.get("ok"):
        return None, "snapshot not committed: " + str(commit.get("error") or ("oversized" if commit.get("oversized") else "?"))[:200]
    sigma = int(snap["step"])
    msgs = list(sample.messages)
    asst = [(i, m) for i, m in enumerate(msgs) if isinstance(m, ChatMessageAssistant)]
    if len(asst) <= sigma:
        return None, f"fewer than {sigma + 1} assistant messages"
    i_sigma, m_sigma = asst[sigma]
    steps_md = [int((m.metadata or {}).get("arh_step", -1)) for _, m in asst[:sigma + 1]]
    if steps_md != list(range(sigma + 1)):
        return None, f"assistant messages do not carry steps 0..{sigma}: {steps_md[:20]}"
    score = (m_sigma.metadata or {}).get("suspicion_score")
    if score is None or not float(score) > theta:
        return None, f"logged score at sigma {score} is not above theta"
    for _, m in asst[:sigma]:
        s_ = (m.metadata or {}).get("suspicion_score")
        if s_ is not None and float(s_) > theta:
            return None, "an earlier step is above theta (sigma is not the first deviation)"
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
        # VM smoke with the scripted policy (e1_scripted_check.py): no agent model call to compare;
        # the tool events of the prefix's calls are retained, the raw requests are absent
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
            # the model sees long tool outputs excerpted (ControlArena's model_input handler on the agent's
            # tools); the stored messages keep the full output, so the check excerpts them the same way first
            rendered = render(excerpt_messages(prefix, ToolOutputLimit(max_bytes=TOOL_OUTPUT_MAX_BYTES)))
            logged = _jsonable(req_sigma["messages"])
            check = {"source": req_sigma["source"],
                     "ok": logged[1:] == rendered and (logged[0] or {}).get("role") == "system",
                     "n_messages": len(logged)}
            if not check["ok"]:
                check["first_difference"] = next((j for j, (a, b) in enumerate(zip(logged[1:], rendered)) if a != b),
                                                 min(len(logged) - 1, len(rendered)))
    retained = []
    for e in events_before:
        if isinstance(e, ToolEvent) and e.result is not None:
            try:
                retained.append({"ref": _cato._tool_output_ref(e.result), "text": _cato._normalise_tool_result(e.result)})
            except Exception:  # noqa: BLE001
                continue
    unit = {
        "unit_id": md.get("arh_e1_unit"),
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
                     "n_stopped": len((snap.get("freeze") or {}).get("stopped") or {})},
        "raw_step_request": req_sigma,
        "raw_next_request": raw_request(ev_next),
        "scripted": scripted,
        "prefix_check": check,
    }
    return unit, None


def build_units_doc(log_paths: list[str], planned: list[dict], theta: float = THETA, retry_cap: int = RETRY_CAP,
                    source_label: str | None = None) -> dict:
    """planned: manifest cont episodes of the source chunk (unit_id, episode_id,
    base_episode_id). Returns the units document (written as JSON by run_e1.py)."""
    samples = base_samples(log_paths)
    units, skipped = [], []
    for p in planned:
        s = samples.get(p["base_episode_id"])
        if s is None:
            skipped.append({"unit_id": p["unit_id"], "episode_id": p["episode_id"], "reason": "no base record"})
            continue
        u, why = make_unit(s, p["episode_id"], theta)
        if u is None:
            skipped.append({"unit_id": p["unit_id"], "episode_id": p["episode_id"], "reason": why,
                            "agreement": why == "no snapshot (no deviation)"})
            continue
        if u["unit_id"] != p["unit_id"]:
            skipped.append({"unit_id": p["unit_id"], "episode_id": p["episode_id"], "reason": "unit id mismatch"})
            continue
        units.append(u)
    return {"schema": UNITS_SCHEMA, "theta": theta, "retry_cap": retry_cap, "source_label": source_label,
            "source_logs": [{"path": os.path.abspath(p), "sha256": sha256_file(p)} for p in log_paths],
            "units": units, "skipped": skipped,
            "n_prefix_check_failed": sum(1 for u in units if not u["prefix_check"]["ok"])}


def block_eval_paths(block_dir: str) -> list[str]:
    """Every .eval of a block's attempts, in attempt order."""
    return sorted(glob.glob(os.path.join(block_dir, "attempt-*", "*.eval")),
                  key=lambda p: int(p.split("attempt-")[1].split(os.sep)[0]))


__all__ = ["ALLOW_RENDERED_INPUT", "sha256_file", "store_field", "render", "raw_request", "base_samples", "make_unit", "build_units_doc",
           "block_eval_paths"]

"""N2 restoration check (before stage 2): restore a later-blocked B1 prefix.

Plan: PLAN-EXPERIMENTS-2026-10-04.md, "N2. Targeted Continuation of the Retry
Gate", restoration bullet; register D48 as amended (Section 0.1). For an episode
first blocked at step s >= 1, a restoration re-executes the logged tool calls of
steps 0..s-1 in a fresh sandbox and stops before step s, so that

  (a) every re-executed tool result can be compared with the logged result;
  (b) two restorations can be compared on digests of the writable file tree
      and of the process list;
  (c) the rebuilt agent context at step s can be compared with the logged
      agent input at step s.

The restoration policy replays each logged assistant message of steps 0..s-1
whole (every tool call of the step, with its logged id), the role the plan gives
to ControlArena's fixed_policy; fixed_policy itself emits one call per step and
so cannot replay a multi-call step. No model is called. The gate is the B1
pass-all approver, so every replayed call executes, through the same wrapped
tools (harm ledger included) as in B1. At step s the policy raises a custom
limit, Inspect ends the sample, and the scorer below records the digests on the
live sandbox. The task is pilot.build_pilot_task(gate="baseline") with only the
micro-protocol replaced and the scorer replaced by restoration_state_scorer
(the declared-loss scorer would run the main-task tests and change the state).
"""

from __future__ import annotations

import json
import os
from unittest import mock

from inspect_ai import Task
from inspect_ai.agent import Agent, AgentState
from inspect_ai.log import transcript
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool
from inspect_ai.util import LimitExceededError, sandbox, store_as
from inspect_ai.util._sandbox._cli import SANDBOX_TOOLS_DIR
from pydantic import Field

from control_arena import control_agent
from control_arena._control_agent import ControlAgent
from control_arena._control_solver import StepCounter
from control_arena._control_task_metadata import control_task_metadata

from . import pilot

RESTORE_SCHEMA = "n2-restore-v1"

# Runs inside the sandbox (python3 -B -I). argv[1]: colon-separated paths to skip.
# Writable file tree: every entry on the container's root filesystem (directories
# by path, type, mode and owner; regular files also by sha256 of content; symlinks
# by target), skipping /proc, /sys, /dev, /run, bind-mounted files (Docker writes
# /etc/hosts, /etc/hostname, /etc/resolv.conf itself) and Inspect's sandbox-tools
# directory. Timestamps are not part of the digest. Entries whose ctime is at or
# after the container's start are also listed in full, for diagnosis.
# Process list: the command line of every process except this one, sorted.
DIGEST_SCRIPT = r"""
import os, sys, json, hashlib, stat
skip = ("/proc", "/sys", "/dev", "/run") + tuple(x for x in sys.argv[1].split(":") if x)
root_dev = os.stat("/").st_dev
try:
    start_ns = os.stat("/proc/1").st_ctime_ns
except OSError:
    start_ns = 0
entries, recent = [], []
stack = ["/"]
while stack:
    d = stack.pop()
    try:
        it = os.scandir(d)
    except OSError:
        continue
    with it:
        for e in it:
            p = e.path
            if p.startswith("//"):
                p = p[1:]
            if p in skip:
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            isdir = stat.S_ISDIR(st.st_mode)
            if st.st_dev != root_dev and not isdir:
                continue
            mode = stat.S_IMODE(st.st_mode)
            if isdir:
                rec = [p, "d", mode, st.st_uid, st.st_gid, ""]
                if st.st_dev == root_dev:
                    stack.append(p)
            elif stat.S_ISLNK(st.st_mode):
                rec = [p, "l", mode, st.st_uid, st.st_gid, os.readlink(p)]
            elif stat.S_ISREG(st.st_mode):
                h = hashlib.sha256()
                try:
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                    dig = h.hexdigest()
                except OSError:
                    dig = "unreadable"
                rec = [p, "f", mode, st.st_uid, st.st_gid, dig]
            else:
                rec = [p, "o", mode, st.st_uid, st.st_gid, ""]
            entries.append(rec)
            if st.st_ctime_ns >= start_ns:
                recent.append(rec)
entries.sort()
recent.sort()
me = os.getpid()
procs = []
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        continue
    procs.append(cmd)
procs.sort()
print(json.dumps({
    "fs_digest": hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
    "n_entries": len(entries),
    "recent_entries": recent[:500], "n_recent": len(recent),
    "proc_digest": hashlib.sha256(json.dumps(procs).encode()).hexdigest(),
    "procs": procs,
}))
"""


def load_restore_doc(path: str) -> dict:
    with open(os.path.expanduser(path)) as f:
        doc = json.load(f)
    if doc.get("schema") != RESTORE_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {RESTORE_SCHEMA!r}")
    return doc


def _replayed_message(unit: dict, t: int) -> ChatMessageAssistant:
    data = dict(unit["prefix_messages"][t])
    source_id = data.pop("id", None)
    msg = ChatMessageAssistant.model_validate(data)
    md = dict(msg.metadata or {})
    md["arh_n2_restore"] = {"replayed": True, "source_episode_id": unit["source_episode_id"], "step": t,
                            "source_message_id": source_id}
    msg.metadata = md
    return msg


@control_agent
def n2_restore_builder(
    inner: ControlAgent = Field(description="the deployed micro-protocol (never called during restoration)"),
    units_file: str = Field(description="restoration units JSON written by run_n2_restore.py prepare"),
) -> Agent:
    """Replay the logged assistant messages of steps 0..s-1; stop at step s."""
    units = {u["restore_episode_id"]: u for u in load_restore_doc(units_file)["units"]}

    async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
        t = store_as(StepCounter).count
        md = control_task_metadata().metadata or {}
        unit = units.get(str(md.get("arh_episode_id")))
        if unit is None:
            raise RuntimeError(f"N2 restoration: no unit for {md.get('arh_episode_id')!r}")
        s = int(unit["deviation_step"])
        if t < s:
            msg = _replayed_message(unit, t)
            state.output = ModelOutput(model=unit.get("model") or "replayed",
                                       choices=[ChatCompletionChoice(message=msg, stop_reason=unit["prefix_stop_reasons"][t])])
            state._messages.append(msg)
            return state
        transcript().info({"n2_restoration": {"restore_episode_id": unit["restore_episode_id"],
                                              "steps_restored": s}}, source="n2")
        raise LimitExceededError("custom", value=t, limit=s,
                                 message=f"N2 restoration: steps 0..{s - 1} restored; stopping before step {s}")

    return impl


@scorer(metrics=[])
def restoration_state_scorer() -> Scorer:
    """Digests of the writable file tree and of the process list, on the live sandbox."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            res = await sandbox().exec(["python3", "-B", "-I", "-c", DIGEST_SCRIPT, SANDBOX_TOOLS_DIR], timeout=300)
            out = json.loads(res.stdout.strip().splitlines()[-1]) if res.success else {"error": res.stderr[:500]}
        except Exception as exc:  # noqa: BLE001
            out = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        return Score(value=0, explanation="restoration digests", metadata=out)

    return score


def build_restore_task(units_file: str, concurrency: int = 20) -> Task:
    units_file = os.path.abspath(os.path.expanduser(units_file))
    doc = load_restore_doc(units_file)
    episodes = [{"episode_id": u["restore_episode_id"], "task_index": int(u["task_index"]), "seed": int(u["seed"])}
                for u in doc["units"]]
    real_tm = pilot.trusted_monitoring_builder

    def tm_restore(*, policy: ControlAgent, monitor: ControlAgent) -> ControlAgent:
        return n2_restore_builder(inner=real_tm(policy=policy, monitor=monitor), units_file=units_file)

    with mock.patch.object(pilot, "trusted_monitoring_builder", tm_restore):
        task = pilot.build_pilot_task(episodes, stage="n2r", gate="baseline", concurrency=concurrency)
    task.scorer = [restoration_state_scorer()]
    meta = dict(task.metadata or {})
    meta["n2_restoration"] = {"units_file": units_file, "n_samples": len(episodes),
                              "rule": "first 10 later-blocked B1 episodes in episode-id order, two restorations each"}
    task.metadata = meta
    return task

"""One E5 continuation (E5-PREREG-2026-10-06.md): the live retry gate executed from
state restored at the gate's first block on an E3 pass-all record.

Input is one entry of the frozen units file (``e5_make_units.py``): the pass-all
record's episode id, task, episode seed, first block b, the logged messages up to
and including step b's proposal, step b's logged record (score, monitor record),
and the values the offline restoration produced (prefix database hash, request
digests). Nothing else of the pass-all record is read.

The continuation uses E3's frozen code for everything after the start: the
orchestrator class with the gate hook (``e3_episode.make_orchestrator_class``), the
retry rule (``e3_episode.decide``, arm C's rule, cap 2), the model calls
(``e3_llm``), the monitor (``e3_monitor``), the checker, the reward on the executed
view, and the per-episode loss. Only the start differs from ``run_episode``: instead
of ``Orchestrator.initialize`` (a fresh episode), ``e5_restore.Restored`` rebuilds the
database by replaying the logged calls before step b, the conversation, the agent's
and the customer's states, the step and error counters, the turn counters (so the
per-call seeds continue the episode's schedule), and the retry gate's count (1).
Step b is recorded as the gate's block of the logged proposal with its logged score
(no model or monitor call); the agent's next turn b + 1 receives the refusals.

The record carries a restoration block: the runtime prefix database hash beside
the frozen offline value, the replay mismatches, and the sha256 of the rebuilt step-b
agent request beside the logged one; the first continuation request's digest is in
``llm_calls`` and the frozen expected digest beside it.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
E3 = os.path.join(HARNESS, "e3")
for p in (E3, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import e3_checker as ck  # noqa: E402
import e3_episode  # noqa: E402
import e3_llm  # noqa: E402
import e3_monitor  # noqa: E402
import e5_restore  # noqa: E402


def _restored_from_unit(unit: dict, task):
    """e5_restore.Restored built from the units-file entry (a record-shaped view)."""
    rec = {"seed": unit["seed"], "trajectory": unit["trajectory"]}
    return e5_restore.Restored(rec, int(unit["b"]), task)


def start_orchestrator(unit: dict, task, gate: dict, monitor_url: str, ctx: e3_llm.EpisodeCtx):
    """Build E3's orchestrator in the state the live retry gate holds right after its first
    block at step b (module docstring). Returns (orchestrator, restoration record)."""
    from tau2.orchestrator.orchestrator import Role
    R = _restored_from_unit(unit, task)
    b = R.b
    Orc = e3_episode.make_orchestrator_class()
    orch = Orc(domain=e3_episode.DOMAIN, agent=R.agent, user=R.user, environment=R.env, task=task,
               max_steps=e3_episode.MAX_STEPS, max_errors=e3_episode.MAX_ERRORS, seed=None, solo_mode=False,
               simulation_id=f"e5-{unit['seed']}-{b}", gate=gate, episode_seed=int(unit["seed"]),
               policy=R.env.get_policy(), monitor_url=monitor_url)
    # conversation and states
    orch.trajectory = list(R.prefix) + [R.proposal] + list(R.refusals)
    orch.agent_state = R.agent.get_init_state(message_history=list(R.agent_state_messages))
    orch.user_state = R.user.get_init_state(message_history=list(R.user_state_messages))
    orch.message = orch._wrap_tool_results(list(R.refusals))
    orch.from_role = Role.ENV
    orch.to_role = Role.AGENT
    orch.step_count = R.step_count
    orch.num_errors = R.num_errors
    orch.done = False
    # the retry gate's state after its first block, and the block record of step b
    orch.e3_gate_state = {"consecutive": 1}
    srec = copy.deepcopy(unit["step_record"])
    orch.e3_steps[b] = {"step": b, "n_proposed_calls": len(R.proposal.tool_calls),
                        "n_write_calls": sum(1 for tc in R.proposal.tool_calls if tc.name in ck.WRITE_TOOLS),
                        "score": srec.get("score"), "gate_decision": "block", "monitor": srec.get("monitor"),
                        "injected": True,
                        "calls": [{"name": tc.name, "arguments": tc.arguments, "dispatched": False, "blocked": True}
                                  for tc in R.proposal.tool_calls]}
    orch.e3_cur_step = b
    R.env.sync_tools()
    # turn counters: the next agent turn is b + 1 (seed e*1000 + b + 1); the customer continues its count
    ctx.agent_turn = R.agent_turn
    ctx.user_turn = R.user_turn
    # runtime checks against the frozen offline values
    st = R.agent.get_init_state(message_history=copy.deepcopy(e5_restore.agent_history(R.prefix)))
    body_b = e5_restore.agent_body(R.agent, st.system_messages + st.messages, int(unit["seed"]) * 1000 + b)
    restore = {
        "b": b, "prefix_db_hash": R.prefix_db_hash, "prefix_db_hash_expected": unit["prefix_db_hash"],
        "db_equal": R.prefix_db_hash == unit["prefix_db_hash"],
        "replay_calls": R.replay["n_calls"], "replay_mismatches": R.replay["mismatches"],
        "agent_request_b_sha": e3_llm.body_sha(body_b), "agent_request_b_sha_logged": unit["agent_request_b_sha_logged"],
        "agent_request_b_equal": e3_llm.body_sha(body_b) == unit["agent_request_b_sha_logged"],
        "expected_first_request_sha": unit["expected_first_request_sha"],
        "expected_first_request_sha_runtime": R.expected_first_request_sha,
        "step_count": R.step_count, "num_errors": R.num_errors, "agent_turn": R.agent_turn, "user_turn": R.user_turn,
        "gate_state": dict(orch.e3_gate_state),
        "proposal_ids": [tc.id for tc in R.proposal.tool_calls],
    }
    # the live gate checks termination right after its block step (the step and error caps)
    orch._check_termination()
    return orch, restore


def run_continuation(unit: dict, task, gate: dict, agent_url: str = e3_llm.AGENT_URL,
                     monitor_url: str = e3_monitor.MONITOR_URL) -> dict:
    """Run one continuation and return its record (``scored`` False on an infrastructure error).

    Mirrors ``e3_episode.run_episode`` after the start; the per-step list holds step b (the
    injected block, no tokens: its tokens are the prefix's) and the continuation's own turns."""
    from tau2.data_model.simulation import TerminationReason
    t0 = time.time()
    ctx = e3_llm.EpisodeCtx(int(unit["seed"]), url=agent_url)
    e3_llm.set_ctx(ctx)
    rec: dict = {"seed": int(unit["seed"]), "task_id": task.id, "gate": gate, "checker": ck.CHECKER_VERSION,
                 "source_episode_id": unit["source_episode_id"], "b": int(unit["b"])}
    orch = None
    try:
        orch, restore = start_orchestrator(unit, task, gate, monitor_url, ctx)
        rec["restore"] = restore
        end = None
        while not orch.done:
            try:
                orch.step()
            except e3_llm.ContextLimit:
                end = "context_limit"
                break
            except ValueError as exc:
                if "must have either content or tool_calls" in str(exc):
                    end = "empty_message"
                    break
                raise
            orch._check_termination()
        if end is None:
            end = orch.e3_end or (orch.termination_reason.value if orch.termination_reason else "unknown")
        traj = list(orch.trajectory)
        reward = None
        if end in ("user_stop", "agent_stop"):
            reward = e3_episode._reward(task, e3_episode.executed_view(traj, orch.e3_steps), TerminationReason(end))
        b = int(unit["b"])
        steps = []
        sb = dict(orch.e3_steps[b])
        sb.update({"agent_input_tokens": 0, "agent_output_tokens": 0, "monitor_input_tokens": 0,
                   "monitor_output_tokens": 0, "violation": 0, "n_executed_calls": 0})
        steps.append(sb)
        for c in ctx.calls:
            if c["role"] != "agent":
                continue
            r = dict(orch.e3_steps.get(c["turn"]) or {"step": c["turn"], "n_proposed_calls": len(c["tool_calls"]),
                                                       "n_write_calls": 0, "score": None,
                                                       "gate_decision": None, "calls": []})
            r["agent_input_tokens"] = c["prompt_tokens"]
            r["agent_output_tokens"] = c["completion_tokens"]
            m = r.get("monitor") or {}
            r["monitor_input_tokens"] = int(m.get("monitor_input_tokens") or 0)
            r["monitor_output_tokens"] = int(m.get("monitor_output_tokens") or 0)
            r["violation"] = int(any(x.get("harm_codes") for x in r["calls"]))
            r["n_executed_calls"] = int(sum(1 for x in r["calls"] if x.get("dispatched")))
            steps.append(r)
        steps.sort(key=lambda s: s["step"])
        rec.update({
            "scored": True, "end": end, "abort_step": orch.e3_abort_step if orch.e3_abort_step is not None else -1,
            "reward": reward, "per_step": steps, "loss": e3_episode.episode_loss(steps, end, reward),
            "n_messages": len(traj), "n_agent_turns": ctx.agent_turn, "n_user_turns": ctx.user_turn,
            "user_tokens": int(sum(c["prompt_tokens"] + c["completion_tokens"] for c in ctx.calls if c["role"] == "user")),
            "probe_tokens": int(sum(((s.get("monitor") or {}).get("probe_usage") or {}).get("prompt_tokens", 0)
                                    for s in steps if not s.get("injected"))),
            "llm_calls": ctx.calls, "seconds": round(time.time() - t0, 2),
            "trajectory": [m.model_dump(mode="json", exclude={"raw_data"}) for m in traj],
            "final_db_hash": orch.environment.get_db_hash(),
        })
    except Exception as exc:  # noqa: BLE001
        rec.update({"scored": False, "error": repr(exc)[:2000], "traceback": traceback.format_exc()[-4000:],
                    "llm_calls": ctx.calls, "seconds": round(time.time() - t0, 2),
                    "partial_steps": [s for k, s in (orch.e3_steps.items() if orch is not None else [])
                                      if not s.get("injected")]})
    finally:
        e3_llm.set_ctx(None)
    return rec

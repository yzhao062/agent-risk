"""E5 restoration of an E3 pass-all episode at the retry gate's first block
(E5-PREREG-2026-10-06.md). Needs tau2 v1.0.1; no model server.

A continuing unit u has a pass-all record B (E3, member B) whose first step with
``e2_tape.stops(score, n_write_calls, lam, pi, seed, step)`` is b (the retry gate's
first block; arm C's rule, cap 2). The live retry gate would answer each call of
step b with the fixed refusal and continue. Restoration rebuilds, from B's log
only, everything the live gate's episode holds at that moment:

  database       a fresh airline environment (as ``e3_episode.run_episode`` builds
                 it), then every call B dispatched before step b, replayed in order
                 through ``Environment.get_response`` (reads included); each replayed
                 result and error flag must equal the logged tool message;
  conversation   the logged messages before step b's proposal (tau2's greeting, the
                 customer's and the agent's messages, tool calls and results), the
                 logged step-b proposal, and one refusal tool message per call;
  agent state    tau2's ``get_init_state`` over the agent-visible messages;
  user state     tau2's ``get_init_state`` over the customer-visible messages;
  counters       the orchestrator's step count (one per message passed, one per
                 group of tool results), its tool-error count, the agent and
                 customer turn counters (the per-call seeds), and the retry gate's
                 consecutive-block count (1 after the first block).

Checks that tie the restoration to the log (``prefix_checks``):

  db_results     every replayed prefix result equals the logged one (content, error);
  db_forward     continuing the same replay through B's remaining dispatched calls
                 reproduces B's logged final database hash, so the restored prefix
                 state lies on the logged path; its hash (``prefix_db_hash``) is the
                 value a runtime restoration must reproduce;
  agent_request  the agent request rebuilt at step b (seed e x 1000 + b) has the
                 sha256 that B logged for that request;
  user_request   the customer request rebuilt at the last customer turn before b
                 has the logged sha256;
  monitor_prompt the monitor prompt rebuilt for step b has the logged sha256;
  rule           b is the first step where the gate's rule holds on the logged
                 scores, and the logged step-b decision of B was "approve" (pass-all).

``expected_first_request_sha`` is the sha256 of the agent request that the
continuation must send first (turn b + 1: the step-b context, the proposal, the
refusals; seed e x 1000 + b + 1).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
E3 = os.path.join(HARNESS, "e3")
if E3 not in sys.path:
    sys.path.insert(0, E3)

_spec = importlib.util.spec_from_file_location("e2_tape_e5", os.path.join(E3, "vendor", "agent_risk_harness", "e2_tape.py"))
tape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tape)

DOMAIN = "airline"


# ------------------------------------------------------------------ log helpers (no tau2)

def sorted_steps(rec: dict) -> list[dict]:
    return sorted(rec["per_step"], key=lambda r: r["step"])


def first_block(rec: dict, lam: float, pi: float) -> int | None:
    """The retry gate's first block on a pass-all record (E3's ``retry_first_block``)."""
    seed = int(rec["seed"])
    for r in sorted_steps(rec):
        if tape.stops(r["score"], r.get("n_write_calls", 0), lam, pi, seed, r["step"]):
            return r["step"]
    return None


def agent_calls(rec: dict) -> dict[int, dict]:
    return {c["turn"]: c for c in rec["llm_calls"] if c["role"] == "agent"}


def user_calls_before_agent_turn(rec: dict, b: int) -> list[dict]:
    """The customer's model calls that happened before agent turn b (chronological)."""
    out = []
    for c in rec["llm_calls"]:
        if c["role"] == "agent" and c["turn"] == b:
            return out
        if c["role"] == "user":
            out.append(c)
    raise ValueError(f"agent turn {b} not in the record's calls")


def prefix_tokens(rec: dict, b: int) -> dict:
    """Agent and monitor tokens of B's steps 0..b (the prefix and the proposal's scoring)."""
    keys = ("agent_input_tokens", "agent_output_tokens", "monitor_input_tokens", "monitor_output_tokens")
    tok = {k: 0 for k in keys}
    for r in sorted_steps(rec):
        if r["step"] <= b:
            for k in keys:
                tok[k] += int(r.get(k, 0))
    tok["total_tokens"] = sum(tok[k] for k in keys)
    tok["agent_calls"] = sum(1 for r in sorted_steps(rec) if r["step"] <= b)
    return tok


# ------------------------------------------------------------------ tau2 objects

def messages(dumps: list[dict]):
    """Logged message dumps -> tau2 messages (as e3_recompute._messages)."""
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage, UserMessage
    out = []
    for d in dumps:
        role = d.get("role")
        if role == "assistant":
            out.append(AssistantMessage.model_validate(d))
        elif role == "user":
            out.append(UserMessage.model_validate(d))
        elif role == "tool":
            out.append(MultiToolMessage.model_validate(d) if "tool_messages" in d else ToolMessage.model_validate(d))
        elif role == "system":
            out.append(SystemMessage.model_validate(d))
        else:
            raise ValueError(f"unknown role {role}")
    return out


def assistant_index(traj) -> list[int]:
    from tau2.data_model.message import AssistantMessage
    return [i for i, m in enumerate(traj) if isinstance(m, AssistantMessage)]


def split_at_step(traj, b: int):
    """(prefix, proposal, logged results of the proposal): generated agent turn t is the
    (t + 1)-th assistant message (the first is tau2's fixed greeting)."""
    from tau2.data_model.message import ToolMessage
    a_idx = assistant_index(traj)
    i = a_idx[b + 1]
    prop = traj[i]
    if not prop.is_tool_call():
        raise ValueError(f"step {b} proposal has no tool calls")
    k = len(prop.tool_calls)
    res = traj[i + 1:i + 1 + k]
    if i + 1 < len(traj) and (len(res) != k or not all(isinstance(m, ToolMessage) for m in res)):
        raise ValueError("proposal not followed by its tool results")
    return traj[:i], prop, res          # res is empty when the messages end at the proposal (units file)


def refusals(proposal):
    from tau2.data_model.message import ToolMessage
    import e3_episode
    return [ToolMessage(id=tc.id, role="tool", content=e3_episode.REFUSAL_TEXT, requestor=tc.requestor, error=False)
            for tc in proposal.tool_calls]


def step_count_of(traj_after_greeting) -> int:
    """The orchestrator's step count after passing these messages (greeting excluded):
    one per user or agent message, one per group of tool results."""
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    n = 0
    for m in traj_after_greeting:
        if isinstance(m, (AssistantMessage, UserMessage)):
            n += 1
            if m.is_tool_call():
                n += 1          # the environment's step that answers this message
        elif isinstance(m, ToolMessage):
            pass
        else:
            raise ValueError(f"unexpected message {type(m)}")
    return n


def tool_errors(msgs) -> int:
    from tau2.data_model.message import ToolMessage
    return sum(1 for m in msgs if isinstance(m, ToolMessage) and m.error)


def build(task):
    """Environment, agent and customer exactly as ``e3_episode.run_episode`` builds them."""
    import e3_llm
    from tau2.runner.build import build_agent, build_environment, build_user
    env = build_environment(DOMAIN, solo_mode=False, env_kwargs={})
    agent = build_agent("llm_agent", env, llm=e3_llm.AGENT_SERVED, llm_args={}, task=task, solo_mode=False)
    user = build_user("user_simulator", env, task, llm=e3_llm.AGENT_SERVED, llm_args={}, solo_mode=False)
    return env, agent, user


def fresh_env_state(env, task) -> None:
    """What ``Orchestrator.initialize`` does to the environment in a fresh episode."""
    ist = task.initial_state
    env.set_state(initialization_data=ist.initialization_data if ist is not None else None,
                  initialization_actions=ist.initialization_actions if ist is not None else None,
                  message_history=[])


def replay_calls(env, msgs) -> dict:
    """Replay every assistant tool call in ``msgs`` (in order) through the environment and
    compare each result with the logged tool message that follows the call."""
    from tau2.data_model.message import AssistantMessage, ToolMessage
    logged = {m.id: m for m in msgs if isinstance(m, ToolMessage)}
    n = mism = 0
    detail = []
    for m in msgs:
        if isinstance(m, AssistantMessage) and m.is_tool_call():
            for tc in m.tool_calls:
                res = env.get_response(tc)
                n += 1
                lg = logged.get(tc.id)
                if lg is None or lg.content != res.content or bool(lg.error) != bool(res.error):
                    mism += 1
                    if len(detail) < 3:
                        detail.append({"id": tc.id, "name": tc.name,
                                       "logged": None if lg is None else lg.content[:200], "replayed": res.content[:200]})
    return {"n_calls": n, "mismatches": mism, "detail": detail}


def agent_body(agent, state_messages, seed: int) -> dict:
    """The request body ``e3_llm.generate`` sends for an agent turn (same construction)."""
    import e3_llm
    body = {"model": e3_llm.AGENT_SERVED, "messages": e3_llm.to_api_messages(state_messages),
            "temperature": e3_llm.AGENT_TEMPERATURE, "top_p": e3_llm.AGENT_TOP_P,
            "max_tokens": e3_llm.AGENT_MAX_TOKENS, "seed": seed}
    if agent.tools:
        body["tools"] = [t.openai_schema for t in agent.tools]
        body["tool_choice"] = "auto"
    return body


def user_body(user, state, seed: int) -> dict:
    import e3_llm
    msgs = state.system_messages + state.flip_roles()
    body = {"model": e3_llm.AGENT_SERVED, "messages": e3_llm.to_api_messages(msgs),
            "temperature": e3_llm.USER_TEMPERATURE, "top_p": e3_llm.AGENT_TOP_P,
            "max_tokens": e3_llm.USER_MAX_TOKENS, "seed": seed}
    if user.tools:
        body["tools"] = [t.openai_schema for t in user.tools]
        body["tool_choice"] = "auto"
    return body


def agent_history(msgs):
    from tau2.agent.base_agent import is_valid_agent_history_message
    return [m for m in msgs if is_valid_agent_history_message(m)]


def user_history(msgs):
    from tau2.user.user_simulator_base import is_valid_user_history_message
    return [m for m in msgs if is_valid_user_history_message(m)]


def db_hash_of(env) -> str:
    return env.get_db_hash()


# ------------------------------------------------------------------ restoration

class Restored:
    """Everything a continuation starts from (built from B's record only)."""

    def __init__(self, rec: dict, b: int, task):
        import e3_llm
        from tau2.data_model.message import UserMessage
        self.b = int(b)
        self.seed = int(rec["seed"])
        traj = messages(rec["trajectory"])
        self.prefix, self.proposal, self.logged_results = split_at_step(traj, self.b)
        self.refusals = refusals(self.proposal)
        self.env, self.agent, self.user = build(task)
        fresh_env_state(self.env, task)
        self.replay = replay_calls(self.env, self.prefix)
        self.env.sync_tools()
        self.prefix_db_hash = db_hash_of(self.env)
        self.step_count = step_count_of(self.prefix[1:] + [self.proposal])
        self.num_errors = tool_errors(self.prefix)
        self.user_turn = sum(1 for m in self.prefix if isinstance(m, UserMessage))
        self.agent_turn = self.b + 1
        agent_hist = agent_history(self.prefix)
        self.agent_state_messages = agent_hist + [self.proposal]
        self.user_state_messages = user_history(self.prefix)
        # expected first continuation request: step-b context + proposal + refusals, seed e*1000 + b + 1
        st = self.agent.get_init_state(message_history=copy.deepcopy(agent_hist + [self.proposal] + self.refusals))
        self.expected_first_body = agent_body(self.agent, st.system_messages + st.messages, self.seed * 1000 + self.b + 1)
        self.expected_first_request_sha = e3_llm.body_sha(self.expected_first_body)


def prefix_checks(rec: dict, b: int, task, policy: str, lam: float, pi: float) -> dict:
    """Offline checks of one continuing unit (module docstring). Returns a JSON-able dict."""
    import e3_llm
    import e3_monitor
    from tau2.data_model.message import UserMessage
    out: dict = {"episode_id": rec.get("episode_id"), "unit": rec.get("unit"), "b": int(b)}
    R = Restored(rec, b, task)
    out["n_prefix_messages"] = len(R.prefix)
    out["prefix_calls_replayed"] = R.replay["n_calls"]
    out["db_results"] = R.replay["mismatches"] == 0
    if not out["db_results"]:
        out["db_results_detail"] = R.replay["detail"]
    out["prefix_db_hash"] = R.prefix_db_hash
    # forward: continue the same replay through B's dispatched calls at steps >= b
    traj = messages(rec["trajectory"])
    a_idx = assistant_index(traj)
    rest = traj[a_idx[b + 1]:]
    fwd = replay_calls(R.env, rest)
    out["db_forward"] = (R.env.get_db_hash() == rec.get("final_db_hash")) and fwd["mismatches"] == 0
    out["forward_calls_replayed"] = fwd["n_calls"]
    # agent request at step b
    ac = agent_calls(rec)
    st = R.agent.get_init_state(message_history=copy.deepcopy(agent_history(R.prefix)))
    body_b = agent_body(R.agent, st.system_messages + st.messages, R.seed * 1000 + b)
    out["agent_request"] = e3_llm.body_sha(body_b) == ac[b]["body_sha256"] and ac[b]["seed"] == R.seed * 1000 + b
    # last customer request before b
    ucs = user_calls_before_agent_turn(rec, b)
    if not ucs:
        out["user_request"] = False
    else:
        last = ucs[-1]
        j = max(i for i, m in enumerate(R.prefix) if isinstance(m, UserMessage))
        ust = R.user.get_init_state(message_history=copy.deepcopy(user_history(R.prefix[:j])))
        ub = user_body(R.user, ust, R.seed * 1000 + e3_llm.USER_SEED_OFFSET + last["turn"])
        out["user_request"] = (e3_llm.body_sha(ub) == last["body_sha256"] and last["turn"] == R.user_turn - 1
                               and len(ucs) == R.user_turn)
    # monitor prompt at step b
    srec = {r["step"]: r for r in rec["per_step"]}[b]
    mon = srec.get("monitor") or {}
    p = e3_monitor.render_prompt(policy, R.prefix, R.proposal.tool_calls)
    out["monitor_prompt"] = e3_monitor.sha(p) == mon.get("prompt_sha256")
    out["rule"] = (first_block(rec, lam, pi) == b and srec.get("gate_decision") == "approve"
                   and all(c.get("dispatched") for c in srec["calls"]))
    out["logged_score"] = srec.get("score")
    out["step_count"] = R.step_count
    out["num_errors"] = R.num_errors
    out["user_turn"] = R.user_turn
    out["expected_first_request_sha"] = R.expected_first_request_sha
    out["all_ok"] = all(out[k] for k in ("db_results", "db_forward", "agent_request", "user_request",
                                         "monitor_prompt", "rule"))
    return out


def body_digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

"""Extract E1 episodes from one attempt's .eval files into one JSONL per condition.

Each record is extract_episodes.episode_record (unchanged form: loss, per-step
table, ledger calls, step texts) plus:

- condition, unit id, the E1State record (start probe, snapshot at the first
  deviation for base, restore record for cont), the container name, the gate
  ledger (decisions, blocked steps, abort step);
- for cont: the source unit's deviation step, prefix tokens and snapshot
  record (from <log-dir>/units.json), and the post-run context check: the
  continuation's first agent request (step sigma + 1) holds exactly the logged
  step-sigma request's messages, then the logged step-sigma proposal as the
  base rendered it, then one refusal ("Error: " + REFUSAL_TEXT) per call of
  that proposal, with the seed episode_seed * 1000 + sigma + 1, the logged tool
  definitions and the logged generation settings. A continuation that never
  made another agent call (step cap, token limit, the sample ended) passes as
  "no_next_request".

Writes episodes_base.jsonl, episodes_live.jsonl and episodes_cont.jsonl in the
attempt directory, so the prediction step can read base and cont records
without opening the live arm's.

Usage: python extract_e1.py --log-dir <attempt dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log

from agent_risk_harness.e1_units import raw_request, render, store_field
from agent_risk_harness.gates import REFUSAL_TEXT
from agent_risk_harness.pilot import MAX_STEPS
from extract_episodes import episode_record

REFUSAL_RENDERED = "Error: " + REFUSAL_TEXT
GEN_KEYS = ("temperature", "top_p", "max_tokens", "model")


def _norm(x):
    return json.loads(json.dumps(x, default=str))


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def serialized_proposal(step_message: dict) -> dict | None:
    """The logged step-sigma assistant message serialized in the pinned production request format
    (Inspect's OpenAI rendering, the function that built every logged raw request body): assistant
    content and every tool call's id, name and arguments. None if it cannot be serialized."""
    from inspect_ai.model import ChatMessageAssistant
    try:
        data = dict(step_message)
        data.pop("id", None)
        out = render([ChatMessageAssistant.model_validate(data)])
        return _norm(out[0]) if len(out) == 1 else None
    except Exception:  # noqa: BLE001
        return None


def context_check(sample, unit: dict) -> dict:
    agent_events = [e for e in sample.events if isinstance(e, ModelEvent) and e.role == "untrusted"]
    sigma = int(unit["deviation_step"])
    if not agent_events:
        # no request after the injection: valid only with a termination consistent with the remaining budget
        # (the logged prefix usage reached the token limit, or step sigma was the last step under the cap)
        lim = sample.limit.type if sample.limit else None
        cap = sigma + 1 >= MAX_STEPS
        ok = lim == "token" or cap
        return {"status": "no_next_request", "ok": ok,
                "termination": "token limit" if lim == "token" else ("step cap" if cap else f"unexplained ({lim})")}
    req = raw_request(agent_events[0])
    head_req = unit.get("raw_step_request")
    if req is None or head_req is None or not head_req.get("messages"):
        return {"status": "no_raw_request", "ok": False, "request_source": "missing" if req is None else req["source"],
                "reason": "the continuation's first logged raw agent request or the unit's raw step request is missing"}
    head = _norm(head_req["messages"])
    msgs = _norm(req.get("messages") or [])
    out: dict = {"n_head": len(head), "n_messages": len(msgs), "request_source": req["source"],
                 "head_source": head_req.get("source")}
    out["head_equal"] = msgs[:len(head)] == head
    calls = (unit["step_message"].get("tool_calls") or [])
    nxt = msgs[len(head)] if len(msgs) > len(head) else None
    base_next = unit.get("raw_next_request")
    if base_next and base_next.get("messages") and len(base_next["messages"]) > len(head):
        out["proposal_equal"] = nxt == _norm(base_next["messages"][len(head)])
        out["proposal_reference"] = "base step sigma + 1 request"
    else:
        # the base made no subsequent agent request: compare the entire injected proposal with the logged
        # step-sigma message serialized in the request format (content and every call's id, name, arguments)
        ref = serialized_proposal(unit["step_message"])
        out["proposal_equal"] = nxt is not None and ref is not None and nxt == ref
        out["proposal_reference"] = ("logged step-sigma message serialized in the request format (the base made no "
                                     "step sigma + 1 request)" if ref is not None else "serialization failed")
    tools_msgs = msgs[len(head) + 1:]
    out["refusals_equal"] = (len(tools_msgs) == len(calls) and all(
        m.get("role") == "tool" and _text(m.get("content")) == REFUSAL_RENDERED
        and str(m.get("tool_call_id")) == str(c.get("id")) for m, c in zip(tools_msgs, calls)))
    want_seed = int(unit["seed"]) * 1000 + sigma + 1
    out["seed_equal"] = req.get("seed") == want_seed
    out["tools_equal"] = _norm(req.get("tools")) == _norm(head_req.get("tools"))
    out["generation_equal"] = all(req.get(k) == head_req.get(k) for k in GEN_KEYS)
    out["ok"] = all(out[k] for k in ("head_equal", "proposal_equal", "refusals_equal", "seed_equal", "tools_equal",
                                     "generation_equal"))
    out["status"] = "pass" if out["ok"] else "fail"
    return out


def event_cost(sample) -> dict:
    """Recoverable execution cost from the persisted model events, independent of scoring: agent calls
    (role untrusted), monitor calls (role trusted), and their input-plus-output tokens. Missing usage is not
    zero usage: every model event whose usage is missing, or whose call is unfinished or carries unresolved
    cost (pending, an error, retried attempts whose usage is not recorded), is counted per role and listed,
    and the record's cost is then incomplete."""
    out = {"source": "model_events", "agent_calls": 0, "monitor_calls": 0, "agent_tokens": 0, "monitor_tokens": 0,
           "other_calls": 0, "events_without_usage": 0, "unfinished_calls": 0,
           "unresolved": {"agent": 0, "monitor": 0, "other": 0}, "unresolved_reasons": []}
    for e in sample.events or []:
        if not isinstance(e, ModelEvent):
            continue
        role = "agent" if e.role == "untrusted" else ("monitor" if e.role == "trusted" else "other")
        u = e.output.usage if (e.output is not None and e.output.usage is not None) else None
        t = int((u.input_tokens or 0) + (u.output_tokens or 0)) if u is not None else 0
        reasons = []
        if u is None:
            out["events_without_usage"] += 1
            reasons.append("usage missing")
        if e.pending or e.error is not None or (e.retries or 0) > 0:
            out["unfinished_calls"] += 1
            reasons.append("pending" if e.pending else ("error" if e.error is not None else f"retries={e.retries}"))
        if reasons:
            out["unresolved"][role] += 1
            out["unresolved_reasons"].append({"role": role, "event": e.uuid, "reasons": reasons})
        out[f"{role}_calls"] += 1
        if role != "other":
            out[f"{role}_tokens"] += t
    out["tokens"] = out["agent_tokens"] + out["monitor_tokens"]
    out["complete"] = not out["unresolved_reasons"]
    out["unresolved_reasons"] = out["unresolved_reasons"][:20]
    return out


def e1_record(sample, log_meta: dict, units: dict[str, dict]) -> dict:
    rec = episode_record(sample, log_meta)
    rec["e1_cost"] = event_cost(sample)
    md = sample.metadata or {}
    cond = md.get("arh_e1_condition")
    rec["condition"] = cond
    rec["unit_id"] = md.get("arh_e1_unit")
    rec["e1"] = {k: store_field(sample, "E1State", k) for k in
                 ("condition", "container", "start_probe", "deviation_step", "snapshot", "restore", "injected")}
    rec["gate_ledger"] = {k: store_field(sample, "GateLedger", k) for k in
                          ("aborted_at_step", "blocked_steps", "consecutive_blocked")}
    if cond == "cont":
        u = units.get(rec["unit_id"])
        if u is None:
            rec["cont_source"] = None
            rec["context_check"] = {"status": "no_unit", "ok": False}
        else:
            rec["cont_source"] = {"deviation_step": u["deviation_step"], "base_episode_id": u["base_episode_id"],
                                  "prefix_usage": u["prefix_usage"], "snapshot": u["snapshot"],
                                  "prefix_check": u["prefix_check"], "step_score": u["step_score"]}
            rec["context_check"] = (context_check(sample, u) if not u.get("scripted") else
                                    {"status": "scripted policy (no agent request to compare)", "ok": True})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    args = ap.parse_args()
    d = os.path.expanduser(args.log_dir)
    units = {}
    up = os.path.join(d, "units.json")
    if os.path.exists(up):
        units = {u["unit_id"]: u for u in json.load(open(up))["units"]}
    files = {c: open(os.path.join(d, f"episodes_{c}.jsonl"), "w") for c in ("base", "live", "cont")}
    n = {c: 0 for c in files}
    try:
        for p in sorted(glob.glob(os.path.join(d, "*.eval"))):
            log = read_eval_log(p, resolve_attachments=True)
            meta = dict(log.eval.metadata or {})
            for s in log.samples or []:
                rec = e1_record(s, meta, units)
                c = rec["condition"]
                if c in files:
                    files[c].write(json.dumps(rec, default=str) + "\n")
                    n[c] += 1
    finally:
        for f in files.values():
            f.close()
    print(json.dumps({"log_dir": d, "n_records": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Extract E7 episodes from one attempt's .eval files into one JSONL per condition.

Each record is extract_episodes.episode_record (loss, per-step table, ledger calls, step texts)
plus: the condition and unit id, the gate (1 or 2, for live and cont) and its threshold, the
E1State record (container, start probe, restore record of a cont), the E7State record (base: one
snapshot record per step above the floor), the gate ledger (decisions, blocked steps, abort step),
the per-step monitor record (the gate score s, the discrete score, the status, the number of
probes), the model-event cost (extract_e1.event_cost), and the agent trace (per agent model call:
the step, the sha256 of the request with tool-call ids replaced by their order of first
appearance, the sha256 of the output without ids, and the seed). For a cont record also the
source unit's block step, prefix usage and snapshot record (from <log-dir>/units.json) and the
post-run context check (extract_e1.context_check, unchanged).

Writes episodes_base.jsonl, episodes_live.jsonl and episodes_cont.jsonl in the attempt directory,
so the prediction step can read base and cont records without opening a live record.

Usage: python extract_e7.py --log-dir <attempt dir>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant

from agent_risk_harness.e1_units import raw_request, store_field
from extract_e1 import context_check, event_cost
from extract_episodes import episode_record


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(getattr(p, "text", "") or "") for p in content)
    return content or ""


def norm_request_sha(raw: dict | None) -> str | None:
    """sha256 of the request body with tool-call ids replaced by their order of first appearance
    (vLLM draws a fresh id for every tool call, so two executions that produce identical outputs
    still log different ids)."""
    if raw is None:
        return None
    ids: dict[str, str] = {}

    def cid(x):
        return None if x is None else ids.setdefault(str(x), f"T{len(ids)}")
    msgs = []
    for m in raw.get("messages") or []:
        m = json.loads(json.dumps(m, default=str))
        if isinstance(m, dict):
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and "id" in tc:
                    tc["id"] = cid(tc["id"])
            if "tool_call_id" in m:
                m["tool_call_id"] = cid(m["tool_call_id"])
        msgs.append(m)
    body = {"messages": msgs, "tools": raw.get("tools"), "seed": raw.get("seed"),
            "temperature": raw.get("temperature"), "top_p": raw.get("top_p"), "max_tokens": raw.get("max_tokens"),
            "model": raw.get("model")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def out_sha(message) -> str:
    """sha256 of an agent output: its text and every tool call's function and arguments (no ids)."""
    calls = [(tc.function, json.dumps(tc.arguments, sort_keys=True, default=str)) for tc in (message.tool_calls or [])]
    return hashlib.sha256(json.dumps({"text": _text(message.content), "calls": calls}, sort_keys=True).encode()).hexdigest()


def agent_trace(sample) -> list[dict]:
    md = sample.metadata or {}
    ep_seed = int(md.get("arh_episode_seed", 0))
    out = []
    for e in sample.events or []:
        if not (isinstance(e, ModelEvent) and e.role == "untrusted"):
            continue
        req = raw_request(e)
        seed = (req or {}).get("seed")
        if seed is None and e.config is not None:
            seed = e.config.seed
        step = (int(seed) - ep_seed * 1000) if isinstance(seed, int) else None
        msg = e.output.message if (e.output is not None and e.output.choices) else None
        out.append({"step": step, "seed": seed, "req": norm_request_sha(req), "out": out_sha(msg) if msg is not None else None,
                    "n_calls": len(msg.tool_calls or []) if msg is not None else None,
                    "stop": getattr(e.output, "stop_reason", None) if e.output is not None else None})
    return out


def monitor_steps(sample) -> list[dict]:
    mon = []
    for m in sample.messages:
        if not isinstance(m, ChatMessageAssistant):
            continue
        mm = m.metadata or {}
        det = mm.get("e2_cscore_detail") or {}
        mon.append({"step": mm.get("arh_step"), "score": mm.get("suspicion_score"), "discrete": mm.get("e2_score_discrete"),
                    "status": mm.get("e2_cscore_status"), "source": mm.get("e2_gate_score_source"),
                    "n_probes": det.get("n_probes"), "n_calls": len(m.tool_calls or [])})
    return mon


def e7_record(sample, log_meta: dict, units: dict[str, dict]) -> dict:
    rec = episode_record(sample, log_meta)
    rec["e1_cost"] = event_cost(sample)
    md = sample.metadata or {}
    cond = md.get("arh_e1_condition")
    rec["condition"] = cond
    rec["unit_id"] = md.get("arh_e1_unit")
    rec["base_unit_id"] = md.get("arh_e7_base_unit")
    rec["gate_index"] = md.get("arh_e7_gate")
    rec["lam"] = md.get("arh_e7_theta")
    rec["e1"] = {k: store_field(sample, "E1State", k) for k in
                 ("condition", "container", "start_probe", "start_pids", "restore", "injected")}
    rec["e7"] = {"snapshots": store_field(sample, "E7State", "snapshots") or []}
    rec["gate_ledger"] = {k: store_field(sample, "GateLedger", k) for k in
                          ("aborted_at_step", "blocked_steps", "consecutive_blocked")}
    rec["monitor"] = monitor_steps(sample)
    rec["trace"] = agent_trace(sample)
    if cond == "cont":
        u = units.get(rec["unit_id"])
        if u is None:
            rec["cont_source"] = None
            rec["context_check"] = {"status": "no_unit", "ok": False}
        else:
            rec["cont_source"] = {"deviation_step": u["deviation_step"], "base_episode_id": u["base_episode_id"],
                                  "gate": u["gate"], "prefix_usage": u["prefix_usage"], "snapshot": u["snapshot"],
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
                rec = e7_record(s, meta, units)
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

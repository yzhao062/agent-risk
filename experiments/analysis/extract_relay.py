"""Extract D66 relay episodes from .eval logs into JSONL.

Each record is extract_episodes.episode_record (unchanged; the relay scorer's
score is passed to it under the declared-loss name, so the per-step table,
ledger calls and step texts have the same form as in every earlier run), plus:

- relay fields: unit id, mode, A2 seed, task class, the RelayState record
  (handoff record, snapshot, start probe summary, identity check);
- the first A2 agent request: sha256 of the logged raw request without
  ``extra_headers`` (Inspect's per-request id header), the roles of its
  messages, and whether its system message ends with the handoff paragraph and
  its user message is the task. With full request logging these are taken from
  the raw request; otherwise from Inspect's OpenAI rendering of the logged
  input, marked as such.

Usage: python extract_relay.py --log-dir <attempt dir> --out <attempt dir>/episodes.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant

from extract_episodes import episode_record

SCORER = "relay_loss_scorer"


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def _rendered(ev: ModelEvent) -> tuple[dict, list]:
    from inspect_ai.model._openai import messages_to_openai
    msgs = _jsonable(asyncio.run(messages_to_openai(list(ev.input))))
    req = {"messages": msgs, "seed": ev.config.seed, "temperature": ev.config.temperature,
           "top_p": ev.config.top_p, "max_tokens": ev.config.max_tokens,
           "tools": sorted(t.name for t in (ev.tools or []))}
    return req, msgs


def request_view(ev: ModelEvent) -> dict:
    """Canonical form of one agent request for the leak check.

    The raw request (OpenAI body as sent to vLLM, without extra_headers) when it
    was logged and is in OpenAI form (no per-message ids); otherwise Inspect's
    OpenAI rendering of the logged input with the generation settings and tool
    names (mockllm logs message dumps with ids, which differ between runs)."""
    rreq, rmsgs = _rendered(ev)
    raw = ev.call.request if (ev.call is not None and isinstance(ev.call.request, dict)) else None
    raw_msgs = (raw or {}).get("messages") or []
    if raw is not None and raw_msgs and all(isinstance(m, dict) and "id" not in m for m in raw_msgs):
        req = {k: v for k, v in raw.items() if k != "extra_headers"}
        msgs = raw_msgs
        source = "raw_request"
    else:
        req, msgs = rreq, rmsgs
        source = "rendered_input"
    canon = json.dumps(_jsonable(req), sort_keys=True)

    def text(m):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""
    return {"sha256": hashlib.sha256(canon.encode()).hexdigest(), "source": source,
            "rendered_sha256": hashlib.sha256(json.dumps(_jsonable(rreq), sort_keys=True).encode()).hexdigest(),
            "roles": [m.get("role") for m in msgs if isinstance(m, dict)],
            "system_text": text(msgs[0]) if msgs else "", "user_text": text(msgs[1]) if len(msgs) > 1 else "",
            "messages_sha256": hashlib.sha256(json.dumps(_jsonable(msgs), sort_keys=True).encode()).hexdigest(),
            "seed": req.get("seed")}


def relay_record(sample, log_meta: dict) -> dict:
    sc = (sample.scores or {}).get(SCORER)
    if sc is not None:
        sample.scores = {**(sample.scores or {}), "declared_loss_scorer": sc}
    rec = episode_record(sample, log_meta)
    md = sample.metadata or {}
    rec["unit_id"] = md.get("arh_relay_unit")
    rec["mode"] = md.get("arh_relay_mode")
    rec["seed_a2"] = md.get("arh_episode_seed_a2")
    rec["task_class"] = md.get("arh_relay_task_class")
    rec["relay"] = ((sc.metadata or {}).get("relay") if sc is not None else None)
    rec["a2_first_request"] = None
    mode = rec["mode"]
    if mode and mode != "first":
        agent_events = [e for e in sample.events if isinstance(e, ModelEvent) and e.role == "untrusted"]
        if mode == "relay":
            h = (rec["relay"] or {}).get("handoff_step", -1)
            n_a1 = int(h) if h is not None and int(h) >= 0 else sum(
                1 for m in sample.messages if isinstance(m, ChatMessageAssistant)
                and (m.metadata or {}).get("arh_phase") != "A2")
        else:
            n_a1 = 0
        if n_a1 < len(agent_events):
            v = request_view(agent_events[n_a1])
            task = sample.input if isinstance(sample.input, str) else None
            v["user_is_task"] = (task is not None and v["user_text"] == task)
            v["n_a1_agent_events"] = n_a1
            rec["a2_first_request"] = v
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.log_dir), "*.eval")))
    n = 0
    with open(os.path.expanduser(args.out), "w") as f:
        for p in paths:
            log = read_eval_log(p, resolve_attachments=True)
            meta = log.eval.metadata or {}
            for s in log.samples or []:
                f.write(json.dumps(relay_record(s, meta), default=str) + "\n")
                n += 1
    print(f"wrote {n} relay episodes from {len(paths)} log(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

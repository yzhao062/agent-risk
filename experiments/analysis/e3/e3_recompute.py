"""Offline recomputation from E3 logs (needs tau2 v1.0.1; no model server).

For each episode record it rebuilds the airline environment from db.json, replays
the logged dispatched tool calls in order, and before each write applies checker
e3-airline-1.0 to the database at that point. It reports, per episode:

  harm_steps        steps whose executed writes the recomputed checker flags
                    (the per-step harm that the "minimal log without per-step
                    harm, plus the executed calls" variant recovers);
  flags_equal       recomputed codes and error flags equal the logged ones at
                    every dispatched call;
  results_equal     every recomputed tool result equals the logged tool message;
  db_hash_equal     the final database hash equals the logged one (episodes
                    that ran to their end);
  prompts_equal     every monitor prompt, rebuilt from the logged trajectory with
                    e3_monitor.render_prompt, has the logged sha256.

  ~/venv-tau2/bin/python e3_recompute.py --logroot logs/e3 --members B --out logs/e3/recompute.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import e3_netguard  # noqa: E402

e3_netguard.install()


def _messages(dumps: list[dict]):
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


def recompute_episode(rec: dict, policy: str) -> dict:
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
    from tau2.domains.airline.environment import get_environment
    import e3_checker as ck
    import e3_monitor
    env = get_environment()
    flags_equal = results_equal = True
    harm_steps = []
    traj = _messages(rec["trajectory"])
    tool_results = {m.id: m.content for m in traj if isinstance(m, ToolMessage)}
    for s in sorted(rec["per_step"], key=lambda r: r["step"]):
        hit = False
        for k, c in enumerate(s["calls"]):
            if not c.get("dispatched"):
                continue
            codes = ck.check_call(env.tools.db, c["name"], c["arguments"]) if c["name"] in ck.WRITE_TOOLS else []
            tc = ToolCall(id=f"e3-agent-{s['step']}-{k}", name=c["name"], arguments=c["arguments"], requestor="assistant")
            res = env.get_response(tc)
            if bool(res.error) != bool(c.get("error")) or (codes if c["name"] in ck.WRITE_TOOLS else []) != (c.get("codes") or []):
                flags_equal = False
            logged = tool_results.get(tc.id)
            if logged is not None and logged != res.content:
                results_equal = False
            if codes and not res.error:
                hit = True
        if hit:
            harm_steps.append(s["step"])
    db_equal = env.get_db_hash() == rec.get("final_db_hash")
    # monitor prompts: generated assistant turn t is the (t + 1)-th assistant message (the first is tau2's greeting)
    prompts_equal = True
    a_idx = [i for i, m in enumerate(traj) if isinstance(m, AssistantMessage)]
    for s in rec["per_step"]:
        mon = s.get("monitor")
        if not mon:
            continue
        i = a_idx[s["step"] + 1]
        p = e3_monitor.render_prompt(policy, traj[:i], traj[i].tool_calls)
        if e3_monitor.sha(p) != mon["prompt_sha256"]:
            prompts_equal = False
    return {"ok": True, "harm_steps": harm_steps, "flags_equal": flags_equal, "results_equal": results_equal,
            "db_hash_equal": db_equal, "prompts_equal": prompts_equal}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", required=True)
    ap.add_argument("--members", default="B,P,Q,L,C")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tau2.domains.airline.environment import get_environment
    policy = get_environment().get_policy()
    members = set(a.members.split(","))
    out = {}
    for p in sorted(glob.glob(os.path.join(a.logroot, "**", "episodes.jsonl"), recursive=True)):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("scored") or r.get("member") not in members:
                    continue
                try:
                    out[str(r["episode_id"])] = recompute_episode(r, policy)
                except Exception as exc:  # noqa: BLE001
                    out[str(r["episode_id"])] = {"ok": False, "error": repr(exc)[:500]}
    summ = {k: sum(1 for v in out.values() if v.get(k)) for k in
            ("ok", "flags_equal", "results_equal", "db_hash_equal", "prompts_equal")}
    summ["n"] = len(out)
    summ["egress_attempts"] = list(e3_netguard.ATTEMPTS)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke step S5: the monitor score is a deterministic function of its input on the
batch-invariant monitor server.

For every scored step in the given logs it rebuilds the monitor prompt from the
logged trajectory (as e3_recompute.py does), checks its sha256 against the log,
and recomputes the monitor call and the continuous score twice: one call at a time,
and 16 at a time. Every output text, discrete score, E[N]/100 and gate score s must
equal the logged values bitwise in both passes.

  sg e3net -c "~/venv-tau2/bin/python ~/harness/e3/e3_monitor_det.py --logroot ~/logs/e3_smoke2 --out ~/logs/e3_smoke2/monitor_det.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import e3_netguard  # noqa: E402

e3_netguard.install()


def items(logroot: str) -> list[dict]:
    import e3_recompute as rc
    from tau2.data_model.message import AssistantMessage
    out = []
    for p in sorted(glob.glob(os.path.join(logroot, "**", "episodes.jsonl"), recursive=True)):
        for line in open(p):
            r = json.loads(line)
            if not r.get("scored"):
                continue
            traj = rc._messages(r["trajectory"])
            a_idx = [i for i, m in enumerate(traj) if isinstance(m, AssistantMessage)]
            for s in r["per_step"]:
                m = s.get("monitor")
                if m:
                    i = a_idx[s["step"] + 1]
                    out.append({"episode_id": r["episode_id"], "step": s["step"], "logged": m,
                                "prefix": traj[:i], "calls": traj[i].tool_calls})
    return out


def score(policy: str, it: dict) -> dict:
    import requests
    import e3_monitor as em
    sess = requests.Session()
    m = em.score_step(sess, policy, it["prefix"], it["calls"])
    lg = it["logged"]
    return {"episode_id": it["episode_id"], "step": it["step"],
            "prompt_equal": m["prompt_sha256"] == lg["prompt_sha256"],
            "text_equal": m["output_text"] == lg["output_text"],
            "discrete_equal": m["score_discrete"] == lg["score_discrete"],
            "cscore_equal": m["cscore"] == lg["cscore"], "s_equal": m["score"] == lg["score"],
            "s": m["score"], "s_logged": lg["score"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from loguru import logger
    logger.remove()
    from tau2.domains.airline.environment import get_environment
    policy = get_environment().get_policy()
    its = items(os.path.expanduser(a.logroot))
    seq = [score(policy, it) for it in its]
    with ThreadPoolExecutor(max_workers=16) as pool:
        par = list(pool.map(lambda it: score(policy, it), its))
    keys = ("prompt_equal", "text_equal", "discrete_equal", "cscore_equal", "s_equal")
    summ = {"n_scored_steps": len(its),
            "concurrency_1": {k: sum(1 for x in seq if x[k]) for k in keys},
            "concurrency_16": {k: sum(1 for x in par if x[k]) for k in keys},
            "egress_attempts": list(e3_netguard.ATTEMPTS)}
    with open(os.path.expanduser(a.out), "w") as f:
        json.dump({"summary": summ, "sequential": seq, "parallel": par}, f, indent=1)
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

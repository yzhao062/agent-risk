"""Offline rescoring of logged pass-all episodes with another trusted monitor
(exploratory; for the coordinator's monitor decision before the confirmatory freeze).

A pass-all episode does not depend on the monitor (every call runs whatever the
score), so a different monitor can be evaluated on the pilot's logs without new
episodes: for every scored step the prompt is rebuilt from the logged trajectory
(the registered MONITOR_TEMPLATE, unchanged), sent to the alternative server with
the registered generation settings, and the continuous score s is computed with the
registered probe tree. The output is a map {episode_id: {step: s}} plus the discrete
scores, which e3_analyze.py can substitute for the logged scores to report G1, the
harm curve and the AUC of the alternative monitor on the same episodes.

  sg e3net -c "~/venv-tau2/bin/python ~/harness/e3/e3_monitor_alt.py --logroot ~/logs/e3_pilot \
      --url http://127.0.0.1:8002/v1 --model Qwen/Qwen3-14B-FP8 --concurrency 16 --out ~/logs/e3_pilot/alt_scores.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import e3_netguard  # noqa: E402

e3_netguard.install()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from loguru import logger
    logger.remove()
    import requests
    import e3_monitor as em
    import e3_monitor_det as md
    from tau2.domains.airline.environment import get_environment
    em.MONITOR_SERVED = a.model           # monitor_body and probe_body read this name at call time
    policy = get_environment().get_policy()
    its = md.items(os.path.expanduser(a.logroot))
    t0 = time.time()

    def one(it):
        m = em.score_step(requests.Session(), policy, it["prefix"], it["calls"], url=a.url)
        return {"episode_id": it["episode_id"], "step": it["step"], "s": m["score"], "discrete": m["score_discrete"],
                "cscore_status": m["cscore_status"], "prompt_equal": m["prompt_sha256"] == it["logged"]["prompt_sha256"],
                "monitor_input_tokens": m["monitor_input_tokens"], "monitor_output_tokens": m["monitor_output_tokens"]}

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rows = list(pool.map(one, its))
    out = {"model": a.model, "url": a.url, "n_steps": len(rows), "seconds": time.time() - t0,
           "prompts_equal": sum(1 for r in rows if r["prompt_equal"]), "rows": rows,
           "gate_scores": {}, "discrete_scores": {}, "egress_attempts": list(e3_netguard.ATTEMPTS)}
    for r in rows:
        out["gate_scores"].setdefault(r["episode_id"], {})[str(r["step"])] = r["s"]
        out["discrete_scores"].setdefault(r["episode_id"], {})[str(r["step"])] = r["discrete"]
    with open(os.path.expanduser(a.out), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("model", "n_steps", "seconds", "prompts_equal", "egress_attempts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Arm E: extract the logged agent decisions and their exact requests from an Inspect log.

Needs inspect-ai (the DeltaAI client venv has 0.3.257, the logging version).

  python arm_e_prepare.py steps --log-dir logs/pilot-ab/bi_baseline --source bi_baseline \
      --out data/steps_bi_baseline.jsonl
  python arm_e_prepare.py probe --steps data/steps_bi_baseline.jsonl data/steps_c1_baseline.jsonl \
      --out data/probe.json

``steps`` writes one row per agent model call (role ``untrusted``) with the request
payload exactly as Inspect logged it (``log_model_api=True``: every call carries it;
the script refuses a step without one, so no request is rebuilt) and the logged
decision taken from the raw server response. ``probe`` draws a fixed set of steps
that every replay server sends again one at a time, as a determinism check.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_e_common import decision_from_error, decision_from_message  # noqa: E402

DROP = ("extra_headers", "stream", "stream_options")


def cmd_steps(a) -> int:
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log

    paths = sorted(glob.glob(os.path.join(a.log_dir, "*.eval")))
    if len(paths) != 1:
        raise SystemExit(f"expected one .eval in {a.log_dir}, found {len(paths)}")
    log = read_eval_log(paths[0], resolve_attachments=True)
    md = log.eval.metadata or {}
    episodes = []
    n_rows = 0
    with open(a.out, "w") as f:
        for s in log.samples:
            evs = [e for e in s.events if isinstance(e, ModelEvent) and e.role == a.agent_role]
            ok = s.error is None
            ep_seed = None
            for k, e in enumerate(evs):
                if e.call is None or not e.call.request:
                    raise SystemExit(f"{s.id} step {k}: no logged request payload")
                req = {kk: v for kk, v in e.call.request.items() if kk not in DROP}
                resp = e.call.response or {}
                meta = (e.output.message.metadata or {}) if (e.output and e.output.message) else {}
                if ep_seed is None:
                    ep_seed = meta.get("arh_episode_seed")
                if "choices" in resp:
                    ch = resp["choices"][0]
                    logged = decision_from_message(ch.get("message") or {}, ch.get("finish_reason"))
                else:
                    logged = decision_from_error(str(resp.get("message") or e.error or ""), resp.get("code"))
                usage = resp.get("usage") or {}
                row = {
                    "source": a.source, "sample_id": str(s.id), "episode_ok": ok, "episode_seed": ep_seed,
                    "n_steps": len(evs), "step": k, "seed": req.get("seed"),
                    "seed_matches_metadata": req.get("seed") == meta.get("arh_step_seed") if meta else None,
                    "logged_prompt_tokens": usage.get("prompt_tokens"),
                    "logged_completion_tokens": usage.get("completion_tokens"),
                    "logged_stop_reason": str(e.output.stop_reason) if e.output else None,
                    "logged": logged, "request": req,
                }
                f.write(json.dumps(row) + "\n")
                n_rows += 1
            episodes.append({"sample_id": str(s.id), "episode_ok": ok, "n_steps": len(evs),
                             "episode_seed": ep_seed, "error": (str(s.error)[:300] if s.error else None)})
    summary = {
        "source": a.source, "log": os.path.basename(paths[0]), "n_samples": len(log.samples),
        "n_ok": sum(1 for e in episodes if e["episode_ok"]), "n_steps": n_rows,
        "agent_model": md.get("agent_model"), "agent_revision": md.get("agent_revision"),
        "agent_generation": md.get("agent_generation"), "concurrency": md.get("concurrency"),
        "episodes": episodes,
    }
    with open(a.out.replace(".jsonl", ".episodes.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))
    return 0


def cmd_probe(a) -> int:
    rng = random.Random(a.seed)
    rows = []
    for p in a.steps:
        with open(p) as f:
            rows.extend({k: r[k] for k in ("source", "sample_id", "step", "logged_prompt_tokens", "episode_ok")}
                        for r in map(json.loads, f))
    rows = [r for r in rows if r["episode_ok"]]
    bi = [r for r in rows if r["source"] == a.primary]
    other = [r for r in rows if r["source"] != a.primary]
    long_ = [r for r in bi if (r["logged_prompt_tokens"] or 0) > 8192]
    short = [r for r in bi if 0 < (r["logged_prompt_tokens"] or 0) <= 8192]
    pick = rng.sample(long_, a.n_long) + rng.sample(short, a.n_short) + rng.sample(other, a.n_other)
    out = [[r["source"], r["sample_id"], r["step"]] for r in pick]
    with open(a.out, "w") as f:
        json.dump({"seed": a.seed, "steps": out,
                   "note": f"{a.n_long} {a.primary} steps with logged prompt > 8192 tokens (chunked prefill), "
                           f"{a.n_short} shorter {a.primary} steps, {a.n_other} steps of the other source"}, f, indent=1)
    print(f"probe: {len(out)} steps -> {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("steps")
    s.add_argument("--log-dir", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--agent-role", default="untrusted")
    p = sub.add_parser("probe")
    p.add_argument("--steps", nargs="+", required=True)
    p.add_argument("--primary", default="bi_baseline")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--n-long", type=int, default=15)
    p.add_argument("--n-short", type=int, default=15)
    p.add_argument("--n-other", type=int, default=10)
    a = ap.parse_args()
    return cmd_steps(a) if a.cmd == "steps" else cmd_probe(a)


if __name__ == "__main__":
    raise SystemExit(main())

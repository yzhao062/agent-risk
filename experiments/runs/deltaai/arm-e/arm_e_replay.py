"""Arm E: send logged decision inputs to a replay server and record its decisions.

Each logged step is evaluated at its logged input: the request payload Inspect
recorded (chat context, tools, tool_choice, seed, temperature, top_p, max_tokens),
with only the model name replaced and ``return_token_ids`` added. Agreement after
step k is judged on the logged input of step k+1, so the steps of an episode are
independent requests.

  python arm_e_replay.py --steps data/steps_bi_baseline.jsonl --base-url http://127.0.0.1:PORT/v1 \
      --model Qwen/Qwen3-14B-FP8 --out out/bi_all.jsonl --mode all --concurrency 32

Modes:
  all    every step of every selected episode (the primary design; the analysis
         stops each episode at its first disagreement per interface).
  early  steps in order, stopping each episode after its first action-only
         disagreement; enough for both interfaces' agreement mass, first-step rate,
         and hazards, because a full-interface disagreement never comes later than
         the action-only one.
  --select FILE  only the listed (source, sample_id, step) triples (probe set).

Concurrency is the number of requests in flight. Episodes whose sample ended in an
Inspect error are skipped (``--include-failed``). The output is appended row by
row; a rerun with the same --out skips steps already decided (resume).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_e_common import agree, decision_from_error, decision_from_message  # noqa: E402


def load_steps(paths: list[str]) -> dict:
    eps: dict = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                eps.setdefault((r["source"], r["sample_id"]), []).append(r)
    for k in eps:
        eps[k].sort(key=lambda r: r["step"])
    return eps


async def call(client: httpx.AsyncClient, url: str, model: str, row: dict, timeout: float) -> dict:
    body = dict(row["request"])
    body["model"] = model
    body["return_token_ids"] = True
    t0 = time.time()
    last = None
    for attempt in range(4):
        try:
            r = await client.post(url, json=body, timeout=timeout)
            if r.status_code == 200:
                d = r.json()
                ch = d["choices"][0]
                dec = decision_from_message(ch.get("message") or {}, ch.get("finish_reason"))
                return {"status": "decision", "replay": dec, "token_ids": ch.get("token_ids"),
                        "usage": d.get("usage"), "http_status": 200, "seconds": round(time.time() - t0, 3)}
            if 400 <= r.status_code < 500:
                try:
                    msg = r.json().get("error", {}).get("message") or r.text
                except Exception:  # noqa: BLE001
                    msg = r.text
                return {"status": "decision", "replay": decision_from_error(msg, r.status_code),
                        "http_status": r.status_code, "seconds": round(time.time() - t0, 3)}
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        await asyncio.sleep(5 * (attempt + 1))
    return {"status": "transport_error", "error": last, "seconds": round(time.time() - t0, 3)}


async def main_async(a) -> int:
    eps = load_steps(a.steps)
    keys = sorted(k for k, rows in eps.items() if a.include_failed or rows[0]["episode_ok"])
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        keys = [k for j, k in enumerate(keys) if j % n == i]
    done: dict = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "decision":
                    done[(r["source"], r["sample_id"], r["step"])] = r.get("agree_action")
    url = a.base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    fout = open(a.out, "a")
    counts = {"sent": 0, "decision": 0, "transport_error": 0, "skipped_done": 0}

    async def do(client, row) -> dict | None:
        key = (row["source"], row["sample_id"], row["step"])
        if key in done:
            counts["skipped_done"] += 1
            return None
        async with sem:
            counts["sent"] += 1
            res = await call(client, url, a.model, row, a.timeout)
        out = {"source": row["source"], "sample_id": row["sample_id"], "step": row["step"],
               "tag": a.tag, "model": a.model, "concurrency": a.concurrency, **res}
        if res["status"] == "decision":
            out["agree_action"] = agree(row["logged"], res["replay"], "action")
            out["agree_full"] = agree(row["logged"], res["replay"], "full")
        counts[res["status"]] += 1
        async with lock:
            fout.write(json.dumps(out) + "\n")
            fout.flush()
        return out

    t0 = time.time()
    limits = httpx.Limits(max_connections=a.concurrency + 8, max_keepalive_connections=a.concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        if a.select:
            with open(a.select) as f:
                want = {tuple(x) for x in json.load(f)["steps"]}
            rows = [r for k, rs in eps.items() for r in rs if (r["source"], r["sample_id"], r["step"]) in want]
            # the probe is sent strictly one request at a time, in a fixed order
            for r in sorted(rows, key=lambda r: (r["source"], r["sample_id"], r["step"])):
                await do(client, r)
        elif a.mode == "all":
            rows = [r for k in keys for r in eps[k]]
            await asyncio.gather(*[do(client, r) for r in rows])
        else:
            async def episode(k):
                for r in eps[k]:
                    res = await do(client, r)
                    if res is None:  # resumed step: use the verdict already on file
                        if not done.get((r["source"], r["sample_id"], r["step"])):
                            break
                        continue
                    if res["status"] != "decision" or not res["agree_action"]:
                        break
            await asyncio.gather(*[episode(k) for k in keys])
    fout.close()
    print(json.dumps({"tag": a.tag, "model": a.model, "mode": "select" if a.select else a.mode,
                      "episodes": len(keys), "wall_seconds": round(time.time() - t0, 1), **counts}))
    return 0 if counts["transport_error"] == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", choices=("all", "early"), default="all")
    ap.add_argument("--select", default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--shard", default=None, help="i/n: keep episodes whose sorted index is i mod n")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    raise SystemExit(main())

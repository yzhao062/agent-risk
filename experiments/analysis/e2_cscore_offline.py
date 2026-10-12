"""Recompute the E2 continuous monitor score for logged monitor calls (E2-PREREG-2026-10-05.md).

For every monitor call in the given .eval logs (each assistant message with tool
calls paired with its trusted-model event, as d60_rescore.py pairs them), send
the probes of ``e2_cscore.expected_score`` for the logged input messages and
logged output text to a running monitor server, through the same function the
live E2 monitor uses (``e2_monitor.cscore_from_io``).

Uses:
  standin   D58 logs (discrete scores only; no logprobs were logged): the
            continuous score each B2 step would have had, for the 14B stand-in of
            the pilot rules (e2_analyze.py standin-pilot-b2 --cscores).
  repeat    --repeat: a second full pass at concurrency 1; every value must be
            bitwise equal to the first pass (determinism of the score).
  live      --compare-live: E2 logs; every recomputed value must equal the value
            the live gate read (message metadata e2_cscore).
  original  --compare-original N: for the first N calls, resend the logged request
            with top-20 logprobs; reports whether the output text is unchanged
            and the largest difference between the first-digit distribution at
            the decode position and the probe's (a diagnostic of the probe
            context).

Run on the VM (monitor server up), from ~/control-arena with PYTHONPATH=~/harness:
  .venv/bin/python ~/harness/e2_cscore_offline.py --log-dir ~/logs/d58/block-01-B2/attempt-1 \
      --out ~/logs/e2_smoke/cscore_b2_block01.json --repeat --compare-original 100
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import httpx

from agent_risk_harness.e2_cscore import DIGITS, EPS, _norm, dist_from_top_logprobs, score_prefix
from agent_risk_harness.e2_monitor import MONITOR_SERVED, MONITOR_URL, cscore_from_io, event_messages, request_messages


def collect(log_dirs: list[str]) -> tuple[list[dict], list[str]]:
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log_samples
    from inspect_ai.model import ChatMessageAssistant
    calls, problems = [], []
    for d in log_dirs:
        for p in sorted(glob.glob(os.path.join(os.path.expanduser(d), "*.eval"))):
            for s in read_eval_log_samples(p, all_samples_required=False, resolve_attachments=True):
                md = s.metadata or {}
                scored = [m for m in s.messages if isinstance(m, ChatMessageAssistant) and m.tool_calls]
                mon = [e for e in s.events if isinstance(e, ModelEvent) and e.role == "trusted"]
                if len(scored) != len(mon):
                    problems.append(f"{s.id}: {len(scored)} proposals with tool calls, {len(mon)} monitor calls")
                for m, e in zip(scored, mon):
                    mm = m.metadata or {}
                    msgs, src = event_messages(e)
                    req_msgs = request_messages(e)
                    calls.append({
                        "input_equals_request": (None if req_msgs is None else req_msgs == msgs),
                        "episode_id": md.get("arh_episode_id"), "step": mm.get("arh_step"),
                        "logged_score": mm.get("e2_score_discrete", mm.get("suspicion_score")),
                        "live_cscore": mm.get("e2_cscore"), "live_doubt": mm.get("e2_doubt"),
                        "text": e.output.message.text if (e.output is not None and e.output.choices) else "",
                        "messages": msgs, "messages_source": src,
                        "request": e.call.request if (e.call is not None and isinstance(e.call.request, dict)) else None,
                    })
    return calls, problems


async def score_all(calls: list[dict], url: str, eps: float, concurrency: int, key: str) -> None:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=120.0, limits=httpx.Limits(max_connections=4 * concurrency)) as client:
        async def one(c: dict) -> None:
            async with sem:
                try:
                    c[key] = await cscore_from_io(c["messages"], c["text"], url=url, eps=eps, client=client)
                except Exception as exc:  # noqa: BLE001
                    c[key] = {"value": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        await asyncio.gather(*(one(c) for c in calls))


async def compare_original(calls: list[dict], url: str, n: int) -> list[dict]:
    """Resend logged requests with logprobs; compare text and the first-digit distribution."""
    out = []
    async with httpx.AsyncClient(base_url=url, timeout=300.0) as client:
        for c in calls[:n]:
            req = c.get("request")
            if not req or c["first"].get("status") != "ok":
                continue
            body = {k: v for k, v in req.items() if k not in ("extra_headers", "tools", "tool_choice") and v is not None}
            body.update({"logprobs": True, "top_logprobs": 20})
            r = await client.post("/chat/completions", json=body)
            r.raise_for_status()
            ch = r.json()["choices"][0]
            text = ch["message"]["content"] or ""
            toks = ch["logprobs"]["content"]
            pre = score_prefix(c["text"])
            acc, pos = "", None
            for i, t in enumerate(toks):
                if acc == pre:
                    pos = i
                    break
                acc += t["token"]
            rec = {"episode_id": c["episode_id"], "step": c["step"], "text_equal": text == c["text"],
                   "aligned": pos is not None}
            if pos is not None:
                dec = {k: v for k, v in _norm(dist_from_top_logprobs(toks[pos]["top_logprobs"])).items() if k in DIGITS}
                prb = {k: v for k, v in c["first"]["probes"][""].items() if k in DIGITS}
                keys = sorted(set(dec) | set(prb))
                rec["max_abs_diff_first_digit"] = max(abs(dec.get(k, 0.0) - prb.get(k, 0.0)) for k in keys)
            out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=MONITOR_URL)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--compare-live", action="store_true")
    ap.add_argument("--compare-original", type=int, default=0)
    ap.add_argument("--export-jsonl", default=None,
                    help="write the collected calls (episode, step, logged score, messages, output text) and exit")
    args = ap.parse_args()
    t0 = time.time()
    calls, problems = collect(args.log_dir)
    t1 = time.time()
    if args.export_jsonl:
        with open(os.path.expanduser(args.export_jsonl), "w") as f:
            for c in calls:
                f.write(json.dumps({k: c[k] for k in ("episode_id", "step", "logged_score", "messages", "text",
                                                      "input_equals_request")}) + "\n")
        print(json.dumps({"exported": len(calls), "pairing_problems": len(problems), "out": args.export_jsonl}))
        return 0
    asyncio.run(score_all(calls, args.url, args.eps, args.concurrency, "first"))
    t2 = time.time()
    summ = {"n_calls": len(calls), "pairing_problems": problems, "seconds_read": t1 - t0, "seconds_first": t2 - t1,
            "status": {}, "n_probes_total": 0, "probe_prompt_tokens": 0}
    for c in calls:
        st = c["first"].get("status")
        summ["status"][st] = summ["status"].get(st, 0) + 1
        summ["n_probes_total"] += int(c["first"].get("n_probes") or 0)
        summ["probe_prompt_tokens"] += int((c["first"].get("usage") or {}).get("prompt_tokens") or 0)
        g, ls = c["first"].get("greedy"), c.get("logged_score")
        c["greedy_agrees"] = (g is not None and ls is not None and abs(int(g) / 100.0 - float(ls)) < 1e-12)
    ok = [c for c in calls if c["first"].get("status") == "ok"]
    summ["greedy_agrees"] = [sum(c["greedy_agrees"] for c in ok), len(ok)]
    summ["messages_source"] = {s: sum(1 for c in calls if c["messages_source"] == s) for s in ("request", "input")}
    summ["input_messages_equal_logged_request"] = [sum(1 for c in calls if c["input_equals_request"] is True),
                                                   sum(1 for c in calls if c["input_equals_request"] is not None)]
    if args.repeat:
        asyncio.run(score_all(calls, args.url, args.eps, 1, "second"))
        same = sum(1 for c in calls if c["first"].get("value") == c["second"].get("value")
                   and c["first"].get("probes") == c["second"].get("probes"))
        summ["repeat_bitwise_equal"] = [same, len(calls)]
        summ["seconds_second"] = time.time() - t2
    if args.compare_live:
        have = [c for c in calls if c.get("live_cscore") is not None]
        summ["live_bitwise_equal"] = [sum(1 for c in have if c["first"].get("value") == c["live_cscore"]
                                          and c["first"].get("doubt") == c["live_doubt"]), len(have)]
    if args.compare_original:
        orig = asyncio.run(compare_original(calls, args.url, args.compare_original))
        diffs = [o["max_abs_diff_first_digit"] for o in orig if "max_abs_diff_first_digit" in o]
        summ["original"] = {"n": len(orig), "text_equal": sum(o["text_equal"] for o in orig),
                            "aligned": sum(o["aligned"] for o in orig),
                            "max_abs_diff_first_digit_max": max(diffs) if diffs else None,
                            "max_abs_diff_first_digit_median": sorted(diffs)[len(diffs) // 2] if diffs else None,
                            "records": orig}
    scores, gate_scores = {}, {}
    for c in calls:
        scores.setdefault(str(c["episode_id"]), {})[str(c["step"])] = c["first"].get("value")
        gate_scores.setdefault(str(c["episode_id"]), {})[str(c["step"])] = c["first"].get("doubt")
    out = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "log_dirs": args.log_dir, "url": args.url, "eps": args.eps, "summary": summ, "cscores": scores,
           "gate_scores": gate_scores,
           "calls": [{"episode_id": c["episode_id"], "step": c["step"], "logged_score": c["logged_score"],
                      "greedy_agrees": c["greedy_agrees"],
                      "first": {k: v for k, v in c["first"].items() if k != "probes"},
                      "second_value": (c.get("second") or {}).get("value")} for c in calls]}
    with open(os.path.expanduser(args.out), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "original"} |
                     ({"original": {k: v for k, v in summ["original"].items() if k != "records"}}
                      if "original" in summ else {}), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

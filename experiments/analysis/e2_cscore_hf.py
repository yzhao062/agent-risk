"""Approximate E2 continuous scores for logged monitor calls with Hugging Face
transformers (stand-in only; E2-PREREG-2026-10-05.md).

The registered continuous score is computed by the batch-invariant vLLM monitor
server (e2_monitor.py, e2_cscore_offline.py). The D58 logs carry no logprobs,
and no GPU host exists yet, so this tool runs the same tree
(e2_cscore.expected_score, the same eps and top-20 truncation) with the same
weights (Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f) through transformers on
another device (here Apple MPS, bf16). Kernels differ from the server's, so the
values approximate the registered score; they are used only to show what the
pilot rule G1 would do on the 14B agent's B2 log, never as a gate input.

Context: the chat template with the generation prompt, followed by the
monitor's own output up to the last "<score>" (vLLM's continue_final_message
renders the same text), tokenized as one string; each further digit is one
token, fed through the KV cache.

  python e2_cscore_hf.py --calls b2_monitor_calls.jsonl --out cscore_b2_hf.json [--episodes ids.txt]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("e2_cscore", os.path.join(HERE, "agent_risk_harness", "e2_cscore.py"))
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOP = 20


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default=None, help="file with episode ids to score first (one per line)")
    ap.add_argument("--only-listed", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--eps", type=float, default=cs.EPS)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).to(args.device).eval()
    calls = [json.loads(l) for l in open(args.calls)]
    if args.episodes:
        first = set(x.strip() for x in open(args.episodes) if x.strip())
        calls.sort(key=lambda c: (c["episode_id"] not in first, c["episode_id"], int(c["step"])))
        if args.only_listed:
            calls = [c for c in calls if c["episode_id"] in first]
    digit_ids = {d: tok.convert_tokens_to_ids(d) for d in cs.DIGITS}
    out_calls, scores = [], {}
    if os.path.exists(args.out):   # resume: keep the calls already scored
        prev = json.load(open(args.out))
        out_calls = prev.get("calls", [])
        scores = prev.get("cscores", {})
        done = {(x["episode_id"], str(x["step"])) for x in out_calls}
        calls = [c for c in calls if (c["episode_id"], str(c["step"])) not in done]
        print(json.dumps({"resumed": len(done), "remaining": len(calls)}), flush=True)
    n_total = len(out_calls) + len(calls)
    t0 = time.time()

    @torch.no_grad()
    def score_call(c: dict) -> dict:
        base = cs.score_prefix(c["text"])
        if base is None:
            return {"value": None, "status": "no_score_tag"}
        prompt = tok.apply_chat_template(c["messages"], add_generation_prompt=True, tokenize=False) + base
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        o = model(ids, use_cache=True)
        pkv = o.past_key_values
        n0 = ids.shape[1]
        logits = {"": o.logits[0, -1].float()}

        def logits_for(s: str):
            # feed the digits of s after the base prefix through the cache, then crop it back
            if s not in logits:
                t = torch.tensor([[digit_ids[ch] for ch in s]], device=args.device)
                o2 = model(t, past_key_values=pkv, use_cache=True)
                logits[s] = o2.logits[0, -1].float()
                pkv.crop(n0)
            return logits[s]

        async def nd(s: str) -> dict:
            lp = torch.log_softmax(logits_for(s), dim=-1)
            v, i = torch.topk(lp, TOP)
            out: dict[str, float] = {}
            for val, idx in zip(v.tolist(), i.tolist()):
                t = tok.decode([idx])
                out[t] = out.get(t, 0.0) + float(torch.tensor(val).exp())
            return out
        res = asyncio.run(cs.expected_score(nd, args.eps))
        res["status"] = "ok" if res["value"] is not None else "no_valid_mass"
        return res

    for k, c in enumerate(calls):
        r = score_call(c)
        g, ls = r.get("greedy"), c.get("logged_score")
        agree = g is not None and ls is not None and abs(int(g) / 100.0 - float(ls)) < 1e-12
        scores.setdefault(str(c["episode_id"]), {})[str(c["step"])] = r.get("value")
        out_calls.append({"episode_id": c["episode_id"], "step": c["step"], "logged_score": ls, "greedy_agrees": agree,
                          "value": r.get("value"), "tail": r.get("tail"), "status": r.get("status"),
                          "n_probes": r.get("n_probes"), "unexplored_mass": r.get("unexplored_mass"),
                          "invalid_mass": r.get("invalid_mass"), "greedy": g, "probes": r.get("probes")})
        if (k + 1) % 50 == 0 and args.device == "mps":
            torch.mps.empty_cache()
        if (k + 1) % 250 == 0 or k + 1 == len(calls):
            el = time.time() - t0
            summ = {"n_done": len(out_calls), "n_total": n_total, "seconds_this_run": el,
                    "greedy_agrees": [sum(x["greedy_agrees"] for x in out_calls if x["status"] == "ok"),
                                      sum(1 for x in out_calls if x["status"] == "ok")],
                    "status": {s: sum(1 for x in out_calls if x["status"] == s) for s in {x["status"] for x in out_calls}}}
            with open(args.out, "w") as f:
                json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "approximate": True,
                           "engine": f"transformers {__import__('transformers').__version__}, torch {torch.__version__}, "
                                     f"{args.device}, bf16", "model": MODEL, "revision": REVISION, "eps": args.eps,
                           "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                           "summary": summ, "cscores": scores, "calls": out_calls}, f)
            print(json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

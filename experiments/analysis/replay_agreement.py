"""Arm E: replay logged agent decisions through a served model and measure agreement.

For every episode (Inspect sample) in the given .eval logs, this takes the agent's
logged model calls in order (ModelEvents on the agent role) and, for each step t:

  1. rebuilds the decision input exactly as it was sent to the server: the
     ``call.request`` payload Inspect recorded (full chat context, tool list,
     tool_choice, and the sampling fields that were sent: seed, temperature,
     top_p, max_tokens). Inspect 0.3.257 keeps that payload only for the first 5
     calls per model per sample unless the eval runs with ``log_model_api=True``;
     for later steps the payload is rebuilt from the event's recorded input,
     tools, tool_choice, and GenerateConfig with the converters of Inspect's vLLM
     provider (``--check-reconstruction`` verifies the rebuild on the steps that
     do carry a logged payload; it matched 14 of 14 smoke-test steps exactly);
  2. sends it to the replay server (an OpenAI-compatible vLLM endpoint), with only
     the model name replaced;
  3. compares the replayed decision with the logged one. A decision is the ordered
     list of tool calls, each as (function name, arguments parsed from JSON), so
     key order and whitespace inside the JSON do not matter. Calls to memory tools
     (names matching ``--memory-tool-regex``) are also reported separately as the
     step's memory writes. With ``--include-content`` the stripped text content
     must also match.

Replay of an episode stops at its first disagreement (``--all-steps`` continues,
for diagnostics only; the summary statistics always use the steps up to and
including the first disagreement). The summary reports:

  - agreement mass: the fraction of episodes whose every logged step is
    reproduced, with a Wilson interval and an episode-level bootstrap interval;
  - first-decision bound: the step-1 agreement rate, which upper-bounds the
    agreement mass, with the same intervals;
  - per-step hazard h_t = (episodes whose first disagreement is at step t) /
    (episodes at risk at step t: logged step t exists and steps 1..t-1 agreed),
    and the implied survival curve;
  - for a same-model run (``--expect-same-model``), the reproducibility verdict and
    the exact-match rate of the raw response (text and raw argument strings);
  - a field audit: which replay inputs each step lacked (no request payload, no
    seed, a constant seed across steps, no temperature, top_p, or max_tokens), and
    which run-level fields the log lacks (model revision, server chat-template
    settings).

Example (logged model re-run on DeltaAI, the reproducibility check):

  python replay_agreement.py --logs logs/smoke --base-url http://127.0.0.1:8000/v1 \
      --model Qwen/Qwen3-14B-FP8 --expect-same-model --out-dir out/replay_smoke_same

Example (second model, honest-mode episodes only):

  python replay_agreement.py --logs logs/pilot-ab/b0 --base-url http://127.0.0.1:8000/v1 \
      --model Qwen/Qwen3.5-27B-FP8 --mode honest --out-dir out/replay_b0_qwen35
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import glob
import json
import math
import os
import random
import re
import time
from typing import Any

import httpx
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log

SAMPLING_FIELDS = ("seed", "temperature", "top_p", "max_tokens")
DROP_REQUEST_KEYS = ("extra_headers", "stream", "stream_options")


# ---------------------------------------------------------------------------
# decisions


def _canon_args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return {"__unparsed__": raw}
    return raw


def decision_from_openai_message(msg: dict, include_content: bool) -> dict:
    calls = []
    raw = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": _canon_args(fn.get("arguments"))})
        raw.append([fn.get("name"), fn.get("arguments")])
    content = msg.get("content")
    d = {"tool_calls": calls, "raw": {"content": content, "tool_calls": raw}}
    if include_content:
        d["content"] = (content or "").strip()
    return d


def decision_from_inspect_message(message) -> dict:
    calls = [{"name": tc.function, "arguments": tc.arguments} for tc in (message.tool_calls or [])]
    try:
        content = message.text
    except Exception:  # noqa: BLE001
        content = None
    return {"tool_calls": calls, "raw": None, "content_text": content}


def memory_writes(decision: dict, pattern: re.Pattern) -> list:
    return [c for c in decision["tool_calls"] if c["name"] and pattern.search(c["name"])]


def decisions_equal(a: dict, b: dict, include_content: bool) -> bool:
    if a["tool_calls"] != b["tool_calls"]:
        return False
    if include_content and a.get("content", "") != b.get("content", ""):
        return False
    return True


# ---------------------------------------------------------------------------
# log reading and field audit


def find_logs(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "*.eval"), recursive=True)))
        else:
            out.append(p)
    return out


def run_level_audit(log) -> dict:
    md = log.eval.metadata or {}
    gen = md.get("agent_generation") or {}
    return {
        "agent_model": md.get("agent_model"),
        "agent_revision": md.get("agent_revision"),
        "has_agent_revision": bool(md.get("agent_revision")),
        "server_settings_recorded": bool(gen.get("server")),
        "server_settings": gen.get("server"),
        "agent_generation_metadata": gen or None,
    }


def episode_steps(sample, agent_role: str, include_content: bool) -> list[dict]:
    steps = []
    for ev in sample.events or []:
        if not isinstance(ev, ModelEvent) or ev.role != agent_role:
            continue
        req = copy.deepcopy(ev.call.request) if ev.call is not None else None
        logged = None
        if ev.call is not None and ev.call.response:
            ch = (ev.call.response.get("choices") or [{}])[0]
            logged = decision_from_openai_message(ch.get("message") or {}, include_content)
        if logged is None and ev.output is not None and ev.output.message is not None:
            logged = decision_from_inspect_message(ev.output.message)
            if include_content:
                logged["content"] = (logged.get("content_text") or "").strip()
        meta = (ev.output.message.metadata or {}) if (ev.output is not None and ev.output.message is not None) else {}
        steps.append({"request": req, "request_source": "logged_call" if req is not None else None,
                      "event": ev, "logged": logged, "meta_step_seed": meta.get("arh_step_seed"),
                      "meta_generation": meta.get("arh_agent_generation")})
    return steps


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)))


async def reconstruct_request(ev, strict: bool | None) -> dict:
    """Rebuild the chat-completions payload from a ModelEvent without a logged call.

    Uses the converters of Inspect's OpenAI-compatible (vLLM) provider on the
    event's recorded input (already preprocessed by Inspect), tools, tool_choice,
    and GenerateConfig. ``--check-reconstruction`` verifies the result against the
    steps whose payload was logged.
    """
    from inspect_ai.model._openai import (
        messages_to_openai,
        openai_chat_tool_choice,
        openai_chat_tools,
        openai_completion_params,
    )
    from inspect_ai.util._json import JSON_SCHEMA_EXTENDED_FIELDS

    req: dict[str, Any] = {"messages": await messages_to_openai(ev.input)}
    if ev.tools:
        tools = openai_chat_tools(ev.tools, exclude=JSON_SCHEMA_EXTENDED_FIELDS)
        if strict is not None:
            for t in tools:
                t["function"]["strict"] = strict
        req["tools"] = tools
        req["tool_choice"] = openai_chat_tool_choice(ev.tool_choice)
    model = str(ev.model).split("/", 1)[1] if str(ev.model).startswith("vllm/") else str(ev.model)
    req.update(openai_completion_params(model=model, config=ev.config, tools=bool(ev.tools),
                                        schema_exclude=JSON_SCHEMA_EXTENDED_FIELDS))
    return _jsonable(req)


def _first_difference(a: Any, b: Any, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path or '/'}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: missing on one side"
            d = _first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:80]!r} vs {str(b)[:80]!r}"


async def fill_requests(episodes: list[dict], check: bool) -> dict:
    """Reconstruct missing payloads; optionally check reconstruction on logged ones."""
    stats = {"n_logged_call": 0, "n_reconstructed": 0, "n_checked": 0, "n_check_equal": 0, "check_differences": []}
    for ep in episodes:
        strict = None
        for st in ep["steps"]:
            if st["request"] is not None and st["request"].get("tools"):
                strict = st["request"]["tools"][0].get("function", {}).get("strict")
                break
        for t, st in enumerate(ep["steps"], start=1):
            ev = st["event"]
            if st["request"] is not None:
                stats["n_logged_call"] += 1
                if check and ev.input:
                    rec = await reconstruct_request(ev, strict)
                    logged = {k: v for k, v in _jsonable(st["request"]).items() if k not in DROP_REQUEST_KEYS}
                    diff = _first_difference(logged, rec)
                    stats["n_checked"] += 1
                    if diff is None:
                        stats["n_check_equal"] += 1
                    elif len(stats["check_differences"]) < 10:
                        stats["check_differences"].append({"sample_id": ep["sample_id"], "step": t, "diff": diff})
            elif ev.input:
                st["request"] = await reconstruct_request(ev, strict)
                st["request_source"] = "reconstructed"
                stats["n_reconstructed"] += 1
            st["event"] = None  # release memory
    return stats


def step_audit(steps: list[dict]) -> list[dict]:
    seeds = [((s["request"] or {}).get("seed")) for s in steps]
    constant_seed = len(steps) > 1 and len(set(map(str, seeds))) == 1
    out = []
    for s, seed in zip(steps, seeds):
        req = s["request"]
        missing = []
        if s["request_source"] != "logged_call":
            missing.append("logged_request_payload")
        if req is None:
            missing.append("request_payload")
        else:
            for k in ("messages", "tools"):
                if not req.get(k):
                    missing.append(k)
            for k in SAMPLING_FIELDS:
                if req.get(k) is None:
                    missing.append(k)
        if s["logged"] is None:
            missing.append("logged_response")
        if s["meta_step_seed"] is None:
            missing.append("metadata_step_seed")
        notes = []
        if constant_seed and seed is not None:
            notes.append("seed_constant_across_steps")
        if s["meta_step_seed"] is not None and seed is not None and int(s["meta_step_seed"]) != int(seed):
            notes.append("metadata_seed_differs_from_request_seed")
        out.append({"missing": missing, "notes": notes})
    return out


# ---------------------------------------------------------------------------
# replay


async def replay_step(client: httpx.AsyncClient, url: str, model: str, req: dict, args) -> dict:
    body = {k: v for k, v in req.items() if k not in DROP_REQUEST_KEYS}
    body["model"] = model
    imputed = []
    if body.get("max_tokens") is None and args.default_max_tokens:
        body["max_tokens"] = args.default_max_tokens
        imputed.append("max_tokens")
    if args.override_temperature is not None:
        body["temperature"] = args.override_temperature
        imputed.append("temperature(override)")
    if args.chat_template_kwargs:
        body["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    t0 = time.time()
    last_err = None
    for _ in range(3):
        try:
            r = await client.post(url, json=body, timeout=args.timeout)
            if r.status_code == 200:
                d = r.json()
                ch = d["choices"][0]
                return {"message": ch.get("message") or {}, "finish_reason": ch.get("finish_reason"),
                        "usage": d.get("usage"), "seconds": round(time.time() - t0, 3), "imputed": imputed}
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            if 400 <= r.status_code < 500:
                break
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        await asyncio.sleep(2)
    return {"error": last_err, "seconds": round(time.time() - t0, 3), "imputed": imputed}


async def replay_episode(client, url, ep: dict, args, mem_pat) -> list[dict]:
    rows = []
    disagreed = False
    for t, (step, audit) in enumerate(zip(ep["steps"], ep["audit"]), start=1):
        row = {"log": ep["log"], "sample_id": ep["sample_id"], "epoch": ep["epoch"], "step": t,
               "missing": audit["missing"], "notes": audit["notes"]}
        req = step["request"]
        if req is None or step["logged"] is None:
            row["status"] = "unreplayable"
            rows.append(row)
            break
        res = await replay_step(client, url, args.model, req, args)
        row["request_source"] = step["request_source"]
        row["request_sampling"] = {k: req.get(k) for k in SAMPLING_FIELDS}
        row["imputed"] = res.get("imputed")
        row["seconds"] = res.get("seconds")
        if "error" in res:
            row["status"] = "error"
            row["error"] = res["error"]
            rows.append(row)
            break
        rep = decision_from_openai_message(res["message"], args.include_content)
        log_d = step["logged"]
        agree = decisions_equal(log_d, rep, args.include_content)
        row.update({
            "status": "replayed",
            "agree": agree,
            "exact_raw_match": (None if log_d.get("raw") is None else log_d["raw"] == rep["raw"]),
            "memory_write_agree": memory_writes(log_d, mem_pat) == memory_writes(rep, mem_pat),
            "logged_tools": [c["name"] for c in log_d["tool_calls"]],
            "replay_tools": [c["name"] for c in rep["tool_calls"]],
            "finish_reason": res.get("finish_reason"),
            "replay_usage": res.get("usage"),
        })
        if not agree:
            row["logged_decision"] = log_d["tool_calls"]
            row["replay_decision"] = rep["tool_calls"]
            row["replay_content"] = (res["message"].get("content") or "")[:2000]
        rows.append(row)
        if not agree and not disagreed:
            disagreed = True
            if not args.all_steps:
                break
    return rows


# ---------------------------------------------------------------------------
# statistics


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def bootstrap_ci(xs: list[int], B: int = 2000, seed: int = 0) -> list[float] | None:
    if not xs:
        return None
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(means[int(0.025 * B)], 4), round(means[int(0.975 * B) - 1], 4)]


def summarize(episodes: list[dict], rows_by_ep: dict, args) -> dict:
    per_ep = []
    for ep in episodes:
        key = (ep["log"], ep["sample_id"], ep["epoch"])
        rows = rows_by_ep.get(key, [])
        replayed = [r for r in rows if r.get("status") == "replayed"]
        first_dis = next((r["step"] for r in replayed if not r["agree"]), None)
        complete = bool(rows) and all(r.get("status") == "replayed" for r in rows[: (first_dis or len(rows))])
        n_logged = len(ep["steps"])
        full_agree = complete and first_dis is None and len(replayed) == n_logged
        per_ep.append({"key": key, "n_logged_steps": n_logged, "first_disagreement_step": first_dis,
                       "complete": complete, "full_agreement": full_agree,
                       "step1_agree": (replayed[0]["agree"] if replayed and replayed[0]["step"] == 1 else None)})
    usable = [e for e in per_ep if e["complete"] or e["first_disagreement_step"] is not None]
    mass = [1 if e["full_agreement"] else 0 for e in usable]
    s1 = [1 if e["step1_agree"] else 0 for e in usable if e["step1_agree"] is not None]
    max_t = max((e["n_logged_steps"] for e in usable), default=0)
    hazard = []
    surv = 1.0
    for t in range(1, max_t + 1):
        at_risk = [e for e in usable if e["n_logged_steps"] >= t and (e["first_disagreement_step"] is None or e["first_disagreement_step"] >= t)]
        events = [e for e in at_risk if e["first_disagreement_step"] == t]
        h = len(events) / len(at_risk) if at_risk else None
        if h is not None:
            surv *= (1 - h)
        hazard.append({"step": t, "at_risk": len(at_risk), "first_disagreements": len(events),
                       "hazard": None if h is None else round(h, 4), "survival": round(surv, 4)})
    all_rows = [r for rs in rows_by_ep.values() for r in rs]
    replayed = [r for r in all_rows if r.get("status") == "replayed"]
    missing_counts: dict[str, int] = {}
    note_counts: dict[str, int] = {}
    for ep in episodes:
        for a in ep["audit"]:
            for m in a["missing"]:
                missing_counts[m] = missing_counts.get(m, 0) + 1
            for m in a["notes"]:
                note_counts[m] = note_counts.get(m, 0) + 1
    out = {
        "replay_model": args.model, "base_url": args.base_url, "expect_same_model": args.expect_same_model,
        "concurrency": args.concurrency, "include_content": args.include_content, "all_steps": args.all_steps,
        "n_episodes": len(episodes), "n_usable_episodes": len(usable),
        "n_logged_steps_total": sum(len(ep["steps"]) for ep in episodes),
        "n_steps_replayed": len(replayed),
        "n_step_agree": sum(1 for r in replayed if r["agree"]),
        "n_step_agree_by_request_source": {
            src: f"{sum(1 for r in replayed if r.get('request_source') == src and r['agree'])}/"
                 f"{sum(1 for r in replayed if r.get('request_source') == src)}"
            for src in ("logged_call", "reconstructed")},
        "n_step_exact_raw_match": f"{sum(1 for r in replayed if r.get('exact_raw_match'))}/"
                                  f"{sum(1 for r in replayed if r.get('exact_raw_match') is not None)}",
        "request_payloads": getattr(args, "reconstruction_stats", None),
        "n_errors": sum(1 for r in all_rows if r.get("status") == "error"),
        "agreement_mass": round(sum(mass) / len(mass), 4) if mass else None,
        "agreement_mass_k_n": [sum(mass), len(mass)],
        "agreement_mass_wilson95": wilson(sum(mass), len(mass)),
        "agreement_mass_bootstrap95": bootstrap_ci(mass),
        "first_decision_bound": round(sum(s1) / len(s1), 4) if s1 else None,
        "first_decision_k_n": [sum(s1), len(s1)],
        "first_decision_wilson95": wilson(sum(s1), len(s1)),
        "hazard_by_step": hazard,
        "field_audit_step_missing_counts": missing_counts,
        "field_audit_step_notes": note_counts,
        "run_level_audit": sorted({json.dumps(ep["run_audit"], sort_keys=True) for ep in episodes}),
        "episodes": per_ep,
    }
    if args.expect_same_model:
        out["reproducibility_check"] = {
            "passed": bool(replayed) and all(r["agree"] for r in replayed) and len(mass) == sum(mass),
            "steps_reproduced": f"{sum(1 for r in replayed if r['agree'])}/{len(replayed)}",
            "steps_raw_exact": f"{sum(1 for r in replayed if r.get('exact_raw_match'))}/"
                               f"{sum(1 for r in replayed if r.get('exact_raw_match') is not None)}",
        }
    return out


# ---------------------------------------------------------------------------


async def main_async(args) -> int:
    mem_pat = re.compile(args.memory_tool_regex, re.I)
    episodes = []
    for path in find_logs(args.logs):
        log = read_eval_log(path, resolve_attachments=True)
        run_audit = run_level_audit(log)
        for s in log.samples or []:
            mode = (s.metadata or {}).get("eval_mode")
            if args.mode and str(mode).lower().split(".")[-1] != args.mode:
                continue
            steps = episode_steps(s, args.agent_role, args.include_content)
            if not steps:
                continue
            if args.max_steps:
                steps = steps[: args.max_steps]
            episodes.append({"log": os.path.basename(path), "sample_id": str(s.id), "epoch": s.epoch,
                             "eval_mode": str(mode), "steps": steps, "run_audit": run_audit})
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    rec_stats = await fill_requests(episodes, args.check_reconstruction)
    for ep in episodes:
        ep["audit"] = step_audit(ep["steps"])
    print(f"episodes: {len(episodes)}; logged agent steps: {sum(len(e['steps']) for e in episodes)}")
    print("request payloads:", json.dumps(rec_stats))
    args.reconstruction_stats = rec_stats

    os.makedirs(args.out_dir, exist_ok=True)
    if args.audit_only:
        audit = {"n_episodes": len(episodes), "n_logged_steps_total": sum(len(e["steps"]) for e in episodes),
                 "request_payloads": rec_stats, "step_missing_counts": {}, "step_notes": {},
                 "request_sampling_examples": [], "run_level_audit": sorted({json.dumps(e["run_audit"], sort_keys=True) for e in episodes})}
        for ep in episodes:
            for st, a in zip(ep["steps"], ep["audit"]):
                for m in a["missing"]:
                    audit["step_missing_counts"][m] = audit["step_missing_counts"].get(m, 0) + 1
                for m in a["notes"]:
                    audit["step_notes"][m] = audit["step_notes"].get(m, 0) + 1
            if ep["steps"] and ep["steps"][0]["request"] is not None and len(audit["request_sampling_examples"]) < 5:
                audit["request_sampling_examples"].append(
                    {"sample_id": ep["sample_id"], **{k: ep["steps"][0]["request"].get(k) for k in SAMPLING_FIELDS},
                     "request_keys": sorted(ep["steps"][0]["request"].keys())})
        with open(os.path.join(args.out_dir, "audit.json"), "w") as f:
            json.dump(audit, f, indent=1)
        print(json.dumps(audit, indent=1))
        return 0
    url = args.base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    rows_by_ep: dict = {}

    async def worker(ep):
        async with sem:
            rows = await replay_episode(client, url, ep, args, mem_pat)
            rows_by_ep[(ep["log"], ep["sample_id"], ep["epoch"])] = rows
            first = next((r["step"] for r in rows if r.get("status") == "replayed" and not r["agree"]), None)
            print(f"  {ep['sample_id']}: {len(rows)}/{len(ep['steps'])} steps replayed; first disagreement: {first}", flush=True)

    t0 = time.time()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[worker(ep) for ep in episodes])
    with open(os.path.join(args.out_dir, "steps.jsonl"), "w") as f:
        for ep in episodes:
            for r in rows_by_ep.get((ep["log"], ep["sample_id"], ep["epoch"]), []):
                f.write(json.dumps(r) + "\n")
    summary = summarize(episodes, rows_by_ep, args)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    brief = {k: v for k, v in summary.items() if k not in ("episodes", "hazard_by_step")}
    print(json.dumps(brief, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", nargs="+", required=True, help=".eval files or directories")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None, help="served model name on the replay server")
    ap.add_argument("--audit-only", action="store_true", help="only audit the replay fields; no server needed")
    ap.add_argument("--check-reconstruction", action="store_true",
                    help="rebuild the payload of steps that have a logged call and compare with the logged payload")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--agent-role", default="untrusted")
    ap.add_argument("--mode", default=None, help="keep only samples with this eval_mode (honest or attack)")
    ap.add_argument("--concurrency", type=int, default=1, help="episodes replayed in parallel (steps within an episode stay sequential)")
    ap.add_argument("--include-content", action="store_true", help="also require the text content to match")
    ap.add_argument("--memory-tool-regex", default=r"memory|remember|note")
    ap.add_argument("--all-steps", action="store_true", help="continue past the first disagreement (diagnostics only)")
    ap.add_argument("--expect-same-model", action="store_true", help="reproducibility check: the replay model is the logged model")
    ap.add_argument("--default-max-tokens", type=int, default=4096, help="used only when the logged request has no max_tokens (recorded as imputed)")
    ap.add_argument("--override-temperature", type=float, default=None)
    ap.add_argument("--chat-template-kwargs", default=None, help='JSON, for example \'{"enable_thinking": false}\'')
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()
    if not args.audit_only and not (args.base_url and args.model):
        ap.error("--base-url and --model are required unless --audit-only is given")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

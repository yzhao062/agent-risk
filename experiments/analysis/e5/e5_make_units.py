"""E5 inputs built from E3's pass-all records (E5-PREREG-2026-10-06.md). Needs tau2 v1.0.1.

Reads only the member-B lines of E3's episode logs (every other line is skipped by
its member field before anything else of it is used) and writes two frozen files:

  e5/e5_b_extract.json   one row per E3 unit (1,200): task, seed, whether B is scored,
                         B's first block b under the retry gate (lam, pi), and B's
                         logged outcomes the prediction needs: for an agreement unit
                         (b none) harm (with E3's unknown rule), failure, tokens, tool
                         calls; for a continuing unit the harm before b (H_pre, with
                         the unknown rule on steps < b) and the agent and monitor
                         tokens and agent calls of steps 0..b; plus B's whole-episode
                         cost (the baseline collection cost, reported separately).
  e5/e5_units.json       one entry per continuing unit: the logged messages up to and
                         including step b's proposal (B's results of that proposal are
                         left out), step b's logged per-step record (score, monitor),
                         and the offline restoration (``e5_restore.prefix_checks``):
                         prefix database hash, request digests, counters, check flags.

  ~/venv-tau2/bin/python e5/e5_make_units.py --logroot logs/e3 --manifest e3/manifest_e3.json \
      --lam 11.186319 --pi 0 --member B --out-extract e5/e5_b_extract.json --out-units e5/e5_units.json
  (smoke: --logroot logs/e3_pilot --manifest e3/manifest_e3_pilot.json --member P --units-only 4)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
E3 = os.path.join(HARNESS, "e3")
for p in (E3, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import e3_netguard  # noqa: E402

e3_netguard.install()

COST = ("agent_input_tokens", "agent_output_tokens", "monitor_input_tokens", "monitor_output_tokens")


def read_member(logroot: str, member: str) -> tuple[dict, dict]:
    """episode_id -> record for one member (a scored record wins), and source file sha256s."""
    out: dict = {}
    files = {}
    for p in sorted(glob.glob(os.path.join(logroot, "**", "episodes.jsonl"), recursive=True)):
        files[os.path.relpath(p, logroot)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        with open(p) as f:
            for line in f:
                if f'"member": "{member}"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("member") != member:
                    continue
                eid = r.get("episode_id")
                if eid not in out or (r.get("scored") and not out[eid].get("scored")):
                    out[eid] = r
    return out, files


def check_failed_steps(rec: dict) -> list[int]:
    return [r["step"] for r in rec["per_step"] for x in r["calls"]
            if x.get("dispatched") and not x.get("error") and str(x.get("check_status", "")).startswith("failed")]


def harm_steps(rec: dict) -> list[int]:
    return [r["step"] for r in rec["per_step"] if any(x.get("harm_codes") for x in r["calls"])]


def extract_row(e: dict, rec: dict | None, lam: float, pi: float) -> dict:
    import e5_restore
    row = {"unit": e["unit"], "task_index": e["task_index"], "task_id": e["task_id"], "seed_index": e.get("seed_index"),
           "seed": e["seed"], "episode_id": e["episode_id"], "scored": bool(rec and rec.get("scored"))}
    if not row["scored"]:
        return row
    lo = rec["loss"]
    row["whole_episode_cost"] = {"agent_calls": sum(1 for c in rec["llm_calls"] if c["role"] == "agent"),
                                 **{k: int(lo[k]) for k in COST}, "total_tokens": int(lo["total_tokens"]),
                                 "user_tokens": int(rec.get("user_tokens") or 0),
                                 "probe_tokens": int(rec.get("probe_tokens") or 0)}
    b = e5_restore.first_block(rec, lam, pi)
    row["b"] = b
    hs, cf = harm_steps(rec), check_failed_steps(rec)
    row["pass_all"] = {"harm": int(lo["harm"]), "harm_unknown": int(lo["harm"]) == 0 and bool(cf),
                       "failure": int(lo["failure"])}          # the reference "ignore the gate" (descriptive)
    if b is None:
        h = int(lo["harm"])
        row["agree"] = {"harm": h, "harm_unknown": h == 0 and bool(cf), "failure": int(lo["failure"]),
                        "aborted": 0, "total_tokens": int(lo["total_tokens"]), "tool_calls": int(lo["tool_calls"]),
                        "agent_calls": row["whole_episode_cost"]["agent_calls"], "end": rec.get("end"),
                        **{k: int(lo[k]) for k in COST}}
    else:
        hp = int(any(t < b for t in hs))
        tok = e5_restore.prefix_tokens(rec, b)
        row["cont"] = {"h_pre": hp, "h_pre_unknown": hp == 0 and any(t < b for t in cf),
                       "prefix_tokens": tok, "logged_score_b": {r["step"]: r for r in rec["per_step"]}[b]["score"]}
    return row


def unit_entry(e: dict, rec: dict, b: int, task, policy: str, lam: float, pi: float) -> dict:
    import e5_restore
    chk = e5_restore.prefix_checks(rec, b, task, policy, lam, pi)
    traj = e5_restore.messages(rec["trajectory"])
    a_idx = e5_restore.assistant_index(traj)
    i = a_idx[b + 1]
    srec = {r["step"]: r for r in rec["per_step"]}[b]
    ac = e5_restore.agent_calls(rec)
    return {"unit": e["unit"], "task_index": e["task_index"], "task_id": e["task_id"], "seed_index": e.get("seed_index"),
            "seed": int(rec["seed"]), "source_episode_id": e["episode_id"], "b": int(b),
            "trajectory": rec["trajectory"][:i + 1],
            "step_record": {k: srec[k] for k in ("step", "n_proposed_calls", "n_write_calls", "score", "gate_decision",
                                                  "monitor")},
            "prefix_db_hash": chk["prefix_db_hash"], "agent_request_b_sha_logged": ac[b]["body_sha256"],
            "expected_first_request_sha": chk["expected_first_request_sha"],
            "step_count": chk["step_count"], "num_errors": chk["num_errors"], "user_turn": chk["user_turn"],
            "offline_checks": {k: chk[k] for k in ("db_results", "db_forward", "agent_request", "user_request",
                                                   "monitor_prompt", "rule", "all_ok", "prefix_calls_replayed",
                                                   "forward_calls_replayed")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--pi", type=float, required=True)
    ap.add_argument("--member", default="B")
    ap.add_argument("--out-extract", default=None)
    ap.add_argument("--out-units", required=True)
    ap.add_argument("--units-only", type=int, default=None, help="smoke: the first N continuing units, no extract")
    a = ap.parse_args()
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    from tau2.domains.airline.environment import get_environment, get_tasks
    tasks = {t.id: t for t in get_tasks("base")}
    policy = get_environment().get_policy()
    man_bytes = open(a.manifest, "rb").read()
    man = json.loads(man_bytes)
    recs, files = read_member(a.logroot, a.member)
    planned = sorted([e for e in man["episodes"] if e["member"] == a.member], key=lambda e: e["unit"])
    rows, units = [], []
    for e in planned:
        rec = recs.get(e["episode_id"])
        row = extract_row(e, rec, a.lam, a.pi)
        rows.append(row)
        if row.get("b") is not None:
            if a.units_only is not None and len(units) >= a.units_only:
                continue
            units.append(unit_entry(e, rec, row["b"], tasks[e["task_id"]], policy, a.lam, a.pi))
    src = {"logroot": a.logroot, "member": a.member, "manifest": a.manifest,
           "manifest_sha256": hashlib.sha256(man_bytes).hexdigest(), "episode_files_sha256": files,
           "lam": a.lam, "pi": a.pi}
    if a.out_extract and a.units_only is None:
        with open(a.out_extract, "w") as f:
            json.dump({"experiment": "E5", "kind": "b_extract", "source": src, "n_units": len(rows), "rows": rows}, f,
                      indent=1)
    with open(a.out_units, "w") as f:
        json.dump({"experiment": "E5", "kind": "units", "source": src, "n_units": len(units), "units": units}, f)
    ok = sum(1 for u in units if u["offline_checks"]["all_ok"])
    print(json.dumps({"rows": len(rows), "scored": sum(1 for r in rows if r["scored"]),
                      "agreement": sum(1 for r in rows if r.get("scored") and r.get("b") is None),
                      "continuing": sum(1 for r in rows if r.get("b") is not None),
                      "units": len(units), "units_offline_checks_ok": ok, "egress_attempts": list(e3_netguard.ATTEMPTS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

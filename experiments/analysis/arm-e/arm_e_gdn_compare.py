"""Arm E rung 3 diagnostic: compare Qwen3.5 probe outputs across server processes and GPUs.

  python arm_e_gdn_compare.py --results results --data data

Reads results/r3_gdncheck-*/task{0,1}/probe_<backend>_<a|b>.jsonl (two GPUs, two
server processes per GPU and backend) and reports, per Gated DeltaNet prefill
backend, how many of the probe steps gave identical token ids in both processes on
one GPU, in all four runs, and identical decisions in all four runs. It also
compares the default backend's runs with the six rung 3 task processes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_e_common import agree  # noqa: E402


def load(p: str) -> dict:
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "decision":
                out[(r["source"], r["sample_id"], r["step"])] = r
    return out


def tok(r: dict) -> tuple:
    return tuple(r.get("token_ids") or ())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    with open(os.path.join(a.data, "probe_q35_div.json")) as f:
        steps = [tuple(x) for x in json.load(f)["steps"]]
    divergent = set(steps[:16])
    res = {}
    for be in ("flashinfer", "triton", "cutedsl"):
        runs = {}
        for p in sorted(glob.glob(os.path.join(a.results, "r3_gdncheck-*", "task*", f"probe_{be}_*.jsonl"))):
            runs[os.path.relpath(p, a.results)] = load(p)
        if not runs:
            res[be] = {"runs": 0}
            continue
        names = list(runs)
        keys = [k for k in steps if all(k in runs[n] for n in names)]
        same_gpu = {}
        for task in sorted({n.split("/")[1] for n in names}):
            pair = [n for n in names if n.split("/")[1] == task]
            if len(pair) == 2:
                same_gpu[task] = sum(1 for k in keys if tok(runs[pair[0]][k]) == tok(runs[pair[1]][k]))
        all_tok = sum(1 for k in keys if len({tok(runs[n][k]) for n in names}) == 1)
        all_dec = sum(1 for k in keys if all(agree(runs[names[0]][k]["replay"], runs[n][k]["replay"], "raw") for n in names))
        all_tok_div = sum(1 for k in keys if k in divergent and len({tok(runs[n][k]) for n in names}) == 1)
        res[be] = {"runs": len(names), "steps": len(keys),
                   "token_identical_two_processes_same_gpu": same_gpu,
                   "token_identical_all_runs": all_tok, "decision_identical_all_runs": all_dec,
                   "token_identical_all_runs_among_16_divergent": all_tok_div}
    # default backend vs the six rung 3 task processes
    r3 = {p: load(p) for p in glob.glob(os.path.join(a.results, "r3_qwen35-*", "task*", "probe.jsonl"))}
    fi = {p: load(p) for p in glob.glob(os.path.join(a.results, "r3_gdncheck-*", "task*", "probe_flashinfer_*.jsonl"))}
    if r3 and fi:
        allruns = list(r3.values()) + list(fi.values())
        keys = [k for k in steps if all(k in r for r in allruns)]
        res["flashinfer_with_rung3_processes"] = {
            "runs": len(allruns), "steps": len(keys),
            "distinct_token_outputs_per_step": [len({tok(r[k]) for r in allruns}) for k in keys]}
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.join(a.results, "analysis"), exist_ok=True)
    with open(os.path.join(a.results, "analysis", "gdn_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

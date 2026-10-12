"""D64 descriptive diagnostics; NOT part of the registered analysis.

The D64 pre-registration (Run Protocol, revised after review) says block
timing, interruptions, serving changes and group-specific contrasts are
reported descriptively, without excluding triplets or selecting the primary
analysis from them. This script computes those numbers, plus per-condition
abort-step histograms, P(failure) and step counts, from the frozen analysis
output (rows), the episode records and chain.out. Group-specific intervals use
the frozen analyze_d64.paired_with_missing (method 10) and carry no
multiplicity adjustment.

Usage:
  python describe_d64.py --analysis logs/d64/analysis_d64.json --manifest d64/manifest_d64.json \
      --logroot logs/d64 --out logs/d64/descriptive_d64.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ad = _load("analyze_d64")
apm = ad.apm
arp = ad.arp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--logroot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = json.load(open(args.analysis))
    man = json.load(open(args.manifest))
    rows = res["rows"]
    out: dict = {"note": "descriptive only; not registered tests; no multiplicity adjustment"}

    # group-specific contrasts (group g holds triplets 100(g-1) .. 100g-1)
    order = {b["group"]: "".join(x["condition"] for x in man["block_order"] if x["group"] == b["group"])
             for b in man["block_order"]}
    per_group = {}
    for g in sorted(order):
        rr = [r for r in rows if r["triplet"] // 100 + 1 == g]
        col = lambda k: [x[k] for x in rr]
        d = {"order": order[g], "n": len(rr),
             "abort_rate": {c: sum(x[f"{c}_abort"] or 0 for x in rr) / len(rr) for c in ("A", "U", "B")}}
        for name, (a, b) in {"A_minus_U": ("A_abort", "U_abort"), "A_minus_Brep": ("A_abort", "B_abort"),
                             "U_minus_Brep": ("U_abort", "B_abort")}.items():
            p = ad.paired_with_missing(col(a), col(b))
            d[name] = {"cells": p["cells_complete"], "point_range": p["point_range"],
                       "newcombe_paired_95": p["newcombe_paired_95"]}
        per_group[g] = d
    out["per_group_abort"] = per_group

    # U live against B replay: abort concordance and abort steps
    both = [r for r in rows if r["U_abort"] is not None and r["B_abort"] is not None]
    out["U_vs_Brep_abort"] = {
        "both_abort": sum(r["U_abort"] == 1 and r["B_abort"] == 1 for r in both),
        "U_only": sum(r["U_abort"] == 1 and r["B_abort"] == 0 for r in both),
        "B_only": sum(r["U_abort"] == 0 and r["B_abort"] == 1 for r in both),
        "neither": sum(r["U_abort"] == 0 and r["B_abort"] == 0 for r in both),
        "same_step_given_both_abort": sum(r["U_abort"] == 1 and r["B_abort"] == 1
                                          and r["U_abort_step"] == r["B_abort_step"] for r in both),
        "step0_abort_U_and_B": sum(r["U_abort_step"] == 0 and r["B_abort_step"] == 0 for r in both),
        "step0_abort_U_only": sum(r["U_abort_step"] == 0 and r["B_abort_step"] != 0 for r in both),
        "step0_abort_B_only": sum(r["U_abort_step"] != 0 and r["B_abort_step"] == 0 for r in both),
    }
    au = [r for r in rows if r["A_abort"] is not None and r["U_abort"] is not None]
    out["A_vs_U_step0_abort"] = {
        "both": sum(r["A_abort_step"] == 0 and r["U_abort_step"] == 0 for r in au),
        "A_only": sum(r["A_abort_step"] == 0 and r["U_abort_step"] != 0 for r in au),
        "U_only": sum(r["A_abort_step"] != 0 and r["U_abort_step"] == 0 for r in au),
    }
    for c in ("A", "U", "B"):
        hist: dict = {}
        for r in rows:
            if r[f"{c}_abort"] == 1:
                hist[r[f"{c}_abort_step"]] = hist.get(r[f"{c}_abort_step"], 0) + 1
        out[f"{c}_abort_step_hist"] = dict(sorted(hist.items()))

    # P(failure), executed steps, step-0 submission (A, U observed; B replayed for failure)
    recs = arp.collect(args.logroot, man)
    apm.recompute_flags(list(recs.values()))
    cond = {}
    for c in ("A", "U", "B"):
        ids = [e["episode_id"] for e in man["episodes"] if e["condition"] == c and e["episode_id"] in recs]
        vals = [apm.replay(recs[i], ad.THETA) if c == "B" else apm.observed(recs[i]) for i in ids]
        steps = [int(recs[i]["loss"]["n_steps"]) for i in ids]
        cond[c] = {"n_recovered": len(ids),
                   "failure": sum(int(v["failure"]) for v in vals),
                   "mean_steps_executed": sum(steps) / len(steps),
                   "step0_submission_without_abort": sum(1 for i, v in zip(ids, vals)
                                                         if not v["aborted"] and int(recs[i]["loss"]["n_steps"]) == 1)}
        if c == "B":
            cond[c]["failure_without_gate"] = sum(int(recs[i]["loss"]["failure"]) for i in ids)
            cond[c]["note"] = "failure is the replayed value at theta 0.94; steps are the baseline's executed steps"
    out["per_condition"] = cond

    # block timing and preemption counters from chain.out
    pat = re.compile(r"^\[(\S+)\] (start|done) (block-\d+-\w)\S*.*preemptions agent=(\S+) monitor=(\S+)")
    t: dict = {}
    for line in open(os.path.join(args.logroot, "chain.out")):
        m = pat.match(line)
        if m:
            t.setdefault(m.group(3), {})[m.group(2)] = (datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ"),
                                                         m.group(4), m.group(5))
    out["blocks"] = {b: {"start": v["start"][0].isoformat() + "Z", "done": v["done"][0].isoformat() + "Z",
                         "minutes": round((v["done"][0] - v["start"][0]).total_seconds() / 60, 1),
                         "preemptions_at_done": {"agent": v["done"][1], "monitor": v["done"][2]}}
                     for b, v in t.items() if "start" in v and "done" in v}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

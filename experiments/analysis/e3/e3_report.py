"""E3 reporting step (after the registered analysis; changes no registered computation).

Reads the frozen output of `e3_analyze.py confirm` and applies the reporting-time
corrections that review round 1 of the registration required (E3-PREREG, Medium 2 and Medium 4):

  outcome map   the registered reading of `primary_reading`, then its reporting
                interpretation (Newcombe-only wording), with the task-cluster 95%
                interval beside any nonzero-difference statement and whether it also
                excludes zero;
  abort rate    secondary (i): both intervals; the frozen `equivalent_at_0.03` flag
                labelled "Newcombe-only secondary equivalence";
  live C        secondary (vi): [k/400, (k+m)/400] and the Clopper-Pearson completion
                envelope [CP_lower(k, 400), CP_upper(k+m, 400)], 95% two-sided endpoints;
                the frozen `live_C.exact95` is shown only when m = 0.

  python e3_report.py --analysis logs/e3/analysis_e3.json --out logs/e3/report_e3.json
"""

from __future__ import annotations

import argparse
import json

from scipy.stats import beta

INTERP = {
    "equivalent within delta (both 90% intervals inside), with a nonzero replay error inside the margin":
        "equivalent within delta, with Newcombe-only evidence of a nonzero marginal difference",
    "equivalent within delta (Newcombe 90% and task-cluster bootstrap 90% both inside the margin)":
        "equivalent within delta",
    "discrepancy: the 95% Newcombe envelope excludes 0; equivalence not shown":
        "Newcombe-only evidence of a marginal discrepancy; equivalence not shown",
    "inconclusive": "inconclusive",
}


def cp_lower(k: int, n: int) -> float:
    return 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))


def cp_upper(k: int, n: int) -> float:
    return 1.0 if k >= n else float(beta.ppf(0.975, k + 1, n - k))


def excludes0(iv) -> bool:
    return iv[0] > 0 or iv[1] < 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.analysis))
    pr = d["primary"]
    nc, cb = pr["newcombe"], pr["task_cluster_bootstrap"]
    reading = pr["reading"]
    interp = INTERP.get(reading, reading)
    nonzero = "nonzero" in interp or "discrepancy" in interp
    out = {"delta": d["delta"], "lam": d["lam"], "n_units": d["n_units"],
           "primary": {
               "registered_reading_verbatim": reading,
               "reporting_interpretation": interp,
               "point_range": nc["point_range"],
               "newcombe90": nc["newcombe_paired_90"], "newcombe95": nc["newcombe_paired_95"],
               "cluster90": cb["cluster_boot_90"], "cluster95": cb["cluster_boot_95"],
               "envelope_capped": bool(nc.get("envelope_capped", False)),
               "p_live_range": nc["p1_range"], "p_replay_range": nc["p2_range"],
               "cells_complete": nc.get("cells_complete"), "missing_patterns": nc.get("missing_patterns"),
               "cluster95_excludes_0": excludes0(cb["cluster_boot_95"]),
               "note": ("Newcombe-only exclusion is not a conclusion robust to task clustering. Failure to establish "
                        "equivalence does not establish a difference outside the margin."),
               "nonzero_statement_made": nonzero}}
    ab = d["secondary"]["aborted"]
    out["abort_rate_secondary_i"] = {
        "newcombe90": ab["newcombe"]["newcombe_paired_90"], "newcombe95": ab["newcombe"]["newcombe_paired_95"],
        "cluster90": ab["cluster"]["cluster_boot_90"], "cluster95": ab["cluster"]["cluster_boot_95"],
        "point_range": ab["newcombe"]["point_range"],
        "newcombe_only_secondary_equivalence_at_0.03": ab["equivalent_at_0.03"]}
    sc = d["secondary"]["success"]
    out["success_secondary_ii"] = {"newcombe95": sc["newcombe"]["newcombe_paired_95"],
                                   "cluster95": sc["cluster"]["cluster_boot_95"], "point_range": sc["newcombe"]["point_range"]}
    lc = d["retry_gate"]["live_C"]
    n, k, m = lc["n"], lc["harm_known"], lc["harm_unknown"]
    out["retry_secondary_vi"] = {
        "bounds_from_B_on_C_units": d["retry_gate"]["bounds_from_B_on_C_units"],
        "live_C": {"n": n, "k": k, "m": m, "range": [k / n, (k + m) / n],
                   "cp_completion_envelope95": [cp_lower(k, n), cp_upper(k + m, n)],
                   "frozen_exact95": lc["exact95"] if m == 0 else "suppressed (m > 0)", "ends": lc["ends"]},
        "live_inside_plugin_bounds": d["retry_gate"]["live_inside_plugin_bounds"]}
    for key in ("certification", "descriptive"):
        out[key] = d[key]
    out["secondary_iii_iv"] = {k2: d["secondary"][k2] for k2 in d["secondary"] if k2 not in ("aborted", "success")}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({"primary": out["primary"], "abort_rate_secondary_i": out["abort_rate_secondary_i"],
                      "retry_secondary_vi": out["retry_secondary_vi"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

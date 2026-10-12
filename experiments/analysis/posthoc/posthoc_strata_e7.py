"""Post hoc E7 Check of execution across strata (Appendix app:budget-strata, Propositions prop:strata-limits and prop:strata-achieve).
Strata are the registered budgeted-analysis strata (first-block step 0-3, 4-7, 8+; analyze_e7.BINS), built by the
registered budget_setup from the frozen prediction rows: settled or unresolvable units count harmful (q0), each
stratum's harm rate is the executed continuations' rate (unknown counted harmful), and its cost is the mean recorded
agent calls per continuation. Live cost is recorded live agent calls per episode. Reads logs read-only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS_DEFAULT = Path(os.environ.get("AGENT_RISK_LOGS", str(REPO / "data" / "logs"))) / "e7"
ANALYZE_E7_PATH = REPO / "experiments" / "analysis" / "analyze_e7.py"

parser = argparse.ArgumentParser(description="Post hoc E7 Check of execution across strata")
parser.add_argument("--logs", default=str(LOGS_DEFAULT), help="Path to logs/e7 directory")
parser.add_argument("--analyze-e7", default=str(ANALYZE_E7_PATH), help="Path to analyze_e7.py")
parser.add_argument("--out", default=str(HERE / "posthoc_strata_e7.json"), help="Output json path")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("analyze_e7", args.analyze_e7)
a7 = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(args.analyze_e7).parent))
spec.loader.exec_module(a7)

pred = json.load(open(os.path.join(args.logs, "prediction_e7.json")))
ana = json.load(open(os.path.join(args.logs, "analysis_e7.json")))
ALPHA, BETA = 0.05, 0.2
LAST = 14   # 15-step cap: a first block at step 14 leaves no agent step, so harm is fixed at H_pre (retry_bounds feasible sets)
_orig_setup = a7.budget_setup


def budget_setup_settled(rows):
    """Registered budget_setup with final-step prefixes settled: no agent step remains, so their harm is H_pre (0 here)."""
    kept, A_extra = [], 0
    for r in rows:
        if r["kind"] not in ("agreement", "no_base") and r.get("h_pre") != 1 and int(r["b"]) == LAST:
            A_extra += int(r["harm"] is None or r["harm"] == 1)   # unknown counts harmful, as registered
            r = dict(r, kind="agreement")                       # settled without execution
        kept.append(r)
    st = _orig_setup(kept)
    st["final_step_settled"] = sum(1 for r in rows if r["kind"] not in ("agreement", "no_base") and r.get("h_pre") != 1 and int(r["b"]) == LAST)
    return st


a7.budget_setup = budget_setup_settled


def kl(a, b):
    t = 0.0
    if a > 0:
        t += a * math.log(a / b)
    if a < 1:
        t += (1 - a) * math.log((1 - a) / (1 - b))
    return t


def kl_inv_up(p, level):
    """z in [p,1) with kl(p,z) = level."""
    lo, hi = p, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if kl(p, mid) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def saddle(w, p, c, gam, zeta):
    S = len(w)
    u = [1 - g for g in gam]
    G = sum(w[s] * min(gam[s], 1 - p[s]) for s in range(S))
    Delta = zeta - G
    assert Delta > 0, (zeta, G)
    J = [s for s in range(S) if p[s] < u[s]]

    def fill(t):
        return sum(w[s] * (min(u[s], kl_inv_up(p[s], c[s] * t)) - p[s]) for s in J)
    lo, hi = 0.0, 1.0
    while fill(hi) < Delta:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if fill(mid) < Delta:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    r = [min(u[s], kl_inv_up(p[s], c[s] * t)) if s in J else p[s] for s in range(S)]
    U = [s for s in J if r[s] < u[s] - 1e-12]
    num = {s: c[s] * w[s] * r[s] * (1 - r[s]) / (r[s] - p[s]) for s in U}
    lam = [num.get(s, 0.0) / sum(num.values()) for s in range(S)]
    inv = [lam[s] / c[s] for s in range(S)]
    a = [x / sum(inv) for x in inv]
    return {"t_star": t, "T_star": 1 / t, "r_star": r, "lambda_cost": lam, "a_exec": a, "Delta": Delta}


def D_of(a, w, p, gam, Delta):
    """inf over the observable null of sum_s a_s kl(p_s, r_s) (gamma unclipped here), by the KKT root."""
    S = len(w)

    def r_of(nu):
        out = []
        for s in range(S):
            d = nu * w[s] / a[s]
            rr = (d - 1 + math.sqrt((1 - d) ** 2 + 4 * d * p[s])) / (2 * d)
            out.append(min(1 - gam[s], rr))
        return out

    def inc(nu):
        return sum(w[s] * (r_of(nu)[s] - p[s]) for s in range(S))
    lo, hi = 1e-12, 1.0
    while inc(hi) < Delta:
        hi *= 2
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if inc(mid) < Delta:
            lo = mid
        else:
            hi = mid
    r = r_of(hi)
    return sum(a[s] * kl(p[s], r[s]) for s in range(S)), r


out = {"alpha": ALPHA, "beta": BETA, "gates": {}}
KLE = kl(1 - BETA, ALPHA)
for g, qstar in (("1", 0.30), ("2", 0.25)):
    rows = pred["gates"][g]["rows"]
    st = a7.budget_setup(rows)
    n, A = st["n"], st["A"]
    keys = sorted(st["strata"])
    R = [len(st["strata"][h]) for h in keys]
    x = [sum(y for y, _ in st["strata"][h]) for h in keys]
    cmean = [sum(cc for _, cc in st["strata"][h]) / len(st["strata"][h]) for h in keys]
    w = [Rh / n for Rh in R]
    p = [x[i] / R[i] for i in range(len(R))]
    q0 = A / n
    q = q0 + sum(w[i] * p[i] for i in range(len(w)))
    zeta = qstar - q
    gam = [0.0] * len(w)
    sd = saddle(w, p, cmean, gam, zeta)
    # proportional allocation (registered design shape) and uniform-by-count
    m = sum(w)
    a_prop = [wi / m for wi in w]
    D_prop, _ = D_of(a_prop, w, p, gam, sd["Delta"])
    cbar_prop = sum(a_prop[i] * cmean[i] for i in range(len(w)))
    a_opt = sd["a_exec"]
    D_opt, _ = D_of(a_opt, w, p, gam, sd["Delta"])
    cbar_opt = sum(a_opt[i] * cmean[i] for i in range(len(w)))
    assert abs(D_opt / cbar_opt - sd["t_star"]) < 1e-6 * sd["t_star"], (D_opt / cbar_opt, sd["t_star"])
    # brute-force check of the max over execution fractions on a simplex grid
    best = 0.0
    N = 200
    for i in range(1, N):
        for j in range(1, N - i):
            aa = [i / N, j / N, (N - i - j) / N]
            Dv, _ = D_of(aa, w, p, gam, sd["Delta"])
            best = max(best, Dv / sum(aa[k] * cmean[k] for k in range(3)))
    assert best <= sd["t_star"] * (1 + 1e-6), (best, sd["t_star"])
    assert best >= sd["t_star"] * (1 - 2e-3), (best, sd["t_star"])
    # homogeneous F.8 reading of the pooled pool
    p_pool = sum(w[i] * p[i] for i in range(len(w))) / m
    pbar = p_pool + zeta / m
    ex_F8 = KLE / kl(p_pool, pbar)
    live_calls = ana["gates"][g]["execution_cost"]["live"]["agent_calls"] / n
    lv_calls = KLE * live_calls / kl(q, qstar)
    aware, _ = D_of(w, w, p, gam, zeta)          # live test that trusts q0 and w and sees each episode's stratum
    aware_calls = KLE * live_calls / aware
    res = {"q_star": qstar, "n": n, "A_settled_or_counted_harmful": A, "q0": q0, "strata_bins": [list(b) for b in a7.BINS],
           "R": R, "harms": x, "w": w, "p": p, "mean_calls": cmean, "q": q, "zeta": zeta,
           "t_star": sd["t_star"], "r_star": sd["r_star"], "lambda_cost": sd["lambda_cost"], "a_exec_opt": a_opt,
           "grid_max_info_per_call": best,
           "calls_bound_opt": KLE * sd["T_star"], "exec_bound_opt": KLE * sd["T_star"] / cbar_opt,
           "a_exec_prop": a_prop, "calls_bound_prop": KLE * cbar_prop / D_prop, "exec_bound_prop": KLE / D_prop,
           "F8_pooled_exec_bound": ex_F8, "F8_pooled_calls_at_prop_cost": ex_F8 * cbar_prop,
           "live_calls_per_episode": live_calls, "live_calls_bound": lv_calls,
           "crossover_ratio_opt": sd["T_star"] * kl(q, qstar) / live_calls,
           "saving_opt_vs_prop": 1 - (KLE * sd["T_star"]) / (KLE * cbar_prop / D_prop),
           "aware_live_calls_bound": aware_calls, "aware_live_episodes_bound": KLE / aware,
           "aware_ratio_of_bounds": (KLE * sd["T_star"]) / aware_calls}
    out["gates"][g] = res
    print(f"gate {g} q*={qstar}: q0={q0:.4f} R={R} harms={x} p={[round(v, 4) for v in p]} calls={[round(v, 2) for v in cmean]} q={q:.4f} zeta={zeta:.4f}")
    print(f"   a_opt={[round(v, 3) for v in a_opt]} a_prop={[round(v, 3) for v in a_prop]} lam_cost={[round(v, 3) for v in sd['lambda_cost']]}")
    print(f"   bound calls opt {res['calls_bound_opt']:.1f} (exec {res['exec_bound_opt']:.1f})  prop {res['calls_bound_prop']:.1f} (exec {res['exec_bound_prop']:.1f})  saving {res['saving_opt_vs_prop']:.3f}")
    print(f"   F8 pooled exec {ex_F8:.1f} (calls at prop cost {ex_F8 * cbar_prop:.1f})  live calls/ep {live_calls:.2f} live bound calls {lv_calls:.1f} episodes {lv_calls / live_calls:.1f}  crossover {res['crossover_ratio_opt']:.3f}  aware {aware_calls:.1f} calls {KLE / aware:.2f} episodes ratio {res['aware_ratio_of_bounds']:.3f}")
json.dump(out, open(args.out, "w"), indent=1)

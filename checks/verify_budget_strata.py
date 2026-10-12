#!/usr/bin/env python
"""Checks for execution across strata (Appendix F.9 (app:budget-strata)).

Paper map: Proposition F.9.1 (prop:strata-limits; necessity, scalar root, saddle point, small-gap form, one-stratum reduction),
Proposition F.9.2 (prop:strata-achieve; T8 checks the stratified design-point test with unlimited pools; T5 checks the
crossover equivalence),
Remark F.9.3 (rem:strata-scope; counterexamples), and the E7 Check (supplement/appendix-F2.md). Nothing under logs/ is read: the E7 inputs are the
stratum counts, harms, and mean recorded agent calls quoted in the E7 Check (supplement/appendix-F2.md), produced from the registered
prediction rows by posthoc_strata_e7.py (ARTIFACT.md, command A36).

Run with: python verify_budget_strata.py     (about two minutes; numpy, standard library otherwise)

Checks:
- (T1) The identity qbar(p) = q + xi, and the clipping counterexample of part (i): the error mass 0.10 exceeds the gap
  0.08, xi = 0.025, every null world has restored rate above 0.16 in stratum 2, and with n = 4,000 the
  1,000-execution test has Hoeffding size bound e^-6.05 and failure bound e^-6.272 + e^-500 (pool shortage), both
  below 0.0024; exact binomial tails, each summed directly, confirm both error probabilities.
- (T2) Parts (ii) and (iii): on random two-stratum instances, some with clipping and unequal costs, a direct grid search
  of max over cost shares of the inner infimum (no KKT) matches the scalar root t_*; the saddle inequality holds at
  random feasible points; every stratum in J_* has kl(p_s, r*_s)/c_s = t_*; a three-stratum mesh agrees.
- (T3) Part (iv): 2 V^2 t_* / zeta'^2 -> 1 and the execution shares tend to the cost-weighted Neyman allocation as
  zeta' -> 0; with S = 1 and the E1 values, the bound is the 12.3 executions of the E1 Check (supplement/appendix-F2.md).
- (T5) Proposition F.9.2 (prop:strata-achieve(iii)): on random instances, t_* > t_L exactly when the root equation at t_L falls
  below zeta'.
- (T6) Remark F.9.3 (rem:strata-scope): the fixed-cohort bound exceeds 13,000; the biased-restoration example.
- (T7) The E7 Check values (supplement/appendix-F2.md) from the quoted stratum inputs, including the live lower bound for a test that trusts the
  log's masses and sees each episode's stratum, kl(1 - beta, alpha) / min {sum_s w_s kl(p_s, l_s) : q0 + sum_s w_s l_s >= q*}.
- (T8) Remark F.9.3 (rem:strata-scope) and Proposition F.9.2 (prop:strata-achieve): by Monte Carlo (100,000 runs, unlimited pools), the
  stratified execution test with the design shares has per-draw mean of e^Z equal to one and size at most alpha at a
  boundary point of A; the same increments with draws in proportion to w_s have mean above one and size about 0.15; at
  W_1 its power is at least 1 - beta' and its mean cost lies between the bounds of part (iii)."""
from math import comb, exp, lgamma, log, sqrt
import random

import numpy as np


def kl(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(a > 0, a * np.log(a / b), 0.0)
        y = np.where(a < 1, (1 - a) * np.log((1 - a) / (1 - b)), 0.0)
    return x + y


def kl_inv(p, level):
    lo, hi = p, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if kl(p, mid) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def root(q0, w, p, gam, c, qs):
    """The scalar root t_*, r*, the cost shares lambda*, and the execution shares of Proposition prop:strata-limits."""
    S = len(w)
    xi = sum(w[s] * min(gam[s], 1 - p[s]) for s in range(S))
    q = q0 + sum(w[s] * p[s] for s in range(S))
    zp = qs - q - xi
    assert zp > 0
    J = [s for s in range(S) if p[s] < 1 - gam[s]]

    def fill(t):
        return sum(w[s] * (min(1 - gam[s], kl_inv(p[s], c[s] * t)) - p[s]) for s in J)
    lo, hi = 0.0, 1.0
    while fill(hi) < zp:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if fill(mid) < zp:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    r = [min(1 - gam[s], kl_inv(p[s], c[s] * t)) if s in J else p[s] for s in range(S)]
    Js = [s for s in J if r[s] < 1 - gam[s] - 1e-12]
    num = [c[s] * w[s] * r[s] * (1 - r[s]) / (r[s] - p[s]) if s in Js else 0.0 for s in range(S)]
    lam = [x / sum(num) for x in num]
    inv = [lam[s] / c[s] for s in range(S)]
    a = [x / sum(inv) for x in inv]
    return {"t": t, "r": r, "lam": lam, "a": a, "Js": Js, "J": J, "xi": xi, "zp": zp, "q": q, "fill": fill}


def inner(q0, w, p, gam, a, qs):
    """inf over the closed null of sum_s a_s kl(p_s, r_s), by the multiplier root (zero-weight strata free)."""
    S = len(w)
    u = [1 - g for g in gam]
    need = qs - q0 - sum(w[s] * (u[s] + gam[s] if (a[s] == 0 and p[s] < u[s]) else min(p[s] + gam[s], 1)) for s in range(S))
    if need <= 0:
        return 0.0

    def r_of(nu):
        out = []
        for s in range(S):
            if a[s] == 0:
                out.append(max(p[s], u[s]))
                continue
            d = nu * w[s] / a[s]
            rr = (d - 1 + sqrt((1 - d) ** 2 + 4 * d * p[s])) / (2 * d)
            out.append(min(u[s], max(p[s], rr)) if p[s] < u[s] else p[s])
        return out

    def live(nu):
        return q0 + sum(w[s] * min(r_of(nu)[s] + gam[s], 1) for s in range(S))
    lo, hi = 1e-14, 1.0
    while live(hi) < qs:
        hi *= 2
    for _ in range(90):
        mid = sqrt(lo * hi)
        if live(mid) < qs:
            lo = mid
        else:
            hi = mid
    r = r_of(hi)
    return float(sum(a[s] * kl(p[s], r[s]) for s in range(S)))


def ok(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    print(f"PASS {name} {detail}")


# ---------------------------------------------------------------- (T1)
q0, w, p, gam, qs = 0.0, [0.5, 0.5], [0.95, 0.05], [0.2, 0.0], 0.58
q = q0 + sum(wi * pi for wi, pi in zip(w, p))
xi = sum(w[s] * min(gam[s], 1 - p[s]) for s in range(2))
qbar = q0 + sum(w[s] * min(p[s] + gam[s], 1) for s in range(2))
ok("T1 identity qbar(p) = q + xi", abs(qbar - (q + xi)) < 1e-15, f"{qbar:.4f} = {q:.4f} + {xi:.4f}")
ok("T1 counterexample masses", abs(sum(wi * gi for wi, gi in zip(w, gam)) - 0.10) < 1e-15 and abs(qs - q - 0.08) < 1e-12 and abs(xi - 0.025) < 1e-15,
   "sum w gamma 0.10 > zeta 0.08, xi 0.025")
ok("T1 null worlds need restored rate above 0.16 in stratum 2", abs((qs - w[0] * 1.0) / w[1] - 0.16) < 1e-12)
hb = exp(-2 * 1000 * 0.055 ** 2)
hf = exp(-2 * 1000 * 0.056 ** 2) + exp(-500)
size = sum(comb(1000, k) * 0.16 ** k * 0.84 ** (1000 - k) for k in range(0, 106))
fail = sum(comb(1000, k) * 0.05 ** k * 0.95 ** (1000 - k) for k in range(106, 1001))
short = sum(comb(4000, k) for k in range(0, 1000)) / 2 ** 4000
ok("T1 Hoeffding bounds and exact errors of the 1,000-execution test with n = 4,000",
   hb < 0.0024 and hf < 0.0024 and size <= hb and fail + short <= hf and short <= exp(-500),
   f"size bound {hb:.5f}, failure bound {hf:.5f}, exact size {size:.2e}, exact failure {fail:.2e}, P(N_2 < 1000) {short:.1e}")

# ---------------------------------------------------------------- (T2)
rng = random.Random(20261007)
worst = 0.0
for it in range(12):
    while True:
        p2 = [rng.uniform(0.02, 0.9), rng.uniform(0.02, 0.9)]
        g2 = [rng.choice([0.0, rng.uniform(0, 0.3)]), rng.choice([0.0, rng.uniform(0, 0.3)])]
        if it % 3 == 0:
            g2[0] = 1 - p2[0] + rng.uniform(0.01, 0.2)        # stratum 1 clipped from the start (outside J)
        w2 = [rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)]
        c2 = [rng.uniform(0.5, 4), rng.uniform(0.5, 4)]
        q02 = rng.uniform(0, 0.1)
        qq = q02 + w2[0] * p2[0] + w2[1] * p2[1]
        xi2 = sum(w2[s] * min(g2[s], 1 - p2[s]) for s in range(2))
        top = q02 + w2[0] + w2[1]
        if qq + xi2 + 0.01 < top - 0.01:
            qs2 = rng.uniform(qq + xi2 + 0.005, top - 0.005)
            break
    R = root(q02, w2, p2, g2, c2, qs2)
    # direct: max over lambda of min over the boundary of the box (no KKT), on grids
    u = [1 - g for g in g2]
    lo = [p2[s] for s in range(2)]
    hi = [max(p2[s], u[s]) for s in range(2)]
    d1 = np.linspace(lo[0], hi[0], 20001)
    r2 = np.clip(p2[1] + (qs2 - q02 - w2[0] * np.minimum(d1 + g2[0], 1) - w2[1] * min(p2[1] + g2[1], 1)) / w2[1], p2[1], None)
    # where stratum 2 must rise: r2 so that w2*min(r2+g,1) covers the remainder
    need2 = (qs2 - q02 - w2[0] * np.minimum(d1 + g2[0], 1)) / w2[1] - g2[1]
    r2 = np.maximum(p2[1], need2)
    feas = r2 <= max(p2[1], u[1]) + 1e-12
    k1 = np.minimum(kl(p2[0], np.minimum(d1, 1 - 1e-15)) / c2[0], 1e6)
    k2 = np.minimum(kl(p2[1], np.minimum(r2, 1 - 1e-15)) / c2[1], 1e6)
    lams = np.linspace(0, 1, 2001)[:, None]
    obj = np.where(feas[None, :], lams * k1[None, :] + (1 - lams) * k2[None, :], np.inf)
    direct = float(np.max(np.min(obj, axis=1)))
    rel = abs(direct - R["t"]) / R["t"]
    worst = max(worst, rel)
    # saddle inequality at random feasible points, and equal kl/c on J_*
    for _ in range(200):
        z = [rng.uniform(lo[s], hi[s]) for s in range(2)]
        if q02 + sum(w2[s] * min(z[s] + g2[s], 1) for s in range(2)) >= qs2:
            val = sum(R["lam"][s] * kl(p2[s], min(z[s], 1 - 1e-15)) / c2[s] for s in range(2))
            assert val >= R["t"] * (1 - 1e-9), (val, R["t"])
    for s in R["Js"]:
        assert abs(kl(p2[s], R["r"][s]) / c2[s] - R["t"]) < 1e-9 * max(1, R["t"])
ok("T2 scalar root equals direct max-min on 12 two-stratum instances", worst < 2e-3, f"max relative difference {worst:.1e}")
ok("T2 saddle inequality at 2,400 random feasible points; kl/c = t_* on J_*", True)
# three strata, mesh over the simplex of cost shares against the root
w3, p3, g3, c3, q03 = [0.2, 0.15, 0.1], [0.3, 0.1, 0.6], [0.05, 0.0, 0.1], [3.0, 1.0, 2.0], 0.05
qs3 = q03 + sum(a * b for a, b in zip(w3, p3)) + sum(w3[s] * min(g3[s], 1 - p3[s]) for s in range(3)) + 0.04
R3 = root(q03, w3, p3, g3, c3, qs3)
best = 0.0
N = 60
for i in range(N + 1):
    for j in range(N + 1 - i):
        lam = [i / N, j / N, (N - i - j) / N]
        a = [lam[s] / c3[s] for s in range(3)]
        tot = sum(a)
        a = [x / tot for x in a]
        val = inner(q03, w3, p3, g3, a, qs3) / sum(a[s] * c3[s] for s in range(3))
        best = max(best, val)
ok("T3a three-stratum mesh never exceeds t_* and comes within 1%", best <= R3["t"] * (1 + 1e-7) and best >= 0.99 * R3["t"],
   f"mesh {best:.6f}, root {R3['t']:.6f}")

# ---------------------------------------------------------------- (T3)
w4, p4, g4, c4, q04 = [0.2, 0.15, 0.1], [0.3, 0.1, 0.6], [0.02, 0.0, 0.05], [3.0, 1.0, 2.0], 0.05
V = sum(w4[s] * sqrt(p4[s] * (1 - p4[s]) * c4[s]) for s in range(3))
ney = [w4[s] * sqrt(p4[s] * (1 - p4[s]) / c4[s]) for s in range(3)]
ney = [x / sum(ney) for x in ney]
prev = None
for zp in (0.02, 0.005, 0.001, 0.0002):
    qs4 = q04 + sum(a * b for a, b in zip(w4, p4)) + sum(w4[s] * g4[s] for s in range(3)) + zp
    R4 = root(q04, w4, p4, g4, c4, qs4)
    ratio = 2 * V ** 2 * R4["t"] / zp ** 2
    err = max(abs(R4["a"][s] - ney[s]) for s in range(3))
    print(f"      zeta' {zp}: 2 V^2 t_* / zeta'^2 = {ratio:.6f}, share error {err:.2e}")
    if prev is not None:
        assert abs(ratio - 1) < abs(prev[0] - 1) and err < prev[1]
    prev = (ratio, err)
ok("T3 small-gap limit and cost-weighted Neyman shares", abs(prev[0] - 1) < 1e-3 and prev[1] < 1e-3)
# S = 1 at the E1 values: n = 1200, N = 479, b = 27, D = 8, q* = 0.10, gamma = 0
R1 = root(27 / 1200, [479 / 1200], [8 / 479], [0.0], [1.0], 0.10)
pbar = 8 / 479 + (0.10 - 35 / 1200) / (479 / 1200)
klE = float(kl(0.8, 0.05))
ok("T3 one stratum: r* = pbar and the E1 bound 12.3", abs(R1["r"][0] - pbar) < 1e-9 and round(klE / R1["t"], 1) == 12.3,
   f"r* {R1['r'][0]:.6f} = pbar {pbar:.6f}, bound {klE / R1['t']:.2f}")

# ---------------------------------------------------------------- (T5)
agree = 0
for it in range(200):
    S = 3
    pr = [rng.uniform(0.02, 0.8) for _ in range(S)]
    gr = [rng.choice([0.0, rng.uniform(0, 0.2)]) for _ in range(S)]
    wr = [rng.uniform(0.05, 0.3) for _ in range(S)]
    cr = [rng.uniform(0.3, 5) for _ in range(S)]
    q0r = rng.uniform(0, 0.1)
    qq = q0r + sum(a * b for a, b in zip(wr, pr))
    xir = sum(wr[s] * min(gr[s], 1 - pr[s]) for s in range(S))
    top = q0r + sum(wr)
    if not (qq + xir + 0.01 < top - 0.01):
        continue
    qsr = rng.uniform(qq + xir + 0.005, min(top - 0.005, 0.99))
    Rr = root(q0r, wr, pr, gr, cr, qsr)
    tL = float(kl(qq, qsr)) / rng.uniform(0.5, 20)
    assert (Rr["t"] > tL) == (Rr["fill"](tL) < Rr["zp"]), it
    agree += 1
ok("T5 crossover: t_* > t_L exactly when the root equation at t_L is below zeta'", agree > 100, f"{agree} instances")

# ---------------------------------------------------------------- (T6)
fc = float(kl(0.95, 0.05) / kl(0.49, 0.5))
ok("T6 fixed cohort: the Bernoulli bound exceeds 13,000 against a census of 100", fc > 13000, f"{fc:.0f}")
ok("T6 biased restoration: restored 0.3 lies within 0.1 of live 0.2 and of unsafe live 0.4 (q* 0.35, zeta' 0.05)",
   abs(0.3 - 0.2) <= 0.1 + 1e-12 and abs(0.4 - 0.3) <= 0.1 + 1e-12 and 0.4 > 0.35 and abs((0.35 - 0.2) - 0.1 - 0.05) < 1e-12)

# ---------------------------------------------------------------- (T7)
# Last-step first blocks (step 14 of the 15-step cap) leave no agent step, so their harm is fixed: they are settled
# harmless, not unresolved (69 at gate 2, 11 at gate 1; harm 0 and no agent call each).
E7 = {"2": dict(qs=0.25, A=722, R=[986, 879, 661], x=[259, 129, 31], calls=[11.802231237322514, 8.249146757679181, 3.5733736762481088],
                live=102244 / 7200),
      "1": dict(qs=0.30, A=1761, R=[254, 300, 168], x=[148, 85, 18], calls=[12.125984251968504, 8.69, 3.8392857142857144],
                live=102631 / 7200)}
out = {}
for g, d in E7.items():
    n = 7200
    w = [r / n for r in d["R"]]
    p = [d["x"][s] / d["R"][s] for s in range(3)]
    q0 = d["A"] / n
    gam = [0.0] * 3
    R = root(q0, w, p, gam, d["calls"], d["qs"])
    m = sum(w)
    aprop = [x / m for x in w]
    Dp = inner(q0, w, p, gam, aprop, d["qs"])
    cprop = sum(aprop[s] * d["calls"][s] for s in range(3))
    copt = sum(R["a"][s] * d["calls"][s] for s in range(3))
    pp = sum(w[s] * p[s] for s in range(3)) / m
    pooled = klE / float(kl(pp, pp + (d["qs"] - R["q"]) / m))
    live = klE * d["live"] / float(kl(R["q"], d["qs"]))
    aware = inner(q0, w, p, [0.0] * 3, w, d["qs"])        # live test that trusts q0 and w and sees each episode's stratum
    out[g] = dict(q0=q0, q=R["q"], zeta=d["qs"] - R["q"], opt=klE / R["t"], opt_ex=klE / R["t"] / copt, a=R["a"],
                  prop=klE * cprop / Dp, prop_ex=klE / Dp, pooled=pooled, live=live, live_ep=live / d["live"],
                  ratio=float(kl(R["q"], d["qs"])) / (d["live"] * R["t"]), p=p, calls=d["calls"],
                  aware_ep=klE / aware, aware=klE * d["live"] / aware, aware_ratio=(klE / R["t"]) / (klE * d["live"] / aware))
o2, o1 = out["2"], out["1"]
ok("T7 E7 gate 2 inputs", [round(v, 3) for v in o2["p"]] == [0.263, 0.147, 0.047] and [round(v, 1) for v in o2["calls"]] == [11.8, 8.2, 3.6]
   and round(o2["q0"], 3) == 0.100 and round(o2["q"], 4) == 0.1585 and round(o2["zeta"], 4) == 0.0915)
ok("T7 E7 gate 2 optimal bound 92.3 calls (11.9 executions), shares 0.31, 0.36, 0.34",
   round(o2["opt"], 1) == 92.3 and round(o2["opt_ex"], 1) == 11.9 and [round(v, 2) for v in o2["a"]] == [0.31, 0.36, 0.34],
   f"{o2['opt']:.2f} calls, {o2['opt_ex']:.2f} executions, shares {[round(v, 3) for v in o2['a']]}")
ok("T7 E7 gate 2 proportional 96.1 calls (11.4), pooled 12.2 executions",
   round(o2["prop"], 1) == 96.1 and round(o2["prop_ex"], 1) == 11.4 and round(o2["pooled"], 1) == 12.2)
ok("T7 E7 gate 2 live bound 1,098 calls (77.3 episodes at 14.2), ratio 0.084",
   round(o2["live"]) == 1098 and round(o2["live_ep"], 1) == 77.3 and round(o2["ratio"], 3) == 0.084)
ok("T7 E7 gate 1 at 0.30: proportional 169 and optimal 155 calls; savings 4% and 8%",
   round(o1["prop"]) == 169 and round(o1["opt"]) == 155 and round(100 * (1 - o2["opt"] / o2["prop"])) == 4
   and round(100 * (1 - o1["opt"] / o1["prop"])) == 8)

# ---------------------------------------------------------------- (T8)
def design_test(pd, w, gam, c, q0, qs, hat, alpha, betap, reps, seed, shares=None, maxsteps=4000):
    """Monte Carlo of the stratified execution test with unlimited pools: design point from design values pd,
    draws by the design shares (or by the given shares), bits Bernoulli(hat)."""
    R = root(q0, w, pd, gam, c, qs)
    S = len(w)
    pbar = R["r"]
    a = np.array(R["a"] if shares is None else shares, float)
    ih = np.array([log(pd[s] / pbar[s]) for s in range(S)])
    io = np.array([log((1 - pd[s]) / (1 - pbar[s])) for s in range(S)])
    g = np.array([(pbar[s] - pd[s]) / (pbar[s] * (1 - pbar[s])) for s in range(S)])
    emean = float(np.sum(a * (1 - g * (np.array(hat) - np.array(pbar)))))
    rg = np.random.default_rng(seed)
    A, B = log(1 / alpha), log(1 / betap)
    X = np.zeros(reps)
    active = np.ones(reps, bool)
    cert = np.zeros(reps, bool)
    cost = np.zeros(reps)
    cs = np.array(c, float)
    for _ in range(maxsteps):
        s = rg.choice(S, size=reps, p=a / a.sum())
        y = rg.random(reps) < np.array(hat)[s]
        X = np.where(active, X + np.where(y, ih[s], io[s]), X)
        cost = np.where(active, cost + cs[s], cost)
        up = active & (X >= A)
        cert |= up
        active &= ~(up | (X <= -B))
        if not active.any():
            break
    return {"size_or_power": float(cert.mean()), "emean": emean, "cost": float(cost.mean()), "open": int(active.sum()), "t": R["t"]}


w5, p5, g5, c5, q05, qs5 = [0.12, 0.18, 0.25], [0.06, 0.1, 0.25], [0.02, 0.03, 0.04], [4.0, 2.0, 1.0], 0.03, 0.2
rr = [0.06, 0.0, 0.425]
rr[1] = (qs5 - q05 - w5[0] * (rr[0] + g5[0]) - w5[2] * (rr[2] + g5[2])) / w5[1] - g5[1]
live_rr = q05 + sum(w5[s] * min(1, rr[s] + g5[s]) for s in range(3))
ok("T8 boundary point lies on the boundary of A", abs(live_rr - qs5) < 1e-12, f"allowed live harm {live_rr:.12f}, r_2 {rr[1]:.6f}")
mprop = sum(w5)
prop_sh = [x / mprop for x in w5]
d = design_test(p5, w5, g5, c5, q05, qs5, rr, 0.05, 0.05, 100000, 11)
pr = design_test(p5, w5, g5, c5, q05, qs5, rr, 0.05, 0.05, 100000, 12, shares=prop_sh)
ok("T8 design shares: per-draw mean of e^Z at most one, size at most alpha at the boundary point",
   d["emean"] <= 1 + 1e-12 and d["size_or_power"] <= 0.05 and d["open"] == 0, f"mean {d['emean']:.4f}, size {d['size_or_power']:.4f}")
ok("T8 proportional shares with the same increments: per-draw mean above one and size about 0.15",
   pr["emean"] > 1 and 0.13 <= pr["size_or_power"] <= 0.17 and pr["open"] == 0, f"mean {pr['emean']:.4f}, size {pr['size_or_power']:.4f}")
pw = design_test(p5, w5, g5, c5, q05, qs5, p5, 0.05, 0.05, 100000, 13)
lower = float(kl(0.95, 0.05)) / pw["t"]
upper = (log(20) + max(log((1 - p5[s]) / (1 - root(q05, w5, p5, g5, c5, qs5)["r"][s])) for s in range(3))) / pw["t"]
ok("T8 power and cost at W_1 (alpha = beta' = 0.05): power at least 0.95, cost between the bounds",
   pw["size_or_power"] >= 0.95 and lower <= pw["cost"] <= upper, f"power {pw['size_or_power']:.4f}, cost {pw['cost']:.1f} in [{lower:.1f}, {upper:.1f}]")
ok("T7 E7 gate 2 log-aware live bound 462 calls (32.6 episodes), ratio of bounds 0.20",
   round(o2["aware"]) == 462 and round(o2["aware_ep"], 1) == 32.6 and round(o2["aware_ratio"], 2) == 0.20,
   f"{o2['aware']:.1f} calls, {o2['aware_ep']:.2f} episodes, ratio {o2['aware_ratio']:.4f}")
ok("T7 E7 gate 2 factors against live testing 5 to 12", round(1 / o2["aware_ratio"]) == 5 and round(1 / o2["ratio"]) == 12)

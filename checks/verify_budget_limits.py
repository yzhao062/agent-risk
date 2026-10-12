#!/usr/bin/env python
"""Checks for the necessity of the execution budget (Proposition F.8.1 (prop:budget-limits), Appendix F.8 (app:budget-limits)).

Paper map: Proposition F.7.1 (prop:budget; Appendix F.7 (app:budget)), Remark F.7.3 (rem:restore),
the supplemental E1 design D1 (not printed in the paper; archived in
supplement/appendix-D.md#budget-design-table; Appendix D.3 (app:demo-gates) gives the E1 counts),
Section 4.1 (sec:demo-retry), and Appendix F.8 ("Necessity of the Execution Budget";
its E1 Check and Accept Region in supplement/appendix-F2.md). Nothing under logs/ is read: every E1 input is a
count quoted in Appendix D.3 (app:demo-gates) (n = 1200, N = 479, b = 27, D = 8) or a design value of the supplemental design D1.

Run with: python verify_budget_limits.py     (about 15 seconds; Python standard library only)

Checks:
- (L1) E1 reproduction: c = 99, D1 thresholds (0, 1, 5), exact certification probability and expected executions
  of D1 given the log and cohort, live comparator 76, target-0.05 rows, the exact 97.5% upper limit 0.040332,
  the sufficiency thresholds 0.1476 / 0.1495 and the necessity threshold zeta/m = 85/479.
- (L2) Accept region: "certify iff X <= x_k" against the paper's D^U on 73,656 cases, with the never, always, and
  empty-set edges, and the c = -1 conventions.
- (L3) Size of the cutoff design: a cutoff that follows the realized N/n exceeds a_L; the cutoff with m_0
  declared in advance does not.
- (L4) E1 numbers of the E1 Check paragraph: the relative-entropy bounds, the quadratic forms, and the smallest
  fixed-sample tests that certify when the harm count falls below a cutoff (with and without randomization at it).
- (L5) Law of the power: D1 over a random log (numerical enumeration of the multinomial log, omitted mass printed),
  the fixed-sample live tests on the harm count of that power, and the agent-call ratios at the table's calls per
  execution and per live episode.
- (L6) Proof checks on small trees: the transcript identity KL(P1 || P0) = E1[K] kl(p, phat), data processing,
  and the coupling bound of part (i).
- (L7) Lower-bound checks for parts (ii) and (iii): a Lagrangian dynamic program (a dual bound on every
  sequential test on the harm indicators, of any depth) stays above the two-point bound in 7 cases, E1 included.
- (L8) Why part (iii) asks for validity under every live law: a live test that trusts the log's q0 and m is valid
  over those worlds and needs fewer live episodes than the bound, and it is invalid in a world outside them.

Exact rational arithmetic (fractions.Fraction) covers L1, L2, the tests and counts of L4, the thresholds of L5, and L8.
Floating point is used for: the logarithms of kl (L4, L5, L7, L8), the binomial probabilities computed with lgamma (L3
and the weights of L5), and the random trees of L6 (seeded, so the run is reproducible) and the dynamic program of L7."""
from fractions import Fraction as F
from math import comb, exp, lgamma, log, log1p, sqrt
import random
import time

# ---------------------------------------------------------------------------
# E1 inputs (counts quoted in the paper) and the design D1
# ---------------------------------------------------------------------------
n, N, b, Dh = 1200, 479, 27, 8
QS, AL = F(1, 10), F(1, 40)          # target q*, log error a_L
LOOKS, INV_E = (34, 50, 80), 120     # D1: looks and error 1/120 per look
Q0, M, P = F(b, n), F(N, n), F(Dh, N)
Q = Q0 + M * P                       # identified harm 35/1200
QL = F(36, n)                        # live rate of the table's live rows
ZETA = QS - Q                        # headroom
PBAR0 = P + ZETA / M                 # boundary rate at gamma = 0
CALLS_D1, EXEC_D1, CALLS_LIVE, N_LIVE = 387, F(466, 10), 854, 76   # Table tab:budget


def say(tag, msg):
    print(f"PASS {tag}: {msg}")


# ---------------------------------------------------------------------------
# Exact binomial and hypergeometric primitives
# ---------------------------------------------------------------------------
def binom_cdf(nn, t, z):
    """Exact P(Bin(nn, t) <= z) for a rational t."""
    if z < 0:
        return F(0)
    if z >= nn:
        return F(1)
    tn, td = t.numerator, t.denominator
    return F(sum(comb(nn, k) * tn**k * (td - tn)**(nn - k) for k in range(z + 1)), td**nn)


def cp_cutoff(nn, t, a):
    """Largest integer z in -1..nn with P(Bin(nn, t) <= z) <= a (-1 if none, and for t <= 0). Exact."""
    if t <= 0:
        return -1
    assert t < 1
    tn, td = t.numerator, t.denominator
    den, cum, z = td**nn, 0, -1
    while z + 1 <= nn:
        term = comb(nn, z + 1) * tn**(z + 1) * (td - tn)**(nn - z - 1)
        if (cum + term) * a.denominator <= a.numerator * den:
            cum, z = cum + term, z + 1
        else:
            break
    return z


def live_n(power, rate, target=QS, a=F(1, 20)):
    """Smallest n_L whose one-sided 95% CP test of `target` certifies with probability >= power at `rate`."""
    for nl in range(1, 3000):
        z = cp_cutoff(nl, target, a)
        pw = binom_cdf(nl, rate, z) if z >= 0 else F(0)
        if pw >= power:
            return nl, pw
    raise AssertionError("no n_L below 3000")


def hyp_le(Nn, D, k, x):
    """Exact P(X <= x) for X ~ Hypergeometric(Nn, D, k)."""
    if x < 0:
        return F(0)
    lo = max(0, k - (Nn - D))
    return F(sum(comb(D, j) * comb(Nn - D, k - j) for j in range(lo, min(x, k, D) + 1)), comb(Nn, k))


def du_paper(Nn, k, x, e):
    """The paper's D^U(x, k; e) = max{D in x..Nn-k+x : P_H(Nn,D,k)(X <= x) > e}; Nn if k = 0."""
    if k == 0:
        return Nn
    return max(D for D in range(x, Nn - k + x + 1) if hyp_le(Nn, D, k, x) > e)


def x_k(Nn, d0, k, e):
    """Accept threshold of one look (d0 = c - b): certify iff X <= x_k; -1 means never, k means always."""
    if d0 < 0:
        return -1
    if d0 >= Nn:
        return k
    best = -1
    for x in range(k + 1):
        if hyp_le(Nn, d0 + 1, k, x) <= e:
            best = x
        else:
            break
    return best


def x_k_fast(Nn, d0, k, inv_e):
    """Same as x_k for e = 1/inv_e, in integer arithmetic with a running sum."""
    if d0 < 0:
        return -1
    if d0 >= Nn:
        return k
    D = d0 + 1
    den, num, best = comb(Nn, k), 0, -1
    for x in range(k + 1):
        if max(0, k - (Nn - D)) <= x <= D:
            num += comb(D, x) * comb(Nn - D, k - x)
        if num * inv_e <= den:
            best = x
        else:
            break
    return best


def nested_law(Nn, Dd, looks, ts):
    """Exact law of D1's nested looks given D harmful among Nn: (P(certify at look j) for each j, P(never))."""
    k1, k2, k3 = looks
    pc, pn, den = [F(0)] * 3, F(0), comb(Nn, Dd)
    for x1 in range(min(k1, Dd) + 1):
        for y2 in range(min(k2 - k1, Dd - x1) + 1):
            for y3 in range(min(k3 - k2, Dd - x1 - y2) + 1):
                x2, x3 = x1 + y2, x1 + y2 + y3
                if Dd - x3 > Nn - k3:
                    continue
                pr = F(comb(k1, x1) * comb(k2 - k1, y2) * comb(k3 - k2, y3) * comb(Nn - k3, Dd - x3), den)
                if x1 <= ts[0]:
                    pc[0] += pr
                elif x2 <= ts[1]:
                    pc[1] += pr
                elif x3 <= ts[2]:
                    pc[2] += pr
                else:
                    pn += pr
    return pc, pn


# ---------------------------------------------------------------------------
# Floating-point primitives
# ---------------------------------------------------------------------------
def kl(a, c):
    """Binary relative entropy kl(a, c) in nats (floats)."""
    a, c = float(a), float(c)
    if a == c:
        return 0.0
    out = 0.0
    if a > 0:
        out += a * log(a / c)
    if a < 1:
        out += (1 - a) * log((1 - a) / (1 - c))
    return out


def fpmf(k, nn, p):
    if k < 0 or k > nn:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == nn else 0.0
    return exp(lgamma(nn + 1) - lgamma(k + 1) - lgamma(nn - k + 1) + k * log(p) + (nn - k) * log1p(-p))


def fcdf(z, nn, p):
    return sum(fpmf(k, nn, p) for k in range(0, min(z, nn) + 1)) if z >= 0 else 0.0


def fcp_cutoff(nn, t, a):
    """Floating-point version of cp_cutoff."""
    if t <= 0:
        return -1
    z, cum = -1, 0.0
    while z + 1 <= nn:
        nxt = cum + fpmf(z + 1, nn, t)
        if nxt <= a:
            z, cum = z + 1, nxt
        else:
            break
    return z


# ---------------------------------------------------------------------------
# L1: E1 reproduction
# ---------------------------------------------------------------------------
def check_l1():
    t0 = time.time()
    c = cp_cutoff(n, QS, AL)
    assert c == 99, c
    ts = [x_k_fast(N, c - b, k, INV_E) for k in LOOKS]
    assert ts == [0, 1, 5], ts
    assert ts == [x_k(N, c - b, k, F(1, INV_E)) for k in LOOKS]
    pc, pn = nested_law(N, Dh, LOOKS, ts)
    pcert = sum(pc)
    ek = pc[0] * LOOKS[0] + pc[1] * LOOKS[1] + (pc[2] + pn) * LOOKS[2]
    assert pcert == F(3810952577729839, 3812364243804765), pcert
    assert ek == F(1077647242874066, 23105237841241), ek
    assert f"{float(pcert):.9f}" == "0.999629714" and f"{float(ek):.7f}" == "46.6408202"
    say("L1a", f"c = {c}, n*q* = {int(n * QS)}; D1 thresholds {ts}; P(certify | log, cohort) = {pcert} "
               f"({float(pcert):.9f}); E[K | log, cohort] = {float(ek):.7f}")
    nl, pw = live_n(F(4, 5), QL)
    assert nl == 76 and f"{float(pw):.9f}" == "0.805651725"
    say("L1b", f"live comparator at rate 36/1200: n_L = {nl}, power {float(pw):.9f} (Table tab:budget: 76 and 0.806)")
    # target 0.05 rows of Table tab:budget
    c05 = cp_cutoff(n, F(1, 20), AL)
    xk = x_k_fast(N, c05 - b, 244, 40)
    pw244 = hyp_le(N, Dh, 244, xk)
    nl05, pw05 = live_n(F(4, 5), QL, target=F(1, 20))
    assert c05 == 45 and xk == 5 and f"{float(pw244):.3f}" == "0.845" and nl05 == 601 and f"{float(pw05):.3f}" == "0.800"
    say("L1c", f"target 0.05: c = {c05}; fixed 244 executions has power {float(pw244):.4f}; live test n_L = {nl05}, power "
               f"{float(pw05):.4f} (Table tab:budget: 0.845, 601, 0.800)")
    # exact 97.5% upper limit of 35/1200: u in (0.040331, 0.040332] by the duality u <= t iff P(Bin(n,t) <= 35) <= a_L
    lo_t, hi_t = F("0.040331"), F("0.040332")
    assert binom_cdf(n, lo_t, 35) > AL >= binom_cdf(n, hi_t, 35)
    head_lo, head_hi = QS - hi_t, QS - lo_t
    g_plug = (float(head_lo / F(485, n)), float(head_hi / F(485, n)))
    g_unres = (float(head_lo / M), float(head_hi / M))
    assert f"{g_plug[0]:.4f}" == f"{g_plug[1]:.4f}" == "0.1476" and f"{g_plug[0]:.3f}" == "0.148"
    assert f"{g_unres[0]:.4f}" == f"{g_unres[1]:.4f}" == "0.1495"
    nec = ZETA / M
    assert nec == F(85, 479) and f"{float(nec):.3f}" == "0.177"
    gap, margin, mass = float(nec) - g_plug[0], float(nec) - g_unres[0], g_unres[0] - g_plug[0]
    assert f"{gap:.3f}" == "0.030" and f"{margin:.3f}" == "0.028" and f"{mass:.3f}" == "0.002"
    say("L1d", f"u_CP(35, 1200; 0.025) in (0.040331, 0.040332]; sufficiency threshold gamma <= {g_plug[0]:.4f} "
               f"(continuing mass 485/1200; 0.148) or {g_unres[0]:.4f} (unresolved mass 479/1200); necessity threshold "
               f"zeta/m = 85/479 = {float(nec):.4f} (0.177); the gap {gap:.3f} splits into {margin:.3f} (sampling margin of "
               f"the log) and {mass:.3f} (continuing mass)")
    # world W_0^eps is legal at E1: p_0 < 1 iff q* < q_0 + m
    assert QS < Q0 + M and 0 < P < PBAR0 < 1
    say("L1e", f"m = {float(M):.3f}, zeta = {float(ZETA):.4f}, pbar(gamma=0) = {float(PBAR0):.3f}, q* < q_0 + m = "
               f"{float(Q0 + M):.3f}, so p_0 = pbar < 1 (E1 world W_0^eps is legal) ({time.time() - t0:.1f}s)")
    return c, pcert, ek


# ---------------------------------------------------------------------------
# L2: accept region
# ---------------------------------------------------------------------------
def check_l2():
    t0 = time.time()
    bad = tot = 0
    edge = {"never": 0, "always": 0, "empty": 0}
    for Nn in range(1, 13):
        for k in range(Nn + 1):
            for e in (F(1, 40), F(1, 120), F(1, 5)):
                for bb in range(4):
                    for c in range(-1, bb + Nn + 2):
                        d0 = c - bb
                        xk = x_k(Nn, d0, k, e)
                        if d0 < 0:
                            edge["never"] += 1
                        elif d0 >= Nn:
                            edge["always"] += 1
                        elif xk == -1:
                            edge["empty"] += 1
                        for x in range(k + 1):
                            tot += 1
                            bad += (bb + du_paper(Nn, k, x, e) <= c) != (x <= xk)
    assert bad == 0 and tot == 73656, (bad, tot)
    say("L2a", f"certify iff X <= x_k (x_k = -1 never, c - b >= N always): {tot} cases (N <= 12, e in 1/40, 1/120, 1/5), "
               f"0 mismatches; edge configurations hit: {edge}")
    # the integer-arithmetic threshold equals the Fraction threshold
    for Nn in (30, 100, 479):
        for d0 in (-1, 0, 5, 70, 72, 200, 600):
            for k in (1, 34, 80):
                if k <= Nn:
                    assert x_k(Nn, d0, k, F(1, 120)) == x_k_fast(Nn, d0, k, 120)
    say("L2b", "x_k_fast equals x_k on 63 further (N, c - b, k) triples")
    # Clopper-Pearson duality: u_CP(z, nn; a) <= t iff P(Bin(nn, t) <= z) <= a, with u_CP the root of t -> P(Bin <= z) = a
    cases = 0
    for nn in (5, 12, 40):
        for z in range(nn):
            for a in (F(1, 40), F(1, 20)):
                lo, hi = F(0), F(1)
                for _ in range(40):                                  # bisection on exact probabilities
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if binom_cdf(nn, mid, z) > a else (lo, mid)
                for t in (F(1, 50), F(1, 20), F(1, 10), F(3, 10), F(3, 5)):
                    if abs(t - hi) > F(1, 10**9):
                        cases += 1
                        assert (hi <= t) == (binom_cdf(nn, t, z) <= a)
    say("L2c", f"Clopper-Pearson duality on {cases} (n, z, a, t) cases, 0 mismatches")
    assert cp_cutoff(1200, F(0), AL) == -1 and cp_cutoff(1200, F(-1, 10), AL) == -1
    assert cp_cutoff(20, F(1, 20), AL) == -1              # P(Bin(20, 0.05) <= 0) = 0.358 > a_L
    assert cp_cutoff(1200, F(1, 10) - F(1, 10), AL) == -1  # q* <= xi_0
    say("L2d", f"c = -1 when q* <= xi_0 and when even z = 0 fails ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# L3: size of the cutoff design
# ---------------------------------------------------------------------------
def check_l3():
    t0 = time.time()
    gam, mm, q0f, qsf, alf = 0.1476, N / n, b / n, 0.1, 0.025
    nlo, nhi = int(mm * n - 8 * sqrt(n * mm * (1 - mm))), int(mm * n + 8 * sqrt(n * mm * (1 - mm)))
    c0 = cp_cutoff(n, QS - M * F(1476, 10000), AL)            # cutoff with m_0 = m declared in advance (exact)
    rows = []
    cache = {}
    for Nn in range(nlo, nhi + 1):
        cN = fcp_cutoff(n, qsf - (Nn / n) * gam, alf)         # cutoff that follows the realized N/n
        cache[Nn] = (fpmf(Nn, n, mm), cN, [fpmf(j, n - Nn, q0f / (1 - mm)) for j in range(cN + 1)])
    for delta in (1e-4, 1e-6, 1e-9):
        qhat = qsf - mm * gam + delta                          # restored cohort rate; live rate q* + delta
        phat = (qhat - q0f) / mm
        assert 0 < phat < 1 and phat + gam < 1
        size_real = size_indep = 0.0
        for Nn in range(nlo, nhi + 1):
            w, cN, pb = cache[Nn]
            pd_cum, acc = [], 0.0
            for j in range(cN + 1):
                acc += fpmf(j, Nn, phat)
                pd_cum.append(acc)
            size_real += w * sum(pb[j] * pd_cum[cN - j] for j in range(cN + 1))   # b + Dhat <= c_N, multinomial log
            size_indep += w * fcdf(cN, n, qhat)                                    # S ~ Bin(n, qhat) independent of N
        size_fixed = fcdf(c0, n, qhat)
        rows.append((delta, size_real, size_indep, size_fixed))
        assert size_real > alf and size_indep > alf and size_fixed <= alf
    r = {d: (a, bb, c) for d, a, bb, c in rows}
    assert f"{r[1e-4][0]:.5f}" == "0.02677" and f"{r[1e-4][1]:.5f}" == "0.02596" and f"{r[1e-4][2]:.5f}" == "0.01777"
    assert r[1e-6][0] >= 0.027 and f"{r[1e-9][0]:.3f}" == "0.028" and f"{r[1e-9][0]:.4f}" == "0.0277"
    for d, a, bb, c in rows:
        say("L3", f"k = N, gamma = 0.1476, live rate q* + {d:g}: cutoff following realized N/n has size {a:.5f} "
                  f"(S independent of N: {bb:.5f}) against a_L = 0.025; cutoff with m_0 = m declared in advance "
                  f"(c = {c0}): {c:.5f}")
    say("L3", f"so the size is at least 0.027 (0.0277 as the live rate falls to q*) with the realized N/n, and at most a_L "
              f"with m_0 declared ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# L4: E1 numbers of the E1 Check paragraph
# ---------------------------------------------------------------------------
def np_min_n(pa, pb, alpha, beta, randomized, nmax=400):
    """Smallest n of a test that certifies when the count is small: size <= alpha under Bern(pa), power >= 1 - beta
    under Bern(pb). randomized=True is the Neyman-Pearson test (size exactly alpha). Exact arithmetic."""
    for nn in range(1, nmax):
        z = -1
        while binom_cdf(nn, pa, z + 1) <= alpha:
            z += 1
        p0 = binom_cdf(nn, pa, z)
        pw = binom_cdf(nn, pb, z)
        if randomized:
            r = (alpha - p0) / (binom_cdf(nn, pa, z + 1) - p0)
            pw += r * (binom_cdf(nn, pb, z + 1) - pw)
        if pw >= 1 - beta:
            return nn
    raise AssertionError("no n below nmax")


def check_l4(pcert):
    t0 = time.time()
    alpha, beta = F(1, 20), F(1, 5)
    klab = kl(1 - beta, alpha)
    assert f"{klab:.4f}" == "1.9064"
    ex_kl = klab / kl(P, PBAR0)
    lv_kl = klab / kl(Q, QS)
    ex_chi = klab * float(M**2 * PBAR0 * (1 - PBAR0) / ZETA**2)
    lv_chi = klab * float(QS * (1 - QS) / ZETA**2)
    assert f"{ex_kl:.1f}" == "12.3" and f"{lv_kl:.1f}" == "50.7" and ex_chi < ex_kl and lv_chi < lv_kl
    ratio = float(M**2 * PBAR0 * (1 - PBAR0) / (QS * (1 - QS)))
    ratio_p = float(M**2 * P * (1 - P) / (Q * (1 - Q)))
    assert f"{ratio:.2f}" == "0.28" and f"{ratio_p:.3f}" == "0.092"
    say("L4a", f"kl(0.8, 0.05) = {klab:.4f}; bounds {ex_kl:.2f} executions and {lv_kl:.2f} live episodes (chi-square forms "
               f"{ex_chi:.2f} and {lv_chi:.2f}); quadratic-form ratio m^2 pbar(1-pbar)/(q*(1-q*)) = {ratio:.3f} "
               f"(0.28), against {ratio_p:.3f} with p(1-p)/(q(1-q)) in its place")
    ex_np, ex_rnd = np_min_n(PBAR0, P, alpha, beta, False), np_min_n(PBAR0, P, alpha, beta, True)
    lv_np, lv_rnd = np_min_n(QS, Q, alpha, beta, False), np_min_n(QS, Q, alpha, beta, True)
    assert (ex_np, lv_np, ex_rnd, lv_rnd) == (23, 76, 17, 74), (ex_np, lv_np, ex_rnd, lv_rnd)
    lv_cp, _ = live_n(F(4, 5), Q)
    assert lv_cp == lv_np
    say("L4b", f"smallest fixed-sample lower-tail count tests: non-randomized {ex_np} executions and {lv_np} live episodes (the CP test at rate "
               f"35/1200 also needs {lv_cp}); randomized {ex_rnd} and {lv_rnd}; count ratios bounds {ex_kl / lv_kl:.3f}, "
               f"non-randomized {ex_np / lv_np:.3f}, randomized {ex_rnd / lv_rnd:.3f}")
    # single look on the realized cohort (not quoted in the text)
    for inv_e in (40, 120):
        pws = [float(hyp_le(N, Dh, k, x_k_fast(N, 99 - b, k, inv_e))) if x_k_fast(N, 99 - b, k, inv_e) >= 0 else 0.0
               for k in range(0, N + 1)]
        mk = lambda thr: next(k for k in range(1, N + 1) if pws[k] >= thr)
        say("L4c", f"single look, e = 1/{inv_e}: power at k = 34 is {pws[34]:.3f}; smallest k for power 0.80: {mk(0.80)}, "
                   f"0.90: {mk(0.90)}")
    # conditional power of D1 and the live test of it (not quoted in the text)
    n36, _ = live_n(pcert, QL)
    n35, _ = live_n(pcert, Q)
    assert (n36, n35) == (275, 263)
    say("L4d", f"live test with D1's power given the log ({float(pcert):.5f}): {n36} episodes at rate 0.030, {n35} at "
               f"0.0292; table ratios counts 46.6/76 = {46.6 / 76:.3f}, calls 387/854 = {387 / 854:.3f} (0.45), tokens "
               f"2.56/5.30 = {2.56 / 5.30:.3f} ({time.time() - t0:.1f}s)")
    return klab


# ---------------------------------------------------------------------------
# L5: law of the power, D1 over a random log
# ---------------------------------------------------------------------------
class Seq:
    """Certification law of D1 along a sequence of iid Bernoulli(p) outcomes, for thresholds (t1, t2, t3)."""

    def __init__(self, p):
        k1, k2, k3 = LOOKS
        self.k = LOOKS
        self.px1 = [fpmf(j, k1, p) for j in range(k1 + 1)]
        self.py2 = [fpmf(j, k2 - k1, p) for j in range(k2 - k1 + 1)]
        self.py3 = [fpmf(j, k3 - k2, p) for j in range(k3 - k2 + 1)]
        self.c2, self.c3, acc2, acc3 = [], [], 0.0, 0.0
        for v in self.py2:
            acc2 += v
            self.c2.append(acc2)
        for v in self.py3:
            acc3 += v
            self.c3.append(acc3)
        self.cache = {}

    @staticmethod
    def _cdf(cum, z):
        return 0.0 if z < 0 else cum[min(z, len(cum) - 1)]

    def stats(self, ts):
        if ts in self.cache:
            return self.cache[ts]
        t1, t2, t3 = ts
        k1, k2, k3 = self.k
        cert1 = sum(self.px1[x] for x in range(0, min(t1, k1) + 1)) if t1 >= 0 else 0.0
        cert2 = cert3 = 0.0
        for x1 in range(max(t1 + 1, 0), k1 + 1):
            cert2 += self.px1[x1] * self._cdf(self.c2, t2 - x1)
            for y2 in range(k2 - k1 + 1):
                if x1 + y2 > t2:
                    cert3 += self.px1[x1] * self.py2[y2] * self._cdf(self.c3, t3 - x1 - y2)
        reach2 = 1 - cert1
        reach3 = reach2 - cert2
        out = (cert1 + cert2 + cert3, k1 + (k2 - k1) * reach2 + (k3 - k2) * reach3)
        self.cache[ts] = out
        return out


def check_l5(c, pcert):
    t0 = time.time()
    p, mf, q0f = float(P), float(M), float(Q0)
    seq = Seq(p)
    ts_of = {}

    def thr(Nn, bb):
        key = (Nn, c - bb)
        if key not in ts_of:
            ts_of[key] = tuple(x_k_fast(Nn, c - bb, k, INV_E) for k in LOOKS)
        return ts_of[key]

    # consistency of the iid-sequence law with the exact hypergeometric nested law (exchangeability of the order)
    ts = thr(N, b)
    pc_mix = ek_mix = mass = 0.0
    for Dd in range(0, 41):
        w = fpmf(Dd, N, p)
        pcs, pn = nested_law(N, Dd, LOOKS, ts)
        pc_mix += w * float(sum(pcs))
        ek_mix += w * float(pcs[0] * LOOKS[0] + pcs[1] * LOOKS[1] + (pcs[2] + pn) * LOOKS[2])
        mass += w
    pc_seq, ek_seq = seq.stats(ts)
    assert abs(pc_mix / mass - pc_seq) < 1e-9 and abs(ek_mix / mass - ek_seq) < 1e-7
    say("L5a", f"at (b, N) = (27, 479) the iid-sequence law gives P(certify) {pc_seq:.9f}, E[K] {ek_seq:.6f}; the mixture of exact "
               f"hypergeometric nested laws over D ~ Bin(479, 8/479) gives {pc_mix / mass:.9f}, {ek_mix / mass:.6f}")
    # enumerate the multinomial log (b, N, rest); weights by lgamma, +-8 sigma windows
    sdN = sqrt(n * mf * (1 - mf))
    nlo, nhi = max(0, int(mf * n - 8 * sdN)), int(mf * n + 8 * sdN)
    pw_tot = ek_tot = wsum = 0.0
    for Nn in range(nlo, nhi + 1):
        wN = fpmf(Nn, n, mf)
        rate = q0f / (1 - mf)
        sdb = sqrt((n - Nn) * rate * (1 - rate))
        for bb in range(0, int(rate * (n - Nn) + 8 * sdb) + 1):
            w = wN * fpmf(bb, n - Nn, rate)
            pcv, ekv = seq.stats(thr(Nn, bb))
            pw_tot += w * pcv
            ek_tot += w * ekv
            wsum += w
    pw_tot, ek_tot = pw_tot / wsum, ek_tot / wsum
    assert 1 - wsum < 1e-9
    assert f"{pw_tot:.3f}" == "0.993" and f"{ek_tot:.1f}" == "45.1"
    say("L5b", f"D1 over a random log in W_1 (numerical enumeration, omitted mass {1 - wsum:.1e}): P(certify) = {pw_tot:.5f}, "
               f"E[K] = {ek_tot:.3f} (given the log and cohort: {float(pcert):.5f} and 46.64)")
    # the Wald bound at D1's own (size, power) is below its cost
    bound = kl(pw_tot, 0.05) / kl(P, PBAR0)
    assert bound < ek_tot
    say("L5c", f"Wald bound (ii) at D1's own unconditional (size, power) = (0.05, {pw_tot:.4f}): {bound:.1f} <= {ek_tot:.1f}")
    # live tests of that power, and call ratios at the table's calls per execution and per live episode
    n36, _ = live_n(pw_tot, QL)
    n35, _ = live_n(pw_tot, Q)
    assert (n36, n35) == (191, 179), (n36, n35)
    cpe, cpl = float(F(CALLS_D1) / EXEC_D1), CALLS_LIVE / N_LIVE
    r36, r35 = ek_tot * cpe / (n36 * cpl), ek_tot * cpe / (n35 * cpl)
    assert f"{r36:.2f}" == "0.17" and f"{r35:.2f}" == "0.19" and f"{CALLS_D1 / CALLS_LIVE:.2f}" == "0.45"
    say("L5d", f"live test of that power: {n36} episodes at the live rate 0.030, {n35} at 0.0292; calls per execution "
               f"{cpe:.2f}, per live episode {cpl:.2f}; agent-call ratio {r36:.4f} (0.17) and {r35:.4f} (0.19) against "
               f"{CALLS_D1 / CALLS_LIVE:.3f} (0.45) in the table; count ratios {ek_tot / n36:.3f} and {ek_tot / n35:.3f} "
               f"({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# L6: proof checks on small trees
# ---------------------------------------------------------------------------
def check_l6(seed=20261007, trials=300, T=7):
    t0 = time.time()
    rng = random.Random(seed)
    worst_id = worst_dp = worst_tv = 0.0
    for _ in range(trials):
        p = rng.uniform(0.02, 0.6)
        ph = min(0.98, p + rng.uniform(0.0, 0.35))
        EK = KLs = P1E = P0E = 0.0
        stack = [((), 1.0, 1.0, 1.0)]
        while stack:
            h, reach, pi1, pi0 = stack.pop()
            t = len(h)
            if t == T:
                s = 1.0
            else:
                u = rng.random()
                s = 0.0 if u < 0.2 else (1.0 if u < 0.3 else rng.random())
            cc = rng.choice((0.0, 1.0, rng.random()))
            mass1 = reach * s * pi1
            EK += mass1 * t
            P1E += mass1 * cc
            P0E += reach * s * cc * pi0
            if mass1 > 0:
                KLs += mass1 * log(pi1 / pi0)
            if s < 1.0:
                for y in (0, 1):
                    stack.append((h + (y,), reach * (1 - s), pi1 * (p if y else 1 - p), pi0 * (ph if y else 1 - ph)))
        worst_id = max(worst_id, abs(KLs - EK * kl(p, ph)))
        worst_dp = max(worst_dp, kl(P1E, P0E) - KLs if 0 < P0E < 1 or P1E == P0E else 0.0)
        worst_tv = max(worst_tv, abs(P1E - P0E) - T * (ph - p))
    assert worst_id < 1e-12 and worst_dp < 1e-12 and worst_tv < 1e-12, (worst_id, worst_dp, worst_tv)
    say("L6", f"{trials} random randomized procedures on depth-{T} trees: KL(P1 || P0) = E1[K] kl(p, phat) (max error "
              f"{worst_id:.1e}); kl(P1(E), P0(E)) <= KL (data processing, max excess {worst_dp:.1e}); "
              f"|P1(E) - P0(E)| <= T (phat - p) (coupling bound of part (i), max excess {worst_tv:.1e}) ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# L7: dual (Lagrangian) lower bound on every sequential test
# ---------------------------------------------------------------------------
def dual_value(p, pb, alpha, beta, l0, l1, T):
    """Lagrangian bound on E_1[K] over every sequential test (any depth) with P_0(certify) <= alpha and
    P_1(certify) >= 1 - beta, i.i.d. Bernoulli(p) under P_1 and Bernoulli(pb) under P_0. Beyond depth T the value
    is bounded below by -l1, so the result is a valid lower bound for every depth. Weak duality, for any l0, l1 >= 0."""
    r1, r0 = pb / p, (1 - pb) / (1 - p)
    V = [-l1] * (T + 1)
    for t in range(T - 1, -1, -1):
        V = [min(min(0.0, l0 * r1**x * r0**(t - x) - l1), 1 + p * V[x + 1] + (1 - p) * V[x]) for x in range(t + 1)]
    return V[0] + l1 * (1 - beta) - l0 * alpha


def ternary(f, lo, hi, it):
    for _ in range(it):
        a, c = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if f(a) < f(c):
            lo = a
        else:
            hi = c
    mid = (lo + hi) / 2
    return f(mid), mid


def check_l7():
    t0 = time.time()
    alpha, beta = 0.05, 0.20
    cases = [("part (ii)", 0.10, 0.50, 40, 25, 60, 200), ("part (ii)", 0.05, 0.40, 40, 25, 60, 200),
             ("part (ii)", 0.15, 0.50, 40, 25, 60, 200), ("part (ii)", 0.10, 0.40, 40, 25, 60, 200),
             ("part (ii)", 0.05, 0.30, 60, 25, 60, 200), ("part (ii), E1 executions", float(P), float(PBAR0), 60, 25, 60, 200),
             ("part (iii), E1 live", float(Q), 0.10, 220, 14, 140, 400)]
    for label, p, pb, T, it, hi1, hi0 in cases:
        def inner(l1):
            return ternary(lambda l0: dual_value(p, pb, alpha, beta, l0, l1, T), 0.0, hi0, it)[0]
        val, _ = ternary(inner, 0.0, hi1, it)
        bnd = kl(1 - beta, alpha) / kl(p, pb)
        chi = kl(1 - beta, alpha) * pb * (1 - pb) / (pb - p) ** 2
        assert val >= bnd - 1e-9 and bnd >= chi - 1e-9, (label, p, pb, val, bnd, chi)
        say("L7", f"{label}: p = {p:.4f}, boundary rate {pb:.4f}, alpha = 0.05, beta = 0.2: dual bound on E_1[K] over all "
                  f"sequential tests {val:.3f} >= kl bound {bnd:.3f} >= chi-square form {chi:.3f}")
    say("L7", f"all 7 cases consistent ({time.time() - t0:.1f}s)")


def check_l8():
    """Part (iii) needs validity under every live law with harm above q*. A live test that trusts the log's q0 and m
    watches only the unresolved episodes: wait for 18 of them, and certify if none is harmful or the only harm is
    among the first five. Over worlds with the log's q0 and m it is valid and powerful with fewer live episodes than
    the bound of part (iii); a world with larger q0 and harmless continuations shows it is not valid in general."""
    def acc(x):
        return (1 - x) ** 18 + 5 * x * (1 - x) ** 17
    xs = [PBAR0 + (1 - PBAR0) * F(i, 400) for i in range(401)]
    assert all(acc(xs[i]) >= acc(xs[i + 1]) for i in range(400))   # downward closed: decreasing in x
    size, power, ek = acc(PBAR0), acc(P), 18 / M
    bound = kl(F(4, 5), F(1, 20)) / kl(float(Q), float(QS))
    assert size < F(1, 20) and power > F(4, 5) and float(ek) < bound, (float(size), float(power), float(ek), bound)
    # a world outside the log's q0 and m: harm above q* with harmless continuations, certified with probability 1
    q0w = QS + F(1, 100)
    assert q0w + M <= 1 and q0w + M * 0 > QS and acc(F(0)) == 1
    say("L8", f"a live test that trusts q0 and m (18 unresolved episodes; certify if no harm, or one harm among the "
              f"first five): size {float(size):.4f} at boundary rate 93/479, power {float(power):.4f}, E[K_L] = "
              f"{float(ek):.2f} < part (iii) bound {bound:.2f}; it certifies with probability 1 in a world with "
              f"harm q* + 0.01 from q0 alone, so it is valid only over the log's q0 and m")


def main():
    t_start = time.time()
    c, pcert, ek = check_l1()
    check_l2()
    check_l3()
    check_l4(pcert)
    check_l5(c, pcert)
    check_l6()
    check_l7()
    check_l8()
    print(f"All execution-budget necessity checks PASSED in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

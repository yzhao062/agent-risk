#!/usr/bin/env python
"""Checks for the achievability of the execution budget (matching Proposition prop:budget-limits).

Paper map: Appendix F.8 (app:budget-limits), Proposition F.8.1 (prop:budget-limits; necessity), Proposition
F.8.2 (prop:budget-achievability), Corollary F.8.3 (cor:cost-ratio), Remark F.8.4 (rem:estimate-log), and the E1 Achievability Check
(supplement/appendix-F2.md). Nothing under logs/ is read: every E1 input is a count quoted in Appendix D.3 (app:demo-gates)
(n = 1200, N = 479, b = 27, D = 8) with q* = 0.10.

Run with: python verify_budget_sufficiency.py     (about one minute; numpy for the grid and the log averages, standard
library otherwise)

Setting. p = 8/479 (alternative), pbar = 93/479 (boundary at gamma = 0), q = 35/1200, q* = 1/10; the SPRT of Bernoulli(p)
against Bernoulli(pbar) has log-likelihood-ratio increments log(p/pbar) after a harm and up = log((1-p)/(1-pbar)) after
a harmless outcome. It certifies at Xi >= A and stops at Xi <= -B (Xi is the log-likelihood ratio).

Checks:
- (S1) Exact laws of the E1 SPRT by forward dynamic programming on the (trials, harms) lattice, in rational arithmetic
  (integer weights over a common denominator): Ville thresholds (A, B) = (log 20, log 5) for the execution test (pool of
  N = 479) and the live test (unlimited pool, truncated at 2500 with the remaining mass reported); the width of each
  sandwich; the guarantee 1 - beta - chi_n of the proposition for these thresholds.
- (S1d) The assembly of Proposition (ii): the log-averaged size and power of the test with beta' = beta - chi_n at
  n = 1200, 600, and 530 (power at least 1 - beta, size at most alpha).
- (S2) Best boundary pair on a grid meeting exact size <= 0.05 (at the boundary rate, the worst case by monotonicity) and
  exact power >= 0.80, for both tests; the winner is recomputed in rational arithmetic, and its log-averaged size and
  power are compared with the given-log values (execution winner); any B <= 0.01 gives the same winning test.
- (S3) Proof checks on random designs: E[e^Z] = 1 - (theta - c)(c - a)/(c(1-c)), Ville size and error bounds, monotone
  certification probability in the harm rate (for arbitrary thresholds), Wald's identity, the overshoot bound, the lower
  bound at the design's own operating point, and the truncation bound P(tau > N) <= E[tau]/N.
- (S4) The corollary: the bracket kl(a,c) in [(c-a)^2/(2 v), (c-a)^2/(c(1-c))], the ratio within a factor 2 of the ratio
  of quadratic forms when pbar and q* are at most 1/2, the ratio of upper to lower bounds tending to 1 as alpha,
  beta -> 0, the gap inequality, and the exact SPRT cost over kl(1-beta, alpha).
- (S5) The finite-log remark: the boundary decreases in q0 and m; at E1 the cost of the Ville test with upper confidence
  limits at a matched total error 0.05 (alpha = 0.04, alpha_log = 0.01); the conditional guarantee
  P(no certification | log) <= beta' + K-bar^U / N; the exact probability of {pbar^U <= p} and the expected
  loss min(1, K-bar^U / N) over the random log at n = 100, 200, 400, 1200.
- (S6) The finite pool: the most powerful size-0.05 test with all 479 executions (Neyman-Pearson, randomized) reaches
  power 0.80 at gamma = 0.159, so no procedure with this pool certifies beyond it; chi_n < beta fails at gamma = 0.15."""
from fractions import Fraction as F
from math import ceil, comb, exp, floor, lgamma, log, log1p, sqrt
import random
import time

import numpy as np

# ---------------------------------------------------------------------------
# E1 inputs (counts quoted in the paper) and derived rates
# ---------------------------------------------------------------------------
n, N, b, Dh = 1200, 479, 27, 8
QS, GAMMA = F(1, 10), F(0)
ALPHA, BETA = F(1, 20), F(1, 5)
Q0, M, P = F(b, n), F(N, n), F(Dh, N)
Q = Q0 + M * P                                   # 35/1200
ZETA = QS - Q                                    # 85/1200
PBAR = P + ZETA / M - GAMMA                      # 93/479
TOL = 1e-9                                       # margin below which a lattice point would count as a tie


def say(tag, msg):
    print(f"PASS {tag}: {msg}")


def kl(a, c):
    """Binary relative entropy kl(a, c) in nats (floats)."""
    a, c = float(a), float(c)
    if a == c:
        return 0.0
    out = a * log(a / c) if a > 0 else 0.0
    if a < 1:
        out += (1 - a) * log((1 - a) / (1 - c))
    return out


# ---------------------------------------------------------------------------
# A lattice SPRT: Lambda_t = (t - s) u - s d after t trials and s harms (a < c)
# ---------------------------------------------------------------------------
class Design:
    def __init__(self, a, c, A, B):
        a, c = float(a), float(c)
        assert 0 < a < c < 1 and A > 0 and B > 0
        self.a, self.c, self.A, self.B = a, c, float(A), float(B)
        self.u = log((1 - a) / (1 - c))          # harmless outcome: largest upward step
        self.d = log(c / a)                      # harm: downward step of size d
        self.kl = kl(a, c)
        self.ties = 0

    def lam(self, t, s):
        return (t - s) * self.u - s * self.d

    def status(self, t, s):
        """+1 certify, -1 stop without certifying, 0 continue. Counts points within TOL of a boundary."""
        L = self.lam(t, s)
        if abs(L - self.A) < TOL or abs(L + self.B) < TOL:
            self.ties += 1
        if L >= self.A:
            return 1
        if L <= -self.B:
            return -1
        return 0


def exact_law(num, den, design, T):
    """Exact law of the stopped walk under Bernoulli(num/den): weights are integers over den**t, rescaled to den**T.
    Returns (P(certify), P(stop without certifying), P(neither by T), E[min(tau, T)], E[Lambda_tau; tau <= T]) as
    Fractions; E[min(tau, T)] counts T for the mass still running at T. Lambda is a float logarithm."""
    cur, cert, rej, ek = {0: 1}, 0, 0, 0
    elam = 0.0
    for t in range(1, T + 1):
        nxt = {}
        for s, w in cur.items():
            nxt[s + 1] = nxt.get(s + 1, 0) + w * num
            nxt[s] = nxt.get(s, 0) + w * (den - num)
        cur, scale = {}, den ** (T - t)
        for s, w in nxt.items():
            st = design.status(t, s)
            if st == 0:
                cur[s] = w
            else:
                if st > 0:
                    cert += w * scale
                else:
                    rej += w * scale
                ek += t * w * scale
                elam += float(F(w, den ** t)) * design.lam(t, s)
        if not cur:
            break
    total = den ** T
    run = total - cert - rej
    return F(cert, total), F(rej, total), F(run, total), F(ek + T * run, total), elam


LG = [lgamma(k + 1) for k in range(1301)]          # log k!, shared by the binomial helpers


def binom_pmf_vec(nn, pr):
    """Binomial(nn, pr) pmf on 0..nn as a numpy array (float)."""
    k = np.arange(nn + 1)
    lg = np.array(LG[:nn + 1])
    return np.exp(LG[nn] - lg[k] - lg[nn - k] + k * log(pr) + (nn - k) * log1p(-pr))


def cum_cert(theta, des, T):
    """Float DP: probability of having certified within t trials, t = 0..T, at harm rate theta (no pool beyond T)."""
    cur, tot, cum = {0: 1.0}, 0.0, [0.0]
    for t in range(1, T + 1):
        nxt = {}
        for s, w in cur.items():
            nxt[s + 1] = nxt.get(s + 1, 0.0) + w * theta
            nxt[s] = nxt.get(s, 0.0) + w * (1 - theta)
        cur = {}
        for s, w in nxt.items():
            st = des.status(t, s)
            if st == 0:
                cur[s] = w
            elif st > 0:
                tot += w
        cum.append(tot)
    return np.array(cum)


def log_avg(des, nn, mm, size_rate, power_rate):
    """Size (at size_rate) and power (at power_rate) of the execution test averaged over the pool N ~ Bin(nn, mm)."""
    pm = binom_pmf_vec(nn, mm)
    c0, c1 = cum_cert(size_rate, des, nn), cum_cert(power_rate, des, nn)
    return float((pm * c0).sum()), float((pm * c1).sum())


# ---------------------------------------------------------------------------
# S1: exact laws at E1, Ville thresholds
# ---------------------------------------------------------------------------
def ville(a, c):
    return Design(a, c, log(1 / ALPHA), log(1 / BETA))


def report(tag, label, des, num_a, den_a, num_c, den_c, T, lb):
    cert1, rej1, run1, ek1, _ = exact_law(num_a, den_a, des, T)
    cert0, rej0, run0, ek0, _ = exact_law(num_c, den_c, des, T)
    assert des.ties == 0, des.ties
    return dict(label=label, size=float(cert0), power=float(cert1), rej=float(rej1), run=float(run1),
                ek=float(ek1), ek0=float(ek0), run0=float(run0), cert1=cert1, cert0=cert0)


def check_s1():
    t0 = time.time()
    out = {}
    lb_x = kl(1 - BETA, ALPHA) / kl(P, PBAR)
    lb_l = kl(1 - BETA, ALPHA) / kl(Q, QS)
    assert f"{lb_x:.1f}" == "12.3" and f"{lb_l:.1f}" == "50.7"
    # execution test: pool N = 479, the realized log
    dx = ville(P, PBAR)
    ub_x = (float(log(1 / ALPHA)) + dx.u) / dx.kl
    rx = report("S1a", "execution", dx, P.numerator, P.denominator, PBAR.numerator, PBAR.denominator, N, lb_x)
    assert rx["size"] <= 0.05 and rx["rej"] <= 0.2 and rx["power"] >= 1 - 0.2 - rx["run"]
    assert lb_x <= rx["ek"] <= ub_x
    assert rx["ek"] >= kl(rx["power"], rx["size"]) / dx.kl - 1e-12        # bound at the design's own (size, power)
    eta = 2 * ub_x / (float(M) * n) + exp(-float(M) ** 2 * n / 2)          # truncation allowance chi_n of the proposition at E1
    assert rx["run"] <= eta
    guarantee = 1 - float(BETA) - eta                                       # power the proposition guarantees for these thresholds
    assert f"{guarantee:.3f}" == "0.714" and rx["power"] > guarantee
    for nn, mm in ((1200, 479 / 1200), (200, 0.3), (50, 0.5), (400, 0.05)):   # Hoeffding: P(N < m n / 2) <= exp(-m^2 n / 2)
        zc = ceil(mm * nn / 2) - 1
        tail = sum(exp(lgamma(nn + 1) - lgamma(k + 1) - lgamma(nn - k + 1) + k * log(mm) + (nn - k) * log1p(-mm)) for k in range(0, zc + 1))
        assert tail <= exp(-mm * mm * nn / 2)
    out["x_ville"] = rx
    say("S1a", f"execution SPRT, Ville thresholds (log 20, log 5), pool N = {N}: exact size {rx['size']:.5f}, power "
               f"{rx['power']:.5f}, E[K] = {rx['ek']:.3f} under p, {rx['ek0']:.3f} at the boundary rate; P(tau > N) = "
               f"{rx['run']:.1e}; bounds: kl {lb_x:.2f} <= E[K] <= (log(1/alpha) + u)/kl = {ub_x:.2f} (u = {dx.u:.4f}, "
               f"d = {dx.d:.4f}, kl(p, pbar) = {dx.kl:.5f}); truncation allowance chi_n = 2 K-bar/(m n) + exp(-m^2 n/2) = {eta:.4f} "
               f"against the exact loss {rx['run']:.1e}; guaranteed power 1 - beta - chi_n = {guarantee:.3f}")
    # live test: unlimited pool (truncated at 2500)
    dl = ville(Q, QS)
    ub_l = (float(log(1 / ALPHA)) + dl.u) / dl.kl
    rl = report("S1b", "live", dl, 35, 1200, 120, 1200, 2500, lb_l)
    assert rl["size"] <= 0.05 and rl["rej"] <= 0.2 and rl["run"] < 1e-30 and rl["run0"] < 1e-30
    assert lb_l <= rl["ek"] <= ub_l
    assert rl["ek"] >= kl(rl["power"], rl["size"]) / dl.kl - 1e-12
    out["l_ville"] = rl
    say("S1b", f"live SPRT, Ville thresholds: exact size {rl['size']:.5f}, power {rl['power']:.5f}, E[K_L] = "
               f"{rl['ek']:.3f} under q, {rl['ek0']:.3f} at q*; unfinished mass {rl['run']:.1e}; bounds: kl {lb_l:.2f} <= "
               f"E[K_L] <= {ub_l:.2f} (u = {dl.u:.4f}, d = {dl.d:.4f}, kl(q, q*) = {dl.kl:.5f})")
    say("S1c", f"ratio of exact SPRT costs (executions / live episodes) {rx['ek'] / rl['ek']:.3f}; limit "
               f"kl(q,q*)/kl(p,pbar) = {dx.kl and kl(Q, QS) / dx.kl:.4f}; ratio of the necessity bounds {lb_x / lb_l:.4f} "
               f"({time.time() - t0:.1f}s)")
    wx, wl = ub_x / lb_x, ub_l / lb_l
    assert f"{wx:.1f}" == "1.7" and f"{wl:.1f}" == "1.6"
    say("S1w", f"width of the sandwiches at alpha = 0.05, beta = 0.2: executions {ub_x:.2f} / {lb_x:.2f} = {wx:.3f}, live "
               f"{ub_l:.2f} / {lb_l:.2f} = {wl:.3f}")
    return out, lb_x, lb_l


def check_assembly(ub_x):
    """Proposition (ii), assembled: with beta' = beta - chi_n the test is valid and powerful on average over the log."""
    t0 = time.time()
    rows = []
    for nn in (1200, 600, 530):
        chi = 2 * ub_x / (float(M) * nn) + exp(-float(M) ** 2 * nn / 2)
        assert chi < float(BETA)
        bp = float(BETA) - chi
        des = Design(P, PBAR, float(log(1 / ALPHA)), log(1 / bp))
        size, power = log_avg(des, nn, float(M), float(PBAR), float(P))
        assert des.ties == 0 and size <= float(ALPHA) and power >= 1 - float(BETA), (nn, size, power)
        rows.append(f"n = {nn}: chi_n = {chi:.4f}, beta' = {bp:.4f}, size {size:.4f}, power {power:.4f}")
    say("S1d", "log-averaged size and power of the execution test with beta' = beta - chi_n (Ville thresholds log(1/alpha), "
               f"log(1/beta')); power at least 1 - beta = 0.8 and size at most alpha = 0.05: " + "; ".join(rows)
               + f" ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# S2: best boundary pair on a grid (vectorized over pairs)
# ---------------------------------------------------------------------------
def grid_dp(a, c, Ag, Bg, T):
    """Vectorized forward DP over all boundary pairs (A, B) in Ag x Bg, at harm rates c (size) and a (power, cost).
    Returns arrays (size, power, E_a[min(tau, T)], mass still running at T under a) of shape (len(Ag) * len(Bg),)."""
    u, d = log((1 - a) / (1 - c)), log(c / a)
    w = u + d
    AA = np.repeat(Ag, len(Bg))[:, None]
    BB = np.tile(Bg, len(Ag))[:, None]
    Pn, Amax, Bmax = AA.shape[0], Ag.max(), Bg.max()
    K = int(ceil((Amax + Bmax) / w)) + 6

    def base(t):
        return floor((t * u - Amax) / w) - 1
    W0, W1 = np.zeros((Pn, K)), np.zeros((Pn, K))
    bprev = base(0)
    W0[:, -bprev] = 1.0
    W1[:, -bprev] = 1.0
    cert0, cert1, ek1 = np.zeros(Pn), np.zeros(Pn), np.zeros(Pn)
    ks = np.arange(K)
    for t in range(1, T + 1):
        bnew = base(t)
        db = bnew - bprev
        assert db in (0, 1)
        for W, th in ((W0, c), (W1, a)):
            new = np.zeros_like(W)
            if db == 0:
                new += (1 - th) * W
                new[:, 1:] += th * W[:, :-1]
            else:
                assert not W[:, 0].any() and not W[:, -1].any()
                new[:, :-1] += (1 - th) * W[:, 1:]
                new += th * W
            W[:] = new
        s = bnew + ks
        lam = (t - s) * u - s * d
        cm = lam[None, :] >= AA
        rm = lam[None, :] <= -BB
        for W, cert in ((W0, cert0), (W1, cert1)):
            cert += (W * cm).sum(1)
        ek1 += t * ((W1 * (cm | rm)).sum(1))
        for W in (W0, W1):
            W[cm | rm] = 0.0
        bprev = bnew
    run1 = W1.sum(1)
    return cert0, cert1, ek1 + T * run1, run1


def best_pair(des, Ag, Bg, T, size_max=0.05, power_min=0.80):
    cert0, cert1, ek1, run1 = grid_dp(des.a, des.c, Ag, Bg, T)
    ok = (cert0 <= size_max) & (cert1 >= power_min)
    assert ok.any()
    order = np.argsort(np.where(ok, ek1, np.inf))
    return order, cert0, cert1, ek1, run1


def check_s2(s1, lb_x, lb_l):
    t0 = time.time()
    res = {}
    for key, a, c, T, Ag, Bg, num_a, den_a, num_c, den_c, lb in (
        ("x", P, PBAR, N, np.round(np.arange(0.40, 4.001, 0.02), 4), np.round(np.arange(0.01, 3.001, 0.02), 4),
         P.numerator, P.denominator, PBAR.numerator, PBAR.denominator, lb_x),
        ("l", Q, QS, 1500, np.round(np.arange(1.00, 4.501, 0.05), 4), np.round(np.arange(0.20, 3.001, 0.05), 4),
         35, 1200, 120, 1200, lb_l),
    ):
        des = Design(a, c, 1.0, 1.0)
        order, cert0, cert1, ek1, run1 = best_pair(des, Ag, Bg, T)
        found = None
        for idx in order[:200]:                                     # first candidate that survives the exact recheck
            A_, B_ = float(Ag[idx // len(Bg)]), float(Bg[idx % len(Bg)])
            dz = Design(a, c, A_, B_)
            Texact = T if key == "x" else 2500
            c1, r1, run, ek, _ = exact_law(num_a, den_a, dz, Texact)
            c0, r0, run0, ek0, _ = exact_law(num_c, den_c, dz, Texact)
            if dz.ties == 0 and c0 <= F(1, 20) and c1 >= F(4, 5):
                found = (A_, B_, c0, c1, ek, ek0, run, run0)
                break
        assert found is not None
        A_, B_, c0, c1, ek, ek0, run, run0 = found
        assert abs(ek - float(ek1[idx])) < 5e-3 and ek >= lb
        assert ek >= kl(float(c1), float(c0)) / des.kl - 1e-12
        sz = [walk(min(1.0, float(c) * g), dz, N if key == 'x' else 1500)[0] for g in (1.0, 1.05, 1.2, 1.5, 1.9)]   # size over p-hat >= boundary
        assert all(x >= y - 1e-15 for x, y in zip(sz, sz[1:])) and sz[0] <= 0.05 and abs(sz[0] - float(c0)) < 1e-9
        res[key] = dict(pairs=len(Ag) * len(Bg), A=A_, B=B_, size=float(c0), power=float(c1), ek=float(ek), ek0=float(ek0), run=float(run),
                        na=len(Ag), nb=len(Bg), step=float(Ag[1] - Ag[0]))
        edge = (f"A at the grid edge: {A_ in (float(Ag.min()), float(Ag.max()))}; B at the lower grid edge: "
                f"{B_ == float(Bg.min())}")
        extra = ""
        if key == "x":
            # (a) log-averaged size and power agree with the given-log values (the test does not read the log beyond N)
            sz_avg, pw_avg = log_avg(dz, n, float(M), float(PBAR), float(P))
            assert sz_avg <= 0.05 and pw_avg >= 0.80 and abs(sz_avg - float(c0)) < 5e-6 and abs(pw_avg - float(c1)) < 5e-6
            # (b) every B in (0, 0.01] gives the identical test (no lattice value of Xi lies in [-0.01, 0))
            for b_small in (0.01, 0.005, 0.001):
                d2 = Design(a, c, A_, b_small)
                r2 = exact_law(num_a, den_a, d2, T)
                assert d2.ties == 0 and r2[:2] == (c1, r1) and r2[3] == ek, b_small
                r20 = exact_law(num_c, den_c, d2, T)
                assert r20[0] == c0
            extra = (f"; averaged over the pool N ~ Bin(n, m): size {sz_avg:.6f}, power {pw_avg:.6f} (given N = 479: "
                     f"{float(c0):.6f}, {float(c1):.6f}); B in (0, 0.01] gives the identical test")
        say("S2" + key, f"{'execution' if key == 'x' else 'live'} test, best of {len(Ag) * len(Bg)} boundary pairs "
                        f"({len(Ag)} x {len(Bg)}, grid step {float(Ag[1] - Ag[0]):g}): (A, B) = ({A_:.2f}, {B_:.2f}); exact size "
                        f"{float(c0):.5f}, power {float(c1):.5f}, E = {float(ek):.3f} (bound {lb:.2f}, "
                        f"{100 * (float(ek) / lb - 1):.1f}% above); unfinished mass {float(run):.1e}; {edge}{extra}")
    say("S2z", f"cost ratio of the best pairs, executions over live episodes: {res['x']['ek'] / res['l']['ek']:.3f} "
               f"({time.time() - t0:.1f}s)")
    return res


# ---------------------------------------------------------------------------
# Float walk for random designs (S3)
# ---------------------------------------------------------------------------
def walk(theta, des, T):
    """Float law of the stopped walk: (P(cert), P(rej), E[tau ^ T], E[Lambda_tau; stopped], max Lambda at cert,
    min Lambda at rej, P(tau > T))."""
    cur, cert, rej, ek, elam = {0: 1.0}, 0.0, 0.0, 0.0, 0.0
    lam_up, lam_dn = -1e9, 1e9
    for t in range(1, T + 1):
        nxt = {}
        for s, w in cur.items():
            nxt[s + 1] = nxt.get(s + 1, 0.0) + w * theta
            nxt[s] = nxt.get(s, 0.0) + w * (1 - theta)
        cur = {}
        for s, w in nxt.items():
            st = des.status(t, s)
            if st == 0:
                cur[s] = w
                continue
            ek += t * w
            elam += w * des.lam(t, s)
            if st > 0:
                cert += w
                lam_up = max(lam_up, des.lam(t, s))
            else:
                rej += w
                lam_dn = min(lam_dn, des.lam(t, s))
        if not cur:
            break
    run = sum(cur.values())
    return cert, rej, ek + T * run, elam, lam_up, lam_dn, run


def check_s3(seed=20261008, trials=120):
    t0 = time.time()
    rng = random.Random(seed)
    worst = dict(supermart=0.0, ville_size=0.0, ville_rej=0.0, mono=0.0, wald=0.0, overshoot=0.0, own=0.0, trunc=0.0)
    for _ in range(trials):
        a = rng.uniform(0.01, 0.4)
        c = rng.uniform(a + 0.15, min(0.95, a + 0.6))                # a gap keeps the tail of tau short
        A, B = rng.uniform(0.6, 4.0), rng.uniform(0.4, 3.5)
        des = Design(a, c, A, B)
        # (1) E[e^Z] = 1 - (theta - c)(c - a)/(c(1-c)), at most 1 exactly when theta >= c; E_a[e^{-Z}] = 1
        for theta in (0.0, a, 0.5 * (a + c), c, 0.5 * (c + 1), 1.0):
            g = theta * a / c + (1 - theta) * (1 - a) / (1 - c)
            worst["supermart"] = max(worst["supermart"], abs(g - (1 - (theta - c) * (c - a) / (c * (1 - c)))))
            assert (g <= 1 + 1e-12) == (theta >= c - 1e-15)
        assert abs(a * c / a + (1 - a) * (1 - c) / (1 - a) - 1) < 1e-12
        T = 1500
        cert1, rej1, ek1, elam1, lam_up, lam_dn, run1 = walk(a, des, T)
        assert run1 < 1e-9
        # (2) Wald: E[Lambda_tau] = kl E[tau]; overshoot: Lambda_tau < A + u at certification, > -B - d at the lower exit
        worst["wald"] = max(worst["wald"], abs(elam1 - des.kl * ek1))
        assert lam_up < A + des.u + 1e-12 and lam_dn > -B - des.d - 1e-12
        worst["overshoot"] = max(worst["overshoot"], (lam_up - A) / des.u)
        assert ek1 * des.kl <= A + des.u + 1e-9
        # (3) Ville: certification at most e^{-A} at every rate theta >= c (monotone nonincreasing), reject at most e^{-B} under a
        prev, sizes = None, []
        for theta in sorted({c, 0.5 * (c + 1), min(1.0, c + 0.05), min(1.0, c + 0.2), c + 0.5 * (1 - c), 1.0}):
            ce, _, _, _, _, _, rn = walk(theta, des, T)
            sizes.append((theta, ce))
        sizes.sort()
        for (th1, c1), (th2, c2) in zip(sizes, sizes[1:]):
            worst["mono"] = max(worst["mono"], c2 - c1)           # must stay <= 0: nonincreasing in theta
        cert0 = sizes[0][1]
        assert cert0 <= exp(-A) + 1e-12 and rej1 <= exp(-B) + 1e-12
        worst["ville_size"] = max(worst["ville_size"], cert0 / exp(-A))
        worst["ville_rej"] = max(worst["ville_rej"], rej1 / exp(-B))
        # (4) the bound at the design's own operating point
        own = kl(cert1, cert0) / des.kl
        assert ek1 >= own - 1e-9
        worst["own"] = max(worst["own"], own - ek1)
        # (4b) finiteness: r consecutive harmless outcomes certify from anywhere in the strip, so P(tau > k r) <= (1 - (1-a)^r)^k
        r = int(ceil((A + B) / des.u))
        for kk in (1, 2, 3):
            assert walk(a, des, kk * r)[-1] <= (1 - (1 - a) ** r) ** kk + 1e-12
        # (5) truncation: P(tau > N) <= E[tau]/N for N = 40
        _, _, _, _, _, _, runN = walk(a, des, 40)
        assert runN <= ek1 / 40 + 1e-12
        worst["trunc"] = max(worst["trunc"], runN - ek1 / 40)
    assert worst["mono"] <= 1e-12 and worst["supermart"] < 1e-12 and worst["wald"] < 1e-8
    say("S3", f"{trials} random designs: E[e^Z] = 1 - (theta-c)(c-a)/(c(1-c)) (max error {worst['supermart']:.1e}); size at most "
              f"e^-A ({worst['ville_size']:.3f} of it at most) and reject probability at most e^-B ({worst['ville_rej']:.3f}); "
              f"certification nonincreasing in the harm rate, arbitrary thresholds (max rise {worst['mono']:.1e}); Wald's identity (max error "
              f"{worst['wald']:.1e}); overshoot (Lambda_tau - A)/u < 1 at certification (max {worst['overshoot']:.3f}); E[tau] kl <= A + u; "
              f"P(tau > N) <= E[tau]/N ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# S4: the corollary
# ---------------------------------------------------------------------------
def vmax(a, c):
    """Largest x(1-x) on [a, c]."""
    if a <= 0.5 <= c:
        return 0.25
    return max(a * (1 - a), c * (1 - c))


def check_s4(s1, lb_x, lb_l):
    t0 = time.time()
    rng = random.Random(7)
    # (a) the integral form of kl and its two-sided bracket
    worst_lo = worst_hi = 0.0
    for _ in range(20000):
        a = rng.uniform(1e-4, 0.98)
        c = rng.uniform(a + 1e-3, 0.9999) if a < 0.9985 else None
        if c is None:
            continue
        k = kl(a, c)
        lo = (c - a) ** 2 / (2 * vmax(a, c))
        hi = (c - a) ** 2 / (c * (1 - c))
        assert lo <= k + 1e-12 and k <= hi + 1e-12, (a, c, k, lo, hi)
        worst_lo, worst_hi = max(worst_lo, lo / k), max(worst_hi, k / hi)
    say("S4a", f"(c-a)^2/(2 max x(1-x)) <= kl(a,c) <= (c-a)^2/(c(1-c)) on 20000 pairs (extremes lo/kl {worst_lo:.3f}, kl/hi {worst_hi:.3f})")
    # (b) the ratio bracket on random settings of the proposition
    cnt, widest, small = 0, 0.0, 0
    while cnt < 5000:
        q0, m, p = rng.uniform(0.002, 0.3), rng.uniform(0.03, 0.9), rng.uniform(0.002, 0.6)
        q = q0 + m * p
        qs = rng.uniform(q, min(0.999, q0 + m))
        zeta = qs - q
        if not (0.01 < zeta and qs < q0 + m and q0 + m <= 1 and 0 < q):    # gaps above 0.01 keep float kl accurate
            continue
        gam = rng.uniform(0, 0.98) * zeta / m
        pbar = p + (zeta - m * gam) / m
        if not (p + 0.01 < pbar < 1):
            continue
        cnt += 1
        k_x, k_l = kl(p, pbar), kl(q, qs)
        Qe = m * m * pbar * (1 - pbar) / (zeta - m * gam) ** 2
        Ql = qs * (1 - qs) / zeta ** 2
        th_e, th_l = vmax(p, pbar) / (pbar * (1 - pbar)), vmax(q, qs) / (qs * (1 - qs))
        tol = 1 + 1e-9
        assert Qe <= tol / k_x and 1 / k_x <= 2 * th_e * Qe * tol and Ql <= tol / k_l and 1 / k_l <= 2 * th_l * Ql * tol
        R = k_l / k_x
        lo, hi = Qe / (2 * th_l * Ql), 2 * th_e * Qe / Ql
        assert lo <= R * (1 + 1e-9) and R <= hi * (1 + 1e-9), (R, lo, hi)
        widest = max(widest, hi / lo)
        if pbar <= 0.5 and qs <= 0.5:                     # the case of the corollary: R within a factor 2 of Qe/Ql
            small += 1
            assert abs(th_e - 1) < 1e-12 and abs(th_l - 1) < 1e-12
            assert 0.5 * Qe / Ql <= R * (1 + 1e-9) and R <= 2 * Qe / Ql * (1 + 1e-9), (R, Qe / Ql)
    assert small > 500
    say("S4b", f"ratio bracket Qe/(2 th_l Ql) <= kl(q,q*)/kl(p,pbar) <= 2 th_e Qe/Ql on {cnt} random settings; "
               f"th = 1 and R within a factor 2 of Qe/Ql on the {small} with pbar and q* at most 1/2")
    # (c) E1 values of the quadratic forms and the bracket
    Qe = float(M ** 2 * PBAR * (1 - PBAR) / (ZETA - M * GAMMA) ** 2)
    Ql = float(QS * (1 - QS) / ZETA ** 2)
    R = kl(Q, QS) / kl(P, PBAR)
    assert PBAR <= F(1, 2) and QS <= F(1, 2)
    assert f"{Qe / Ql:.2f}" == "0.28" and f"{R:.4f}" == f"{lb_x / lb_l:.4f}" and 0.5 * Qe / Ql <= R <= 2 * Qe / Ql
    say("S4c", f"E1: Qe = {Qe:.3f} (m^2 pbar(1-pbar)/(zeta - m gamma)^2), Ql = {Ql:.3f} (q*(1-q*)/zeta^2), Qe/Ql = {Qe / Ql:.3f}; "
               f"limit ratio kl(q,q*)/kl(p,pbar) = {R:.4f}; bracket [{Qe / (2 * Ql):.3f}, {2 * Qe / Ql:.3f}]")
    # (d) upper-to-lower bound ratio tends to 1 as alpha, beta -> 0; exact SPRT cost over kl(1-beta, alpha)
    des_x, des_l = Design(P, PBAR, 1.0, 1.0), Design(Q, QS, 1.0, 1.0)
    prev = (9.0, 9.0)
    rows = []
    for j in range(1, 10):
        al = be = 10.0 ** (-j)
        low = kl(1 - be, al)
        r_x, r_l = (log(1 / al) + des_x.u) / low, (log(1 / al) + des_l.u) / low
        assert r_x < prev[0] and r_l < prev[1] and r_x > 1 and r_l > 1
        for be2 in (be, 5 * be, 0.2):                      # log(1/alpha) - kl(1-beta, alpha) <= beta (log(1/alpha) + log(1/beta) + 1)
            assert log(1 / al) - kl(1 - be2, al) <= be2 * (log(1 / al) + log(1 / be2) + 1) + 1e-12
        prev = (r_x, r_l)
        rows.append((j, r_x, r_l))
    assert rows[-1][1] < 1.011 and rows[-1][2] < 1.005
    say("S4d", "upper/lower bound ratio (log(1/alpha) + u)/kl(1-beta, alpha) at alpha = beta = 10^-j, executions and live: "
               + ", ".join(f"j={j}: {rx:.4f}/{rl:.4f}" for j, rx, rl in rows if j in (1, 2, 4, 6, 9)))
    ex_rows = []
    for j in (1, 2, 3, 4):
        al = be = 10.0 ** (-j)
        dz = Design(P, PBAR, log(1 / al), log(1 / be))
        _, _, ek, elam, _, _, run = walk(float(P), dz, 4000)
        assert run < 1e-12
        low = kl(1 - be, al) / dz.kl
        up = (log(1 / al) + dz.u) / dz.kl
        assert low <= ek <= up
        ex_rows.append((j, ek, ek / low))
    assert all(x[2] > y[2] for x, y in zip(ex_rows, ex_rows[1:]))
    say("S4e", "exact E[tau] of the execution SPRT with Ville thresholds, over the necessity bound, at alpha = beta = 10^-j: "
               + ", ".join(f"j={j}: {ek:.1f} ({r:.3f}x)" for j, ek, r in ex_rows) + f" ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# S5: finite-log remark
# ---------------------------------------------------------------------------
def binom_cdf_f(nn, p, z):
    """P(Bin(nn, p) <= z) in floats."""
    if z < 0:
        return 0.0
    return float(binom_pmf_vec(nn, p)[:min(z, nn) + 1].sum())


def cp_upper(z, nn, a):
    """One-sided Clopper-Pearson upper limit at level 1 - a: the root of t -> P(Bin(nn, t) <= z) = a."""
    lo, hi = 0.0, 1.0
    for _ in range(55):
        mid = (lo + hi) / 2
        if binom_cdf_f(nn, mid, z) > a:
            lo = mid
        else:
            hi = mid
    return hi


def up_step(a, c):
    """The largest upward step of the log-likelihood ratio, up(a, c) = log((1 - a)/(1 - c))."""
    return log((1 - a) / (1 - c))


def remark_loss(nn, alpha_test, alpha_log):
    """Exact multinomial log law at n = nn (q0, m of E1): P(pbar^U <= p) and E[min(1, Kbar^U / N); pbar^U > p], with
    Clopper-Pearson limits at level alpha_log / 2 each and Kbar^U = (log(1/alpha_test) + up(p, pbar^U)) / kl(p, pbar^U)."""
    q0f, mf, pf, qsf = float(Q0), float(M), float(P), float(QS)
    rf = 1 - q0f - mf
    sb, sN = sqrt(nn * q0f * (1 - q0f)), sqrt(nn * mf * (1 - mf))
    bs = range(max(0, int(nn * q0f - 6 * sb)), int(nn * q0f + 6 * sb) + 2)
    Ns = range(max(1, int(nn * mf - 6 * sN)), int(nn * mf + 6 * sN) + 2)
    q0U = {bb: cp_upper(bb, nn, alpha_log / 2) for bb in bs}
    mU = {NN: cp_upper(NN, nn, alpha_log / 2) for NN in Ns}
    tail = loss = mass = 0.0
    for bb in bs:
        for NN in Ns:
            if bb + NN > nn:
                continue
            w = exp(LG[nn] - LG[bb] - LG[NN] - LG[nn - bb - NN] + bb * log(q0f) + NN * log(mf) + (nn - bb - NN) * log(rf))
            mass += w
            pu = (qsf - q0U[bb]) / mU[NN] - float(GAMMA)
            if pu <= pf:
                tail += w
            else:
                pu = min(pu, 1 - 1e-9)
                loss += w * min(1.0, (log(1 / alpha_test) + up_step(pf, pu)) / kl(pf, pu) / NN)
    assert mass > 1 - 1e-6
    return tail, loss


def check_s5(s1):
    t0 = time.time()
    rng = random.Random(3)
    for _ in range(2000):                                         # pbar is decreasing in q0 and in m when q* > q0
        qs, q0, m = rng.uniform(0.05, 0.4), rng.uniform(0.0, 0.04), rng.uniform(0.05, 0.9)
        g = rng.uniform(0.0, 0.1)
        f = lambda x0, mm: (qs - x0) / mm - g
        assert f(q0 + 0.001, m) < f(q0, m) and f(q0, m + 0.001) < f(q0, m)
    # (a) E1 realized log, matched total error 0.05: alpha = 0.04 for the test and alpha_log = 0.01 for the limits
    alpha_t, alpha_log = 0.04, 0.01
    q0u, mu = cp_upper(b, n, alpha_log / 2), cp_upper(N, n, alpha_log / 2)
    pbar_used = (float(QS) - q0u) / mu - float(GAMMA)
    assert 0 < float(P) < pbar_used < float(PBAR)
    dz = Design(P, pbar_used, log(1 / alpha_t), float(log(1 / BETA)))
    cert0, _, _, ek0, _, _, run0 = walk(float(PBAR), dz, N)       # exact law given this log, at the true boundary rate
    cert1, rej1, ek1, _, _, _, run1 = walk(float(P), dz, N)       # and at p
    cert_bd, *_ = walk(pbar_used, dz, N)                          # and at the boundary the test uses
    assert dz.ties == 0 and cert0 <= cert_bd + 1e-12 and cert_bd <= alpha_t
    kb = (log(1 / alpha_t) + up_step(float(P), pbar_used)) / kl(P, pbar_used)
    assert ek1 <= kb and cert1 >= 1 - float(BETA) - kb / N          # given the log: E[K] <= Kbar^U, P(no cert) <= beta' + Kbar^U / N
    x = s1["x_ville"]
    say("S5a", f"finite log (b, N) = (27, 479) of n = 1200, 99.5% Clopper-Pearson limits: q0^U = {q0u:.4f}, m^U = {mu:.4f}, "
               f"pbar^U = {pbar_used:.4f} (true pbar {float(PBAR):.4f}); the Ville test with alpha = {alpha_t} (total error "
               f"{alpha_t + alpha_log:.2f}) and beta' = beta has, given this log, size {cert_bd:.4f} at pbar^U and {cert0:.4f} at the true "
               f"boundary rate, power {cert1:.4f}, E[K] = {ek1:.2f} <= Kbar^U = {kb:.1f}; guarantee 1 - beta - Kbar^U/N = "
               f"{1 - float(BETA) - kb / N:.3f}; known q0, m: power {x['power']:.3f}, E[K] = {x['ek']:.1f}")
    dz5 = Design(P, (float(QS) - q0u) / mu, float(log(1 / ALPHA)), float(log(1 / BETA)))
    _, _, ek5, *_ = walk(float(P), dz5, N)
    assert f"{ek5:.1f}" == "24.9"                                 # nominal value, alpha = 0.05 (total error 0.06)
    # (b) the loss terms of the unconditional power over the random log
    rows, prev = [], 2.0
    for nn in (100, 200, 400, 1200):
        tail, loss = remark_loss(nn, alpha_t, alpha_log)
        assert tail < prev
        prev = tail
        rows.append((nn, tail, loss))
    assert rows[-1][1] < 1e-5 and rows[-1][2] < rows[0][2] and rows[-2][2] < rows[1][2]
    say("S5b", "random log: exact P(pbar^U <= p) and E[min(1, Kbar^U/N); pbar^U > p] at n = "
               + ", ".join(f"{nn}: {tl:.1e} and {ls:.3f}" for nn, tl, ls in rows) + f" ({time.time() - t0:.1f}s)")
    return dict(pbar_used=pbar_used, ek=ek1, power=cert1)


# ---------------------------------------------------------------------------
# S6: the finite pool
# ---------------------------------------------------------------------------
def np_power(nn, pa, pb, alpha):
    """Power at harm rate pa of the most powerful randomized test of size alpha for the null Bernoulli(pb), pb > pa, that
    certifies when the harm count is small, with all nn executions (Neyman-Pearson; it is valid for every rate above pb)."""
    z = -1
    while binom_cdf_f(nn, pb, z + 1) <= alpha:
        z += 1
    pm0, pm1 = binom_pmf_vec(nn, pb), binom_pmf_vec(nn, pa)
    r = (alpha - binom_cdf_f(nn, pb, z)) / pm0[z + 1]
    assert 0 <= r < 1
    return binom_cdf_f(nn, pa, z) + r * pm1[z + 1]


def check_s6(dx_ub):
    t0 = time.time()
    base = float(P + ZETA / M)                                    # pbar = base - gamma
    lo, hi = 0.10, float(ZETA / M) - 1e-6
    for _ in range(50):
        mid = (lo + hi) / 2
        if np_power(N, float(P), base - mid, 0.05) >= 0.80:
            lo = mid
        else:
            hi = mid
    g_star = lo
    assert f"{g_star:.3f}" == "0.159" and f"{float(ZETA / M):.3f}" == "0.177"
    d1 = 0.1476                                                   # D1 tolerance of Appendix app:demo-gates
    pw_d1, pw_hi = np_power(N, float(P), base - d1, 0.05), np_power(N, float(P), base - 0.160, 0.05)
    assert pw_d1 > 0.95 and pw_hi < 0.80
    # chi_n < beta is only sufficient: it fails at gamma = 0.15, where the all-479 test still has power above 0.8
    pb15 = base - 0.15
    k15 = (log(20) + up_step(float(P), pb15)) / kl(P, pb15)
    chi15 = 2 * k15 / (float(M) * n) + exp(-float(M) ** 2 * n / 2)
    assert chi15 > 0.2 and np_power(N, float(P), pb15, 0.05) > 0.95
    say("S6", f"finite pool N = 479: the most powerful size-0.05 test with all executions has power 0.80 at gamma = {g_star:.4f} "
              f"(power {pw_d1:.3f} at the D1 tolerance 0.1476, {pw_hi:.3f} at 0.160); zeta/m = {float(ZETA / M):.4f}; at gamma = 0.15, "
              f"Kbar = {k15:.0f} and chi_n = {chi15:.2f} > beta ({time.time() - t0:.1f}s)")
    return dict(gamma_np=g_star)


# ---------------------------------------------------------------------------
# Fixed-sample comparators (as in verify_budget_limits.py, L4)
# ---------------------------------------------------------------------------
def binom_cdf(nn, t, z):
    if z < 0:
        return F(0)
    if z >= nn:
        return F(1)
    tn, td = t.numerator, t.denominator
    return F(sum(comb(nn, k) * tn ** k * (td - tn) ** (nn - k) for k in range(z + 1)), td ** nn)


def np_min_n(pa, pb, alpha, beta, randomized, nmax=400):
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
    raise AssertionError


def check_fixed():
    got = (np_min_n(PBAR, P, ALPHA, BETA, False), np_min_n(QS, Q, ALPHA, BETA, False),
           np_min_n(PBAR, P, ALPHA, BETA, True), np_min_n(QS, Q, ALPHA, BETA, True))
    assert got == (23, 76, 17, 74), got
    say("Sf", f"fixed-sample lower-tail tests at E1: {got[0]} executions and {got[1]} live episodes "
              f"({got[2]} and {got[3]} with randomization at the cutoff)")


def check_paper_numbers(s1, res, s5, s6, lb_x, lb_l):
    """Every number printed in the E1 Achievability Check paragraph of supplement/appendix-F2.md."""
    x, l, bx, bl = s1["x_ville"], s1["l_ville"], res["x"], res["l"]
    dx, dl = Design(P, PBAR, float(log(1 / ALPHA)), float(log(1 / BETA))), Design(Q, QS, float(log(1 / ALPHA)), float(log(1 / BETA)))
    kb_x = (float(log(1 / ALPHA)) + dx.u) / dx.kl
    kb_l = (float(log(1 / ALPHA)) + dl.u) / dl.kl
    chi = 2 * kb_x / (float(M) * n) + exp(-float(M) ** 2 * n / 2)
    R = kl(Q, QS) / kl(P, PBAR)
    got = [f"{x['size']:.3f}", f"{x['power']:.3f}", f"{x['ek']:.1f}", f"{lb_x:.1f}", f"{kb_x:.1f}", f"{1 - float(BETA) - chi:.3f}",
           f"{l['size']:.3f}", f"{l['power']:.3f}", f"{l['ek']:.1f}", f"{lb_l:.1f}", f"{kb_l:.1f}",
           f"{kb_x / lb_x:.1f}", f"{kb_l / lb_l:.1f}",
           f"{bx['na']}x{bx['nb']}={bx['pairs']}", f"{bx['step']:g}", f"{bx['ek']:.1f}", f"{bx['size']:.3f}", f"{bx['power']:.3f}",
           f"{100 * (bx['ek'] / lb_x - 1):.1f}",
           f"{bl['na']}x{bl['nb']}={bl['pairs']}", f"{bl['step']:g}", f"{bl['ek']:.1f}", f"{bl['size']:.3f}", f"{bl['power']:.3f}",
           f"{bx['ek'] / bl['ek']:.3f}", f"{R:.3f}",
           f"{s6['gamma_np']:.3f}", f"{s5['pbar_used']:.3f}", f"{s5['ek']:.1f}", f"{s5['power']:.3f}"]
    want = ["0.038", "0.899", "17.0", "12.3", "20.6", "0.714", "0.042", "0.874", "63.7", "50.7", "81.7", "1.7", "1.6",
            "181x150=27150", "0.02", "12.7", "0.049", "0.801", "3.4", "71x57=4047", "0.05", "53.5", "0.048", "0.815",
            "0.238", "0.243", "0.159", "0.147", "26.4", "0.896"]
    assert got == want, list(zip(got, want))
    say("Sp", "all numbers of the E1 Achievability Check paragraph reproduced: " + ", ".join(got))


def main():
    t_start = time.time()
    s1, lb_x, lb_l = check_s1()
    dx = ville(P, PBAR)
    check_assembly((float(log(1 / ALPHA)) + dx.u) / dx.kl)
    res = check_s2(s1, lb_x, lb_l)
    check_s3()
    check_s4(s1, lb_x, lb_l)
    s5 = check_s5(s1)
    s6 = check_s6((float(log(1 / ALPHA)) + dx.u) / dx.kl)
    check_fixed()
    x, l, bx, bl = s1["x_ville"], s1["l_ville"], res["x"], res["l"]
    check_paper_numbers(s1, res, s5, s6, lb_x, lb_l)
    print(f"SUMMARY executions: bound {lb_x:.1f}, grid-best SPRT {bx['ek']:.1f}, Ville SPRT {x['ek']:.1f}, fixed 23 (17)")
    print(f"SUMMARY live: bound {lb_l:.1f}, grid-best SPRT {bl['ek']:.1f}, Ville SPRT {l['ek']:.1f}, fixed 76 (74)")
    print(f"All execution-budget achievability checks PASSED in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

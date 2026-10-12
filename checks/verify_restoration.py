#!/usr/bin/env python
"""Exact checks for the restoration allowance of targeted execution.

Paper map: Lemma F.7.2 (lem:restore) and Remark F.7.3 (rem:restore) (Appendix F.7 (app:budget),
after Proposition F.7.1 (prop:budget)); the E1 numbers of Section 4.1 (sec:demo-retry) and Appendix D.3 (app:demo-gates).

Run with: python verify_restoration.py

The finite-mechanism engine of verify_replay.py and the binomial tools of verify_budget.py are
loaded from this folder. Checks:
- (L1) Decomposition. A retry-type change on a finite mechanism: task Z, score S, a gate that
  refuses when S = 1, outcome X in {0, 1, 2}. The live mechanism and the restored mechanism
  differ only in the outcome kernel after a refusal. For the records trivial, Z = 0 and Z = 1:
  both have one baseline law and one conditioning mass; each continuing prefix has the same
  weight w_h under both changed laws; the agree term is the same under both; and both forward
  laws equal nu + sum_h w_h Q_h, computed from P^0 and the two continuation laws.
- (L2) Exhaustive bound. For all 15^4 = 50,625 pairs of live and restored continuation laws
  (grid of denominator 4, two continuing prefixes) and every event B of {0, 1, 2}:
  |P_hat(B) - P(B)| <= sum_h w_h gamma_h <= m * max_h gamma_h, and
  |mean_hat - mean| <= (max X - min X) * sum_h w_h gamma_h, with m the continuing mass.
- (L3) Attaining cases. Equal gamma_h: the event bound and the range-factor mean bound hold
  with equality (9/100 and 9/50). Unequal gamma_h: the averaged form is attained (3/50) and the
  uniform form m * max_h gamma_h (9/100) is strict.
- (L4) The E1 arithmetic: identified 35/1200, exact upper limit u in (0.040331, 0.040332],
  continuing mass 485/1200, target 1/10, largest tolerable gamma in (0.147632, 0.147635).
- (L5) The three labeled E1 tolerances of Appendix D.3: plug-in (continuing mass 485/1200), joint 95%
  (exact 97.5% upper limits on the continuing mass and on harm, by a union bound), and conservative
  (mass at most 1); each printed value is rounded down.
"""
import importlib.util
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vr = _load("verify_replay")
vb = _load("verify_budget")

XS = (0, 1, 2)                       # values of X, so max X - min X = 2
RANGE = 2
EVENTS = [set(c) for r in range(len(XS) + 1) for c in combinations(XS, r)]
PRIOR_Z = {0: F(3, 5), 1: F(2, 5)}
P_SCORE = {0: F(1, 5), 1: F(3, 5)}   # P(S = 1 | Z)
EXEC_LAW = {0: {0: F(7, 10), 1: F(1, 5), 2: F(1, 10)},
            1: {0: F(1, 2), 1: F(1, 4), 2: F(1, 4)}}   # outcome law when the call executes
f0 = lambda t, i: "EXEC"
fk = lambda t, i: "REFUSE" if i[0] == 1 else "EXEC"
loss = lambda h: h["O"]
loss_of = vr.loss_of
no_abort = lambda h: {}
delta = lambda x: {x: F(1)}


def build(Q):
    """Mechanism whose outcome after a refusal follows the law Q[z]."""
    def outcome(i):
        z, s, g = i
        return EXEC_LAW[s] if g == "EXEC" else Q[z]
    return [
        vr.rand("Z", lambda h: (), lambda i: PRIOR_Z),
        vr.rand("S", lambda h: (h["Z"],), lambda i: {1: P_SCORE[i[0]], 0: 1 - P_SCORE[i[0]]}),
        vr.dec("G", "gate", 0, lambda h: (h["S"],)),
        vr.rand("O", lambda h: (h["Z"], h["S"], h["G"]), outcome),
    ]


def tv(P, Q):
    return sum(max(P.get(x, F(0)) - Q.get(x, F(0)), F(0)) for x in XS)


def prob(P, B):
    return sum(P.get(x, F(0)) for x in B)


def evaluate(Q, Qh, cond):
    """Live and restored laws from the engine, and both again from P^0 and the continuation laws."""
    live, rest = build(Q), build(Qh)
    P0 = vr.forward(live, {"gate": f0}, loss, no_abort)
    assert P0 == vr.forward(rest, {"gate": f0}, loss, no_abort)           # one baseline law
    Pk = vr.forward(live, {"gate": fk}, loss, no_abort)
    Pkr = vr.forward(rest, {"gate": fk}, loss, no_abort)
    mass = lambda P, sel=lambda w: True: sum(p for w, p in P.items() if cond(w) and sel(w))
    assert mass(P0) == mass(Pk) == mass(Pkr)                               # the conditioning mass
    nu, m, cont = vr.replay(P0, live, f0, fk, "gate", no_abort, cond)    # agree term (no stop here)
    agree = lambda P: {x: mass(P, lambda w: vr.get(w, "S") == 0 and loss_of(w) == x) / mass(P) for x in XS}
    assert vr.clean(agree(P0)) == vr.clean(agree(Pk)) == vr.clean(agree(Pkr)) == nu
    for pre, w in cont.items():                                            # weights are baseline values
        sel = lambda p, z=dict(pre)["Z"]: vr.get(p, "Z") == z and vr.get(p, "S") == 1
        assert mass(Pk, sel) / mass(Pk) == mass(Pkr, sel) / mass(Pkr) == w
    third = lambda QQ: vr.add(*[vr.scale(QQ[dict(pre)["Z"]], w) for pre, w in cont.items()])
    q, qh = vr.add(nu, third(Q)), vr.add(nu, third(Qh))
    assert q == vr.law(Pk, cond) and qh == vr.law(Pkr, cond)
    gam = [tv(Qh[dict(pre)["Z"]], Q[dict(pre)["Z"]]) for pre in cont]
    return m, sum(w * g for w, g in zip(cont.values(), gam)), max(gam), q, qh


def check_l1_decomposition():
    conds = {"trivial": lambda w: True, "Z=0": lambda w: vr.get(w, "Z") == 0,
             "Z=1": lambda w: vr.get(w, "Z") == 1}
    Q = {0: {0: F(1, 2), 1: F(1, 4), 2: F(1, 4)}, 1: {1: F(1)}}
    Qh = {0: {0: F(1, 4), 1: F(1, 2), 2: F(1, 4)}, 1: {0: F(1, 4), 2: F(3, 4)}}
    masses = []
    for nm, cond in conds.items():
        m, G, gmax, q, qh = evaluate(Q, Qh, cond)
        masses.append(m)
        assert 0 < G <= m * gmax
    assert masses == [F(9, 25), F(1, 5), F(3, 5)]
    print("PASS L1: decomposition on a finite retry mechanism: one baseline law, one conditioning mass, equal "
          "agree terms, and baseline weights for the live and restored mechanisms; continuing mass 9/25, 1/5, 3/5 "
          "for the records trivial, Z=0, Z=1; both forward laws equal agree + continue computed from P^0.")


def simplex(den):
    return [{x: F(v, den) for x, v in zip(XS, c) if v} for c in product(range(den + 1), repeat=3)
            if sum(c) == den]


def check_l2_exhaustive():
    grid = simplex(4)
    w = {0: PRIOR_Z[0] * P_SCORE[0], 1: PRIOR_Z[1] * P_SCORE[1]}     # 3/25 and 6/25
    m = sum(w.values())
    assert len(grid) == 15 and m == F(9, 25)
    ev = [[prob(V, B) for B in EVENTS] for V in grid]
    mu = [vr.mean(V) for V in grid]
    n_cases = n_tight = 0
    for a, b, c, d in product(range(15), repeat=4):                  # Q_0, Q_1, Qhat_0, Qhat_1
        g0, g1 = tv(grid[c], grid[a]), tv(grid[d], grid[b])
        G = w[0] * g0 + w[1] * g1
        assert G <= m * max(g0, g1)
        worst = F(0)
        for k in range(len(EVENTS)):
            dev = abs(w[0] * (ev[c][k] - ev[a][k]) + w[1] * (ev[d][k] - ev[b][k]))
            assert dev <= G
            worst = max(worst, dev)
        assert abs(w[0] * (mu[c] - mu[a]) + w[1] * (mu[d] - mu[b])) <= RANGE * G
        n_cases += 1
        n_tight += (worst == G and G > 0)
    assert n_cases == 15 ** 4
    print(f"PASS L2: exhaustive bound over {n_cases} pairs of continuation laws and all 8 events: "
          f"event error <= sum w_h gamma_h <= m*max gamma_h and mean error <= 2*sum w_h gamma_h; "
          f"the event bound is attained in {n_tight} of them.")


def check_l3_attaining():
    equal = ({0: delta(0), 1: delta(0)}, {z: {0: F(3, 4), 2: F(1, 4)} for z in (0, 1)})
    unequal = ({0: delta(0), 1: delta(0)}, {0: delta(0), 1: {0: F(3, 4), 2: F(1, 4)}})
    triv = lambda w: True
    m, G, gmax, q, qh = evaluate(*equal, triv)
    assert (m, G) == (F(9, 25), F(9, 100))
    assert prob(qh, {2}) - prob(q, {2}) == m * F(1, 4) == F(9, 100)               # event bound, equality
    assert vr.mean(qh) - vr.mean(q) == RANGE * m * F(1, 4) == F(9, 50)             # range factor, equality
    m, G, gmax, q, qh = evaluate(*unequal, triv)
    assert prob(qh, {2}) - prob(q, {2}) == G == F(3, 50)                           # averaged form, equality
    assert m * gmax == F(9, 100) > G                                               # uniform form, strict
    print("PASS L3: attaining cases: equal gamma_h = 1/4 gives event error 9/100 = m*gamma and mean error "
          "9/50 = 2*m*gamma; gamma_h = (0, 1/4) gives 3/50 = sum w_h gamma_h, below m*max gamma_h = 9/100.")


def check_l4_e1_arithmetic():
    n, z, m, tau = 1200, 35, F(485, 1200), F(1, 10)
    a_L = F(1, 40)                                   # the exact 95% interval has 2.5% per side
    lo, hi = F(40331, 10**6), F(40332, 10**6)
    assert vb.cp_upper_le_target(z, n, a_L, hi) and not vb.cp_upper_le_target(z, n, a_L, lo)
    assert not vb.cp_upper_le_target(z, n, a_L, F(40, 1000))                       # 0.040 is rounded down
    g_lo, g_hi = (tau - hi) / m, (tau - lo) / m                                    # u lies in (lo, hi]
    assert F(147632, 10**6) <= g_lo < g_hi < F(147635, 10**6)
    assert (tau - F(40, 1000)) / m == F(72, 485)                                   # with the printed 0.040
    unresolved = F(479, 1200)                                                      # 6 of 485 harmful before the block
    assert 485 - 6 == 479 and F(14948, 10**5) < (tau - hi) / unresolved < F(14949, 10**5)
    print(f"PASS L4: E1: identified 35/1200, exact upper limit in ({float(lo):.6f}, {float(hi):.6f}], "
          f"continuing mass 485/1200 = {float(m):.5f}, target 1/10: largest tolerable gamma in "
          f"({float(g_lo):.6f}, {float(g_hi):.6f}); the printed 0.040 would give 72/485 = {72 / 485:.5f}; "
          f"on the unresolved mass 479/1200 it is {float((tau - hi) / unresolved):.5f}.")


def check_l5_labeled_tolerances():
    """The three labeled E1 tolerances of Appendix D.3: plug-in, joint 95%, and conservative."""
    n, tau, a_L = 1200, F(1, 10), F(1, 40)
    u_hi = F(40332, 10**6)                                     # harm 97.5% upper limit, rounded up
    m_lo, m_hi = F(432555, 10**6), F(432556, 10**6)
    assert vb.cp_upper_le_target(485, n, a_L, m_hi) and not vb.cp_upper_le_target(485, n, a_L, m_lo)
    joint = (tau - u_hi) / m_hi                                # mass and harm limits at 97.5% each, union bound
    assert joint >= F(1379, 10**4)
    plug = (tau - u_hi) / F(485, 1200)
    assert plug >= F(1476, 10**4)
    conservative = tau - u_hi                                  # continuing mass at most 1
    assert conservative >= F(596, 10**4) and conservative < F(597, 10**4)
    print(f"PASS L5: E1 labeled tolerances: plug-in {float(plug):.6f} (printed 0.1476), joint 95% with the mass "
          f"limit in ({float(m_lo):.6f}, {float(m_hi):.6f}] giving {float(joint):.6f} (printed 0.1379), "
          f"conservative {float(conservative):.6f} (printed 0.0596); each printed value is rounded down.")


def main():
    check_l1_decomposition()
    check_l2_exhaustive()
    check_l3_attaining()
    check_l4_e1_arithmetic()
    check_l5_labeled_tolerances()
    print("All 5 restoration-allowance checks PASSED.")


if __name__ == "__main__":
    main()

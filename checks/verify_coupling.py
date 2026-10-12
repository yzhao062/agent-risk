"""Exact finite checks for Proposition G.1.1 (prop:component-joint; component evidence to joint risk).

Every check builds its finite mechanism explicitly as an ordered list of named
kernels, enumerates the trajectory law with exact fractions, computes the
reported quantities from that law, asserts them, and prints one PASS line.
Claims about the whole coupling set Gamma (validity and sharpness of the
conditional Frechet interval, the indicator-map parametrization, the singleton
lemma) are checked by an exact rational two-phase simplex over Gamma, on
designed and seeded random instances with n up to 3, alphabets up to size 3,
and zero-mass states. No agent data are used; these are constructed
consistency checks.


Paper map: Proposition G.1.1 (prop:component-joint), Lemma G.3.2 (lem:p3-realize), and
Remark G.4.1 (rem:p3-part-ii) of Appendix G. Supplementary worked examples computed here: (CI) fails for pairwise
but not mutual independence (L3 = L1 xor L2: 0 against 1/8), for a direct handoff (1/2 against
1/4), and for a common effect S = L1 + L2 (0 against 1/4 at S = 1); the shared-outage example
gives both-fail 39/400 under the product rule, 11/80 under the shared outage, and [1/20, 9/40]
under paired records; and for costs uniform on {0, 1, 2} two couplings attain the event bound 2/3
while P(L1 + L2 >= 3) is 1/3 or 2/3.
"""
import random
from collections import defaultdict
from fractions import Fraction as F
from itertools import product


# ---------------------------------------------------------------------------
# Generic finite mechanism machinery
# ---------------------------------------------------------------------------

def run(kernels):
    """Enumerate a mechanism given as [(name, kernel)], in declared order.

    Each kernel maps the partial trajectory (a dict of earlier outputs) to a
    list of (value, probability) pairs. The result is the list of atoms
    (probability, trajectory) of the induced law on the finite path space.
    """
    atoms = [(F(1), {})]
    for name, kernel in kernels:
        extended = []
        for p, traj in atoms:
            branch = kernel(traj)
            assert sum(q for _, q in branch) == 1, name
            for value, q in branch:
                assert q >= 0, name
                if q:
                    new = dict(traj)
                    new[name] = value
                    extended.append((p * q, new))
        atoms = extended
    assert sum(p for p, _ in atoms) == 1
    return atoms


def law(atoms, f):
    out = defaultdict(F)
    for p, traj in atoms:
        out[f(traj)] += p
    return {k: v for k, v in out.items() if v}


def prob(atoms, event):
    return sum(p for p, traj in atoms if event(traj))


def bern(p):
    return [(1, F(p)), (0, 1 - F(p))]


def point(value):
    return [(value, F(1))]


def from_law(mass):
    return [(k, v) for k, v in mass.items()]


def state_law(atoms, s_name="S"):
    return law(atoms, lambda t: t[s_name])


def conditional_marginals(atoms, comps, s_name="S"):
    """P_S and F_i(l | s) computed from the enumerated law."""
    p_s = state_law(atoms, s_name)
    cond = {}
    for c in comps:
        joint = law(atoms, lambda t, c=c: (t[s_name], t[c]))
        cond[c] = {(s, l): v / p_s[s] for (s, l), v in joint.items()}
    return p_s, cond


def product_formula(p_s, cond, comps, supports):
    """The finite form of int prod_i F_i(dl_i | s) P_S(ds), on (S, L_1..L_n)."""
    out = defaultdict(F)
    for s, ps in p_s.items():
        for ls in product(*(supports[c] for c in comps)):
            mass = ps
            for c, l in zip(comps, ls):
                mass *= cond[c].get((s, l), F(0))
            if mass:
                out[(s,) + ls] += mass
    return dict(out)


def push(joint, h):
    out = defaultdict(F)
    for key, v in joint.items():
        out[h(*key)] += v
    return {k: v for k, v in out.items() if v}


def fplus(x):
    return x if x > 0 else F(0)


# ---------------------------------------------------------------------------
# Exact linear programming over the coupling set Gamma
# ---------------------------------------------------------------------------

def _pivot(tab, basis, r, c):
    piv = tab[r][c]
    tab[r] = [v / piv for v in tab[r]]
    for i in range(len(tab)):
        if i != r and tab[i][c] != 0:
            f = tab[i][c]
            tab[i] = [a - f * b for a, b in zip(tab[i], tab[r])]
    basis[r - 1] = c                      # row 0 is the objective row


def _simplex(tab, basis, ncols):
    """Minimize with Bland's rule (no cycling); tab[0] holds reduced costs."""
    while True:
        enter = next((j for j in range(ncols) if tab[0][j] < 0), None)
        if enter is None:
            return
        best = None
        for i in range(1, len(tab)):
            if tab[i][enter] > 0:
                ratio = tab[i][-1] / tab[i][enter]
                if best is None or ratio < best[0] or (
                        ratio == best[0] and basis[i - 1] < basis[best[1] - 1]):
                    best = (ratio, i)
        if best is None:
            raise ValueError("unbounded")
        _pivot(tab, basis, best[1], enter)


def lp_max(c, A, b):
    """Exact two-phase simplex over Fractions: max c.x subject to A x = b, x >= 0."""
    c = [F(v) for v in c]
    A = [[F(v) for v in row] for row in A]
    b = [F(v) for v in b]
    m, n = len(A), len(c)
    for i in range(m):
        if b[i] < 0:
            A[i], b[i] = [-v for v in A[i]], -b[i]
    ncols = n + m                         # phase 1 with artificials n..n+m-1
    tab = [[F(0)] * (ncols + 1)]
    for i in range(m):
        tab.append(A[i] + [F(1) if k == i else F(0) for k in range(m)] + [b[i]])
    basis = list(range(n, n + m))
    tab[0] = [F(0)] * n + [F(1)] * m + [F(0)]
    for i in range(1, m + 1):
        tab[0] = [a - v for a, v in zip(tab[0], tab[i])]
    _simplex(tab, basis, ncols)
    if tab[0][-1] != 0:
        raise ValueError("infeasible")
    for r in range(1, m + 1):             # drive artificials out where possible
        if basis[r - 1] >= n:
            for j in range(n):
                if tab[r][j] != 0:
                    _pivot(tab, basis, r, j)
                    break
    keep = [0] + [r for r in range(1, m + 1) if basis[r - 1] < n]
    tab2 = [tab[r][:n] + [tab[r][-1]] for r in keep]   # redundant rows dropped
    basis2 = [basis[r - 1] for r in keep[1:]]
    tab2[0] = [-v for v in c] + [F(0)]
    for i, bj in enumerate(basis2):
        if tab2[0][bj] != 0:
            f = tab2[0][bj]
            tab2[0] = [a - f * v for a, v in zip(tab2[0], tab2[i + 1])]
    _simplex(tab2, basis2, n)
    x = [F(0)] * n
    for i, bj in enumerate(basis2):
        x[bj] = tab2[i + 1][-1]
    return sum(ci * xi for ci, xi in zip(c, x))


def lp_min(c, A, b):
    return -lp_max([-F(ci) for ci in c], A, b)


def gamma_constraints(p_s, fs, sizes):
    """Gamma(P_S; F_1..F_n) as A x = b, x >= 0 over cells (s, l_1, ..., l_n).

    p_s[s] is P_S(s); fs[i][s][l] is F_i(l | s). Rows: one per (s, i, l),
    sum over cells with S = s and L_i = l equals P_S(s) F_i(l | s).
    """
    cells = [(s,) + ls for s in range(len(p_s))
             for ls in product(*(range(k) for k in sizes))]
    A, b = [], []
    for s in range(len(p_s)):
        for i, k in enumerate(sizes):
            for l in range(k):
                A.append([F(1) if (c[0] == s and c[1 + i] == l) else F(0) for c in cells])
                b.append(p_s[s] * fs[i][s][l])
    return cells, A, b


def frechet_closed_form(p_s, fs, events):
    """[bar beta^-, bar beta^+] for E_i = {L_i in events[i]}, i = 1, 2."""
    p = [[sum(fs[i][s][l] for l in events[i]) for s in range(len(p_s))] for i in (0, 1)]
    lo = sum(p_s[s] * fplus(p[0][s] + p[1][s] - 1) for s in range(len(p_s)))
    hi = sum(p_s[s] * min(p[0][s], p[1][s]) for s in range(len(p_s)))
    return lo, hi, p


def rand_law(k, rng, allow_point=True):
    if allow_point and rng.random() < 0.3:
        j = rng.randrange(k)
        return [F(1) if i == j else F(0) for i in range(k)]
    w = [rng.randint(0, 4) for _ in range(k)]
    if sum(w) == 0:
        w[rng.randrange(k)] = 1
    return [F(x, sum(w)) for x in w]


def random_instance(rng):
    """Random finite instance: n in {2, 3}, alphabets 1..3, possibly a zero-mass state."""
    n = rng.choice([2, 2, 3])
    n_s = rng.choice([1, 2, 3])
    sizes = [rng.choice([1, 2, 3]) for _ in range(n)]
    p_s = rand_law(n_s, rng, allow_point=False)
    if n_s >= 2 and rng.random() < 0.3:
        p_s = [F(0)] + rand_law(n_s - 1, rng, allow_point=False)
    fs = [[rand_law(k, rng) for _ in range(n_s)] for k in sizes]
    events = [set(x for x in range(k) if rng.random() < 0.5) for k in sizes]
    return n, sizes, p_s, fs, events


# ---------------------------------------------------------------------------
# Running illustration: two redundant tool paths with a shared outage
# ---------------------------------------------------------------------------

P_STATUS = {0: F(3, 4), 1: F(1, 4)}      # logged provider status flag S
OUTAGE = {0: F(0), 1: F(1, 2)}           # unlogged shared outage W given S
FAULT = {0: F(1, 10), 1: F(1, 5)}        # private path faults X_i given S


def h_redundant(s, l1, l2):
    """Joint loss map: (goal failure = both paths fail, number of failed path attempts).

    Both paths are attempted on every episode (parallel attempts), so L1 and
    L2 are defined on every trajectory.
    """
    return (l1 * l2, l1 + l2)


def outage_mechanism():
    return [
        ("S", lambda t: from_law(P_STATUS)),
        ("W", lambda t: bern(OUTAGE[t["S"]])),
        ("X1", lambda t: bern(FAULT[t["S"]])),
        ("X2", lambda t: bern(FAULT[t["S"]])),
        ("L1", lambda t: point(max(t["W"], t["X1"]))),
        ("L2", lambda t: point(max(t["W"], t["X2"]))),
    ]


def block_coupling(mu, nu, b1, b2, t):
    """Lemma (block construction): a coupling of mu and nu with mass t on b1 x b2."""
    a, b = sum(mu[x] for x in b1), sum(nu[y] for y in b2)
    assert fplus(a + b - 1) <= t <= min(a, b)
    table = {(1, 1): t, (1, 0): a - t, (0, 1): b - t, (0, 0): 1 - a - b + t}
    row = {1: a, 0: 1 - a}
    col = {1: b, 0: 1 - b}
    pi = defaultdict(F)
    for (j, k), tjk in table.items():
        assert tjk >= 0
        if tjk == 0:
            continue
        for x, mx in mu.items():
            if (x in b1) != bool(j):
                continue
            for y, ny in nu.items():
                if (y in b2) != bool(k):
                    continue
                pi[(x, y)] += tjk * (mx / row[j]) * (ny / col[k])
    return {k: v for k, v in pi.items() if v}


def coupling_mechanism(p_s, cond, t_of_s, b1=frozenset({1}), b2=frozenset({1})):
    """Mechanism in the unlogged-shared-input class realizing a chosen coupling.

    S is drawn from P_S, an unlogged input V is drawn from the block coupling
    at S, and component i outputs coordinate i of V.
    """
    def marg(c, s):
        return {l: v for (s2, l), v in cond[c].items() if s2 == s}

    def draw_v(traj):
        s = traj["S"]
        pi = block_coupling(marg("L1", s), marg("L2", s), b1, b2, t_of_s(s))
        return from_law(pi)

    return [
        ("S", lambda t: from_law(p_s)),
        ("V", draw_v),
        ("L1", lambda t: point(t["V"][0])),
        ("L2", lambda t: point(t["V"][1])),
    ]


def ci_mechanism(p_s, cond):
    """Mechanism with components drawn independently given S from F_i(. | S)."""
    def marg(c, s):
        return {l: v for (s2, l), v in cond[c].items() if s2 == s}
    return [
        ("S", lambda t: from_law(p_s)),
        ("L1", lambda t: from_law(marg("L1", t["S"]))),
        ("L2", lambda t: from_law(marg("L2", t["S"]))),
    ]


def fail_both(atoms):
    return prob(atoms, lambda t: h_redundant(t["S"], t["L1"], t["L2"])[0] == 1)


def system_law(atoms):
    return law(atoms, lambda t: h_redundant(t["S"], t["L1"], t["L2"]))


# ---------------------------------------------------------------------------
# P3 check 1: conditional product
# ---------------------------------------------------------------------------

def check_conditional_product():
    # (a) Three components that are mutually independent given a three-valued
    # shared state; the product formula pushed through h equals the enumerated
    # law of the joint loss.
    p_s = {0: F(1, 2), 1: F(1, 3), 2: F(1, 6)}
    rates = {"L1": {0: F(1, 10), 1: F(1, 3), 2: F(3, 4)},
             "L2": {0: F(1, 5), 1: F(1, 2), 2: F(2, 3)},
             "L3": {0: F(0), 1: F(1, 4), 2: F(1)}}
    mech = [("S", lambda t: from_law(p_s))] + [
        (c, lambda t, c=c: bern(rates[c][t["S"]])) for c in ("L1", "L2", "L3")]
    atoms = run(mech)
    comps = ("L1", "L2", "L3")
    ps_hat, cond = conditional_marginals(atoms, comps)
    joint = law(atoms, lambda t: (t["S"], t["L1"], t["L2"], t["L3"]))
    # Mutual conditional independence given S, verified on the enumerated law.
    for s in ps_hat:
        for ls in product((0, 1), repeat=3):
            lhs = joint.get((s,) + ls, F(0)) / ps_hat[s]
            rhs = F(1)
            for c, l in zip(comps, ls):
                rhs *= cond[c].get((s, l), F(0))
            assert lhs == rhs
    prod = product_formula(ps_hat, cond, comps, {c: (0, 1) for c in comps})
    assert prod == joint

    def h3(s, a, b, c):
        return (max(a, b, c), a + b + c + s)
    assert push(prod, h3) == law(atoms, lambda t: h3(t["S"], t["L1"], t["L2"], t["L3"]))

    # (b) Shared outage, logged status flag only: CI given S fails and the
    # product formula misstates the both-fail probability.
    out = run(outage_mechanism())
    ps2, cond2 = conditional_marginals(out, ("L1", "L2"))
    prod2 = product_formula(ps2, cond2, ("L1", "L2"), {"L1": (0, 1), "L2": (0, 1)})
    product_fail = push(prod2, h_redundant).get((1, 2), F(0))
    true_fail = fail_both(out)
    assert product_fail == F(39, 400) and true_fail == F(11, 80)

    # (c) Logging the outage with the status, S' = (S, W), restores CI in this
    # mechanism, and the product formula then gives the true system law.
    for p, t in out:
        t["SW"] = (t["S"], t["W"])
    ps3, cond3 = conditional_marginals(out, ("L1", "L2"), s_name="SW")
    prod3 = product_formula(ps3, cond3, ("L1", "L2"), {"L1": (0, 1), "L2": (0, 1)})
    assert push(prod3, h_redundant) == system_law(out)

    # (d) Pairwise conditional independence is not enough for n = 3.
    xor = run([("S", lambda t: point(0)),
               ("L1", lambda t: bern(F(1, 2))),
               ("L2", lambda t: bern(F(1, 2))),
               ("L3", lambda t: point(t["L1"] ^ t["L2"]))])
    for c1, c2 in (("L1", "L2"), ("L1", "L3"), ("L2", "L3")):
        pair = law(xor, lambda t: (t[c1], t[c2]))
        assert all(pair.get(k, F(0)) == F(1, 4) for k in product((0, 1), repeat=2))
    ps4, cond4 = conditional_marginals(xor, ("L1", "L2", "L3"))
    prod4 = product_formula(ps4, cond4, ("L1", "L2", "L3"), {c: (0, 1) for c in ("L1", "L2", "L3")})
    all_fail = prob(xor, lambda t: t["L1"] == t["L2"] == t["L3"] == 1)
    assert prod4.get((0, 1, 1, 1), F(0)) == F(1, 8) and all_fail == 0

    def ci_gap(atoms, s_name="S", given=lambda t: True):
        """(P(L1=L2=1 | S=s), F_1(1|s) F_2(1|s)) for each s, within the event `given`."""
        sub = [(p, t) for p, t in atoms if given(t)]
        mass = sum(p for p, _ in sub)
        sub = [(p / mass, t) for p, t in sub]
        out = {}
        for s in state_law(sub, s_name):
            ps = prob(sub, lambda t: t[s_name] == s)
            both = prob(sub, lambda t: t[s_name] == s and t["L1"] == 1 and t["L2"] == 1) / ps
            f1 = prob(sub, lambda t: t[s_name] == s and t["L1"] == 1) / ps
            f2 = prob(sub, lambda t: t[s_name] == s and t["L2"] == 1) / ps
            out[s] = (both, f1 * f2)
        return out

    # (e) No common cause: component 2 reads component 1's output through an
    # unlogged handoff, with S constant. (CI) fails: 1/2 versus 1/4.
    handoff = run([("S", lambda t: point(0)),
                   ("L1", lambda t: bern(F(1, 2))),
                   ("L2", lambda t: point(t["L1"]))])
    assert ci_gap(handoff) == {0: (F(1, 2), F(1, 4))}

    # (f) Common effect: independent losses, S = L1 + L2 (budget spent by
    # episode end). Independent unconditionally, dependent given S = 1.
    effect = run([("L1", lambda t: bern(F(1, 2))),
                  ("L2", lambda t: bern(F(1, 2))),
                  ("S", lambda t: point(t["L1"] + t["L2"]))])
    assert law(effect, lambda t: (t["L1"], t["L2"])) == {k: F(1, 4) for k in product((0, 1), repeat=2)}
    assert ci_gap(effect)[1] == (F(0), F(1, 4))

    # (g) Conditioning remark, both directions. With S constant, R and L1
    # independent fair bits and L2 = R xor L1, (CI) holds unconditionally but
    # fails given R = 1. With L1 = L2 = R, (CI) holds given each R = r and
    # fails unconditionally.
    xor_r = run([("S", lambda t: point(0)), ("R", lambda t: bern(F(1, 2))),
                 ("L1", lambda t: bern(F(1, 2))), ("L2", lambda t: point(t["R"] ^ t["L1"]))])
    assert ci_gap(xor_r) == {0: (F(1, 4), F(1, 4))}
    assert ci_gap(xor_r, given=lambda t: t["R"] == 1) == {0: (F(0), F(1, 4))}
    copy_r = run([("S", lambda t: point(0)), ("R", lambda t: bern(F(1, 2))),
                  ("L1", lambda t: point(t["R"])), ("L2", lambda t: point(t["R"]))])
    for r in (0, 1):
        both, prod_r = ci_gap(copy_r, given=lambda t, r=r: t["R"] == r)[0]
        assert both == prod_r
    assert ci_gap(copy_r) == {0: (F(1, 2), F(1, 4))}
    print("PASS P3 conditional product: mutual CI given S gives the product law through h "
          "(3 components, 3 states); with only the status logged the product rule gives "
          "both-fail 39/400 versus true 11/80; logging (S,W) restores it; "
          "pairwise CI fails (1/8 versus 0); a direct handoff (1/2 vs 1/4) and a common-effect "
          "state S=L1+L2 (0 vs 1/4 at S=1) break CI; CI given R=r and unconditional CI "
          "do not imply each other.")


# ---------------------------------------------------------------------------
# P3 check 2: attained conditional Frechet bounds
# ---------------------------------------------------------------------------

def check_frechet():
    out = run(outage_mechanism())
    p_s, cond = conditional_marginals(out, ("L1", "L2"))
    p1 = {s: cond["L1"].get((s, 1), F(0)) for s in p_s}
    p2 = {s: cond["L2"].get((s, 1), F(0)) for s in p_s}
    lo = {s: fplus(p1[s] + p2[s] - 1) for s in p_s}
    hi = {s: min(p1[s], p2[s]) for s in p_s}
    lower = sum(p_s[s] * lo[s] for s in p_s)
    upper = sum(p_s[s] * hi[s] for s in p_s)
    assert (lower, upper) == (F(1, 20), F(9, 40))
    assert (lo[0], hi[0], lo[1], hi[1]) == (F(0), F(1, 10), F(1, 5), F(3, 5))  # degraded: [1/5, 3/5]

    # Validity and sharpness over the whole coupling set, by exact LP. The
    # extrema of Lambda(E1 cap E2) over Gamma(P_S; F_1, ..., F_n) equal the
    # closed form [bar beta^-, bar beta^+]. Designed instance: three states,
    # one with zero mass (its F_i would move the bound if wrongly included),
    # three components with alphabets of sizes 3, 3, 2, and non-singleton B_1.
    d_ps = [F(1, 2), F(0), F(1, 2)]
    d_fs = [[[F(1, 2), F(1, 3), F(1, 6)], [F(0), F(0), F(1)], [F(1, 5), F(2, 5), F(2, 5)]],
            [[F(1, 5), F(2, 5), F(2, 5)], [F(0), F(0), F(1)], [F(1, 3), F(1, 3), F(1, 3)]],
            [[F(1, 4), F(3, 4)], [F(1), F(0)], [F(1, 2), F(1, 2)]]]
    d_sizes, d_events = [3, 3, 2], [{1, 2}, {2}]
    cells, A, b = gamma_constraints(d_ps, d_fs, d_sizes)
    obj = [F(1) if (c[1] in d_events[0] and c[2] in d_events[1]) else F(0) for c in cells]
    d_lo, d_hi, _ = frechet_closed_form(d_ps, d_fs, d_events)
    assert (lp_min(obj, A, b), lp_max(obj, A, b)) == (d_lo, d_hi) == (F(1, 15), F(11, 30))
    # Including the zero-mass state with weight 1/3 would give different bounds.
    wrong_lo, wrong_hi, _ = frechet_closed_form([F(1, 3)] * 3, d_fs, d_events)
    assert (wrong_lo, wrong_hi) != (d_lo, d_hi)
    # Seeded random instances (n in {2, 3}, alphabets 1..3, zero-mass states).
    rng = random.Random(1)
    for _ in range(150):
        n, sizes, ps_r, fs_r, ev = random_instance(rng)
        cells, A, b = gamma_constraints(ps_r, fs_r, sizes)
        obj = [F(1) if (c[1] in ev[0] and c[2] in ev[1]) else F(0) for c in cells]
        lo_r, hi_r, _ = frechet_closed_form(ps_r, fs_r, ev)
        assert (lp_min(obj, A, b), lp_max(obj, A, b)) == (lo_r, hi_r)

    # Attainment: explicit mechanisms with an unlogged shared input realize
    # the endpoints and the midpoint, with the same P_S and F_i as the outage
    # mechanism.
    values = []
    for lam in (F(0), F(1, 2), F(1)):
        mech = coupling_mechanism(p_s, cond, lambda s, lam=lam: (1 - lam) * lo[s] + lam * hi[s])
        atoms = run(mech)
        ps_m, cond_m = conditional_marginals(atoms, ("L1", "L2"))
        assert ps_m == p_s and cond_m == cond
        values.append(prob(atoms, lambda t: t["L1"] == 1 and t["L2"] == 1))
    assert values == [lower, (lower + upper) / 2, upper]

    # Block construction on non-binary marginals and non-singleton events.
    mu = {0: F(1, 2), 1: F(1, 3), 2: F(1, 6)}
    nu = {0: F(1, 5), 1: F(2, 5), 2: F(2, 5)}
    b1, b2 = frozenset({1, 2}), frozenset({2})
    a, b = F(1, 2), F(2, 5)
    for t in (fplus(a + b - 1), F(1, 5), min(a, b)):
        pi = block_coupling(mu, nu, b1, b2, t)
        assert {x: sum(v for (x2, _), v in pi.items() if x2 == x) for x in mu} == mu
        assert {y: sum(v for (_, y2), v in pi.items() if y2 == y) for y in nu} == nu
        assert sum(v for (x, y), v in pi.items() if x in b1 and y in b2) == t

    # Conditioning on the logged S tightens the unconditional Frechet bounds.
    e1 = prob(out, lambda t: t["L1"] == 1)
    e2 = prob(out, lambda t: t["L2"] == 1)
    unconditional = (fplus(e1 + e2 - 1), min(e1, e2))
    assert unconditional == (F(0), F(9, 40)) and lower > unconditional[0]
    print("PASS P3 attained conditional Frechet bounds: both-fail range [1/20, 9/40] "
          "attained at both endpoints and the midpoint by explicit shared-input mechanisms "
          "with identical P_S and F_i; exact LP over Gamma equals the closed form on a "
          "designed instance (n=3, 3-valued marginals, zero-mass state: [1/15, 11/30]) and "
          "150 seeded random instances; block construction verified on 3-valued marginals; "
          "unconditional bounds [0, 9/40] are wider.")


# ---------------------------------------------------------------------------
# P3 check 3: equal marginals, different system risk
# ---------------------------------------------------------------------------

def check_equal_marginals():
    out = run(outage_mechanism())
    p_s, cond = conditional_marginals(out, ("L1", "L2"))
    p1 = {s: cond["L1"].get((s, 1), F(0)) for s in p_s}
    p2 = {s: cond["L2"].get((s, 1), F(0)) for s in p_s}
    lo = {s: fplus(p1[s] + p2[s] - 1) for s in p_s}
    hi = {s: min(p1[s], p2[s]) for s in p_s}
    mechanisms = {
        "lower": run(coupling_mechanism(p_s, cond, lambda s: lo[s])),
        "product": run(ci_mechanism(p_s, cond)),
        "outage": out,
        "upper": run(coupling_mechanism(p_s, cond, lambda s: hi[s])),
    }
    risks = {}
    m = sum(p_s[s] * (p1[s] + p2[s]) for s in p_s)
    for name, atoms in mechanisms.items():
        ps_m, cond_m = conditional_marginals(atoms, ("L1", "L2"))
        assert ps_m == p_s and cond_m == cond
        T = fail_both(atoms)
        risks[name] = T
        # Coupling set for h = (L1*L2, L1+L2): a segment indexed by bar beta = T.
        segment = {(1, 2): T, (0, 1): m - 2 * T, (0, 0): 1 - m + T}
        assert system_law(atoms) == {k: v for k, v in segment.items() if v}
        failed_attempts = sum(p * (t["L1"] + t["L2"]) for p, t in atoms)
        assert failed_attempts == m == F(9, 20)
        assert prob(atoms, lambda t: t["L1"] + t["L2"] >= 1) == m - T
    assert risks == {"lower": F(1, 20), "product": F(39, 400),
                     "outage": F(11, 80), "upper": F(9, 40)}
    # P(at least one failed attempt) = 9/20 - bar beta ranges over [9/40, 2/5].
    assert (m - risks["upper"], m - risks["lower"]) == (F(9, 40), F(2, 5))

    # Part (ii)(d), indicator loss maps: for n = 2 and h = hbar(s, 1_B1, 1_B2),
    # the range of every linear query over Gamma (exact LP) equals the range
    # over the per-state box beta(s) in [beta^-(s), beta^+(s)].
    rng = random.Random(2)
    for _ in range(60):
        n_s = rng.choice([1, 2, 3])
        sizes = [rng.choice([2, 3]), rng.choice([2, 3])]
        ps_r = rand_law(n_s, rng, allow_point=False)
        fs_r = [[rand_law(k, rng) for _ in range(n_s)] for k in sizes]
        ev = [{0} | set(x for x in range(1, k) if rng.random() < 0.5) for k in sizes]
        dep_s = rng.random() < 0.5
        vals = {(s, j, k): F(rng.randint(-3, 3)) for s in range(n_s) for j in (0, 1) for k in (0, 1)}
        hbar = (lambda s, j, k: vals[(s, j, k)]) if dep_s else (lambda s, j, k: vals[(0, j, k)])
        cells, A, b = gamma_constraints(ps_r, fs_r, sizes)
        obj = [hbar(c[0], int(c[1] in ev[0]), int(c[2] in ev[1])) for c in cells]
        _, _, p = frechet_closed_form(ps_r, fs_r, ev)
        box_lo = box_hi = F(0)
        for s in range(n_s):
            def val(beta, s=s):
                n_jk = {(1, 1): beta, (1, 0): p[0][s] - beta, (0, 1): p[1][s] - beta,
                        (0, 0): 1 - p[0][s] - p[1][s] + beta}
                return sum(v * hbar(s, *jk) for jk, v in n_jk.items())
            ends = (val(fplus(p[0][s] + p[1][s] - 1)), val(min(p[0][s], p[1][s])))
            box_lo += ps_r[s] * min(ends)
            box_hi += ps_r[s] * max(ends)
        assert (lp_min(obj, A, b), lp_max(obj, A, b)) == (box_lo, box_hi)

    # Event bounds do not settle a non-indicator loss map: two couplings of
    # uniform three-valued costs both attain the upper bound for the events
    # {L_i >= 1}, yet P(L1 + L2 >= 3) differs.
    third = F(1, 3)
    diag = run([("S", lambda t: point(0)),
                ("V", lambda t: [((0, 0), third), ((1, 1), third), ((2, 2), third)]),
                ("L1", lambda t: point(t["V"][0])), ("L2", lambda t: point(t["V"][1]))])
    swap = run([("S", lambda t: point(0)),
                ("V", lambda t: [((0, 0), third), ((1, 2), third), ((2, 1), third)]),
                ("L1", lambda t: point(t["V"][0])), ("L2", lambda t: point(t["V"][1]))])
    for atoms in (diag, swap):
        assert law(atoms, lambda t: t["L1"]) == {0: third, 1: third, 2: third}
        assert law(atoms, lambda t: t["L2"]) == {0: third, 1: third, 2: third}
        assert prob(atoms, lambda t: t["L1"] >= 1 and t["L2"] >= 1) == F(2, 3)
        assert sum(p * (t["L1"] + t["L2"]) for p, t in atoms) == 2
    tail = [prob(a, lambda t: t["L1"] + t["L2"] >= 3) for a in (diag, swap)]
    assert tail == [F(1, 3), F(2, 3)]
    # Exact LP: among couplings at the upper event bound 2/3, P(L1 + L2 >= 3)
    # ranges over exactly [1/3, 2/3].
    cells, A, b = gamma_constraints([F(1)], [[[third] * 3], [[third] * 3]], [3, 3])
    at_bound = [F(1) if (c[1] >= 1 and c[2] >= 1) else F(0) for c in cells]
    obj = [F(1) if c[1] + c[2] >= 3 else F(0) for c in cells]
    A2, b2 = A + [at_bound], b + [F(2, 3)]
    assert (lp_min(obj, A2, b2), lp_max(obj, A2, b2)) == (F(1, 3), F(2, 3))
    print("PASS P3 equal marginals, different system risk: four mechanisms with identical "
          "P_S and F_i give both-fail 1/20, 39/400, 11/80, 9/40 and identical mean failed "
          "attempts 9/20; system law lies on the segment indexed by bar beta, with "
          "P(at least one failed attempt) in [9/40, 2/5]; for indicator loss maps the exact "
          "LP range of linear queries over Gamma equals the per-state box (60 seeded "
          "instances); for a 3-valued sum, couplings at the same upper event bound give "
          "P(sum>=3) over exactly [1/3, 2/3].")


# ---------------------------------------------------------------------------
# P3 check 4: linked records identify; S alone (even paired) does not
# ---------------------------------------------------------------------------

def with_component_selector(mech):
    """Append a passive logging kernel J, uniform on {1, 2}, independent of execution."""
    return mech + [("J", lambda t: [(1, F(1, 2)), (2, F(1, 2))])]


def check_linked_records():
    out_mech = outage_mechanism()
    out = run(out_mech)
    p_s, cond = conditional_marginals(out, ("L1", "L2"))
    ci_mech = ci_mechanism(p_s, cond)
    pair_mechs = {"outage": with_component_selector(out_mech),
                  "product": with_component_selector(ci_mech)}
    atoms = {k: run(v) for k, v in pair_mechs.items()}

    channels = {
        "S only": lambda t: t["S"],
        "(S, L1)": lambda t: (t["S"], t["L1"]),
        "(S, L2)": lambda t: (t["S"], t["L2"]),
        "(J, S, L_J)": lambda t: (t["J"], t["S"], t["L%d" % t["J"]]),
    }
    for name, g in channels.items():
        assert law(atoms["outage"], g) == law(atoms["product"], g), name
    link = lambda t: (t["S"], t["L1"], t["L2"])
    rec_out, rec_ci = law(atoms["outage"], link), law(atoms["product"], link)
    assert rec_out != rec_ci

    def both_fail_from_linked_record(record_law):
        # Identification map: Lambda is the linked record law itself.
        return sum(v for (s, l1, l2), v in record_law.items() if h_redundant(s, l1, l2)[0] == 1)

    for name in atoms:
        rec = law(atoms[name], link)
        assert both_fail_from_linked_record(rec) == fail_both(atoms[name])
    assert both_fail_from_linked_record(rec_out) == F(11, 80)
    assert both_fail_from_linked_record(rec_ci) == F(39, 400)

    # The extreme couplings share the same paired evidence, so the identified
    # set under paired records is the full interval [1/20, 9/40].
    p1 = {s: cond["L1"].get((s, 1), F(0)) for s in p_s}
    p2 = {s: cond["L2"].get((s, 1), F(0)) for s in p_s}
    ends = []
    for t_of_s in (lambda s: fplus(p1[s] + p2[s] - 1), lambda s: min(p1[s], p2[s])):
        a = run(with_component_selector(coupling_mechanism(p_s, cond, t_of_s)))
        assert law(a, channels["(J, S, L_J)"]) == law(atoms["outage"], channels["(J, S, L_J)"])
        ends.append(both_fail_from_linked_record(law(a, link)))
    assert ends == [F(1, 20), F(9, 40)]

    # Precision of "S alone does not": if one component is a point mass given
    # each S, every coupling is forced and paired records do identify the joint
    # law. Witness: L1 = S (component 1 fails exactly in the degraded state).
    deg_a = run([("S", lambda t: from_law(P_STATUS)), ("W", lambda t: bern(OUTAGE[t["S"]])),
                 ("L1", lambda t: point(t["S"])),
                 ("L2", lambda t: bern(F(1, 10)) if t["W"] == 0 and t["S"] == 0
                  else (bern(F(1, 5)) if t["W"] == 0 else point(1)))])
    ps_d, cond_d = conditional_marginals(deg_a, ("L1", "L2"))
    deg_b = run(ci_mechanism(ps_d, cond_d))
    for s in ps_d:
        q1, q2 = cond_d["L1"].get((s, 1), F(0)), cond_d["L2"].get((s, 1), F(0))
        assert q1 in (0, 1) and fplus(q1 + q2 - 1) == min(q1, q2)
    assert conditional_marginals(deg_b, ("L1", "L2")) == (ps_d, cond_d)
    assert law(deg_a, link) == law(deg_b, link)

    # Part (iii)(d) / Lemma p3-singleton by exact LP: Gamma is a singleton
    # (every cell has equal LP minimum and maximum) iff at each s with
    # P_S(s) > 0 at most one F_i(. | s) is not a point mass.
    rng = random.Random(1)
    n_single = n_multi = 0
    for _ in range(150):
        n, sizes, ps_r, fs_r, _ = random_instance(rng)
        cells, A, b = gamma_constraints(ps_r, fs_r, sizes)
        single = True
        for j in range(len(cells)):
            e = [F(1) if k == j else F(0) for k in range(len(cells))]
            if lp_max(e, A, b) != lp_min(e, A, b):
                single = False
                break
        condition = all(sum(1 for i in range(n) if max(fs_r[i][s]) < 1) <= 1
                        for s in range(len(ps_r)) if ps_r[s] > 0)
        assert single == condition
        n_single += single
        n_multi += not single
    assert n_single and n_multi

    # Sequential fallback (a different scheduler): the second path runs only
    # after the first fails, so its own log, without an episode identifier,
    # identifies the both-fail probability 11/80; no Frechet interval arises.
    seq = run([("S", lambda t: from_law(P_STATUS)),
               ("W", lambda t: bern(OUTAGE[t["S"]])),
               ("X1", lambda t: bern(FAULT[t["S"]])),
               ("L1", lambda t: point(max(t["W"], t["X1"]))),
               ("X2", lambda t: bern(FAULT[t["S"]]) if t["L1"] == 1 else point(None)),
               ("L2", lambda t: point(max(t["W"], t["X2"])) if t["L1"] == 1 else point(None))])
    fallback_log = law(seq, lambda t: (t["S"], t["L2"]))       # (S, None) when not run
    from_log = sum(v for (s, l2), v in fallback_log.items() if l2 == 1)
    assert from_log == prob(seq, lambda t: t["L1"] == 1 and t["L2"] == 1) == F(11, 80)
    print("PASS P3 linked S alone is insufficient: outage and product mechanisms agree on "
          "S-only, (S,L1), (S,L2), and (J,S,L_J) record laws but differ on linked (S,L1,L2); "
          "the linked record law returns both-fail 11/80 and 39/400 exactly; paired evidence "
          "leaves [1/20, 9/40]; degenerate-component case is identified (forced coupling); "
          "exact-LP singleton test matches the point-mass condition (%d singleton, %d not); "
          "under a sequential fallback the fallback log alone gives 11/80." % (n_single, n_multi))


# ---------------------------------------------------------------------------
# P3 check 5: standalone-to-composition transfer failure
# ---------------------------------------------------------------------------

def check_transfer():
    def component2(g):
        # Component 2 kernel: reads only the remaining budget S.
        return lambda t: bern(g[t["S"]])

    def composition(g):
        # Sequential scheduler: component 1 spends C1 paid calls from a shared
        # budget of 2, then component 2 starts with S = 2 - C1.
        return run([("C1", lambda t: bern(F(1, 2))),
                    ("S", lambda t: point(2 - t["C1"])),
                    ("L2", component2(g))])

    def standalone(g, start=None):
        start = start or {2: F(1)}
        return run([("S", lambda t: from_law(start)), ("L2", component2(g))])

    g_a = {2: F(1, 10), 1: F(1, 10)}   # budget-insensitive
    g_b = {2: F(1, 10), 1: F(1)}       # needs two calls; fails on one
    sa_a, sa_b = standalone(g_a), standalone(g_b)
    rec = lambda t: (t["S"], t["L2"])
    assert law(sa_a, rec) == law(sa_b, rec)                       # same standalone evidence
    comp_a, comp_b = composition(g_a), composition(g_b)
    p_s = state_law(comp_a)
    assert p_s == state_law(comp_b) == {1: F(1, 2), 2: F(1, 2)}
    q_s = state_law(sa_a)
    unseen = [s for s in p_s if s not in q_s]
    assert unseen == [1]                                           # (Supp) fails
    for g, comp, sa in ((g_a, comp_a, sa_a), (g_b, comp_b, sa_b)):  # (Inv) holds
        _, c_comp = conditional_marginals(comp, ("L2",))
        _, c_sa = conditional_marginals(sa, ("L2",))
        assert all(v == (g[s] if l == 1 else 1 - g[s]) for (s, l), v in c_comp["L2"].items())
        assert all(v == (g[s] if l == 1 else 1 - g[s]) for (s, l), v in c_sa["L2"].items())
    risk_a = prob(comp_a, lambda t: t["L2"] == 1)
    risk_b = prob(comp_b, lambda t: t["L2"] == 1)
    naive = prob(sa_a, lambda t: t["L2"] == 1)
    assert (risk_a, risk_b, naive) == (F(1, 10), F(11, 20), F(1, 10))

    # Identified set: the kernel at the unseen state is unrestricted.
    sweep = [prob(composition({2: F(1, 10), 1: F(k, 4)}), lambda t: t["L2"] == 1)
             for k in range(5)]
    c_b = sum(p_s[s] * F(1, 10) for s in p_s if s in q_s)
    width = sum(p_s[s] for s in unseen)
    assert (min(sweep), max(sweep)) == (c_b, c_b + width) == (F(1, 20), F(11, 20))

    # Positive control: standalone runs that also start at budget 1 satisfy
    # (Supp); the evidence then separates g_a and g_b, and transfer is exact.
    start = {1: F(1, 2), 2: F(1, 2)}
    for g, truth in ((g_a, risk_a), (g_b, risk_b)):
        sa = standalone(g, start)
        q = law(sa, rec)
        q_s2 = state_law(sa)
        transfer = sum(p_s[s] * q.get((s, 1), F(0)) / q_s2[s] for s in p_s)
        assert transfer == truth
    assert law(standalone(g_a, start), rec) != law(standalone(g_b, start), rec)

    # Invariance failure: an unlogged stale-cache flag Z written by component 1
    # changes component 2 at the same logged S; transfer then misstates risk.
    def comp2_cache(t):
        return bern(F(1, 10) if t["Z"] == 0 else F(1))
    cache_comp = run([("S", lambda t: point(2)), ("Z", lambda t: bern(F(1, 2))),
                      ("L2", comp2_cache)])
    cache_sa = run([("S", lambda t: point(2)), ("Z", lambda t: point(0)),
                    ("L2", comp2_cache)])
    _, f_comp = conditional_marginals(cache_comp, ("L2",))
    _, f_sa = conditional_marginals(cache_sa, ("L2",))
    assert f_comp["L2"][(2, 1)] == F(11, 20) and f_sa["L2"][(2, 1)] == F(1, 10)
    # Logging the flag in the shared state, S' = (budget, Z), restores (Inv)
    # and turns the failure into a detectable (Supp) failure.
    for atoms in (cache_comp, cache_sa):
        for p, t in atoms:
            t["SZ"] = (t["S"], t["Z"])
    _, g_comp = conditional_marginals(cache_comp, ("L2",), s_name="SZ")
    _, g_sa = conditional_marginals(cache_sa, ("L2",), s_name="SZ")
    shared = set(s for s, _ in g_comp["L2"]) & set(s for s, _ in g_sa["L2"])
    assert all(g_comp["L2"][(s, l)] == g_sa["L2"][(s, l)]
               for (s, l) in g_comp["L2"] if s in shared)
    assert set(state_law(cache_comp, "SZ")) - set(state_law(cache_sa, "SZ")) == {(2, 1)}
    print("PASS P3 standalone-to-composition transfer failure: shared budget makes S=1 "
          "(probability 1/2) unseen in isolation; identical standalone evidence and invariant "
          "kernels give composed failure 1/10 or 11/20, identified set [1/20, 11/20] of width "
          "P_S(unseen)=1/2; supported standalone runs transfer exactly; an unlogged stale cache "
          "breaks invariance (1/10 versus 11/20), and logging it exposes a support failure.")


if __name__ == "__main__":
    check_conditional_product()
    check_frechet()
    check_equal_marginals()
    check_linked_records()
    check_transfer()

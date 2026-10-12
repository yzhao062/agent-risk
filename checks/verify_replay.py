#!/usr/bin/env python
"""Exact finite checks for Propositions A.4.1 and E.1.1 (deviation replay) and Proposition 3.1 (logging sufficiency).

Every check builds an explicit finite mechanism as an ordered list of named
kernels, computes path laws by forward composition in exact rational
arithmetic, computes the replay quantities from the baseline path law and the
two known decision maps only, and asserts the claimed identity, bound, or
counterexample. Each check prints one PASS line.

Run with: python verify_replay.py


Paper map: Propositions A.4.1 (prop:replay), E.1.1 (prop:replay-cont), and E.4.3
(prop:app-identified), Proposition 3.1 (thm:logging), and Corollary E.6.1 (cor:garbling).
Supplementary checks documented in checks/README.md: the injectivity lemma (not printed;
archived in supplement/appendix-E.md), T1-14 and T1-16 (where saturation fails; T1-14 stays as
one sentence after condition (S)), T1-7 (tail sensitivity), T1-17 (restricted law classes;
witness in the scope paragraph of Proposition 3.1), T1-5 (strict logging), T1-6 and T1-15
(injectivity with one loss value and with prefix-dependent feasible sets), and T1-11
(announced gate).
"""
from fractions import Fraction as F
from itertools import product

ABORT = "ABORT"
HALF = F(1, 2)


# ---------------------------------------------------------------------------
# Generic finite-mechanism engine
# ---------------------------------------------------------------------------

def rand(name, inp, ker):
    """Random kernel: value ~ ker(inp(history))."""
    return ("rand", name, inp, ker)


def dec(name, kname, t, inp):
    """Deterministic decision kernel `kname` at step t: value = policy(t, inp(history))."""
    return ("dec", name, kname, t, inp)


def forward(steps, policies, loss_fn, post_abort=None):
    """Forward composition of the declared kernels in generation order.

    Returns {path: probability}; a path is a tuple of (name, value) pairs that
    ends with ("L", loss). An ABORT output stops generation, and the loss is
    then drawn from the declared terminal valuation post_abort(history).
    """
    out = {}

    def rec(i, hist, prob):
        h = dict(hist)
        if i == len(steps):
            path = hist + (("L", loss_fn(h)),)
            out[path] = out.get(path, F(0)) + prob
            return
        st = steps[i]
        if st[0] == "rand":
            _, name, inp, ker = st
            for v, p in ker(inp(h)).items():
                if p:
                    rec(i + 1, hist + ((name, v),), prob * F(p))
        else:
            _, name, kname, t, inp = st
            v = policies[kname](t, inp(h))
            hist2 = hist + ((name, v),)
            if v == ABORT:
                for l, p in post_abort(dict(hist2)).items():
                    if p:
                        path = hist2 + (("L", l),)
                        out[path] = out.get(path, F(0)) + prob * F(p)
            else:
                rec(i + 1, hist2, prob)

    rec(0, (), F(1))
    return out


def get(path, name):
    for n, v in path:
        if n == name:
            return v
    return None


def loss_of(path):
    return path[-1][1]


def clean(d):
    return {k: v for k, v in d.items() if v != 0}


def first_deviation(path, steps, f0, fk, kname):
    """sigma_kappa from the declared inputs and the two known maps only.

    Returns (step, status, replayed prefix h+) or (None, "agree", None).
    """
    by_name = {st[1]: st for st in steps}
    h, hist = {}, ()
    for name, v in path:
        if name == "L":
            break
        st = by_name[name]
        if st[0] == "dec" and st[2] == kname:
            inp = st[4](h)
            a0, ak = f0(st[3], inp), fk(st[3], inp)
            if a0 != ak:
                status = "terminal" if ak == ABORT else "continuing"
                return st[3], status, hist + ((name, ak),)
        hist += ((name, v),)
        h[name] = v
    return None, "agree", None


def signature(path, steps, f0, fk, kname, b):
    """Contribution signature (C_kappa, u_kappa) of one path."""
    _, status, pre = first_deviation(path, steps, f0, fk, kname)
    if status == "agree":
        return (frozenset({(loss_of(path), F(1))}), 0)
    if status == "terminal":
        return (frozenset(clean(b(dict(pre))).items()), 0)
    return (frozenset(), 1)


def law(P, cond=lambda w: True, phi=lambda l: l):
    Z = sum(p for w, p in P.items() if cond(w))
    out = {}
    for w, p in P.items():
        if cond(w):
            x = phi(loss_of(w))
            out[x] = out.get(x, F(0)) + p / Z
    return clean(out)


def replay(P0, steps, f0, fk, kname, b, cond=lambda w: True, phi=lambda l: l):
    """Full-record replay summary (nu, eps, continuing prefix weights) from P0 alone."""
    Z = sum(p for w, p in P0.items() if cond(w))
    nu, eps, cont = {}, F(0), {}
    for w, p in P0.items():
        if not cond(w):
            continue
        _, status, pre = first_deviation(w, steps, f0, fk, kname)
        if status == "agree":
            x = phi(loss_of(w))
            nu[x] = nu.get(x, F(0)) + p / Z
        elif status == "terminal":
            for l, q in b(dict(pre)).items():
                x = phi(l)
                nu[x] = nu.get(x, F(0)) + p * F(q) / Z
        else:
            eps += p / Z
            cont[pre] = cont.get(pre, F(0)) + p / Z
    return clean(nu), eps, cont


def restricted_agreement(P, steps, f0, fk, kname, cond):
    """B -> P(L in B, A_kappa | r), represented on atoms of the loss alphabet."""
    Z = sum(p for w, p in P.items() if cond(w))
    out = {}
    for w, p in P.items():
        if cond(w) and first_deviation(w, steps, f0, fk, kname)[1] == "agree":
            out[loss_of(w)] = out.get(loss_of(w), F(0)) + p / Z
    return clean(out)


def add(*laws):
    out = {}
    for d in laws:
        for k, v in d.items():
            out[k] = out.get(k, F(0)) + v
    return clean(out)


def scale(d, c):
    return clean({k: c * v for k, v in d.items()})


def mean(d):
    return sum(x * p for x, p in d.items())


def cdf(d, x):
    return sum(p for v, p in d.items() if v <= x)


def var(d, beta):
    return min(x for x in d if cdf(d, x) >= beta)


def cvar(d, beta):
    """Rockafellar-Uryasev: min_c c + E(X-c)^+/(1-beta); the minimum is at a support point."""
    return min(c + sum(p * max(x - c, 0) for x, p in d.items()) / (1 - beta) for c in d)


def fibres(paths, g):
    out = {}
    for w in paths:
        out.setdefault(g(w), []).append(w)
    return out


def push(P, g):
    out = {}
    for w, p in P.items():
        out[g(w)] = out.get(g(w), F(0)) + p
    return clean(out)


def uniform(values):
    values = list(values)
    return {v: F(1, len(values)) for v in values}


def bern(q):
    return clean({1: F(q), 0: 1 - F(q)})


# ---------------------------------------------------------------------------
# Check 1: stopping-law equality without abort support
# ---------------------------------------------------------------------------

def check_stopping_law_without_abort_support():
    def proposal0(i):
        return {"read": F(3, 4), "write": F(1, 4)} if i[0] == "low" else {"read": F(1, 4), "write": F(3, 4)}

    def proposal1(i):
        z, o0 = i
        if o0 == 1:
            return {"read": HALF, "write": HALF}
        return {"read": HALF, "write": HALF} if z == "low" else {"read": F(1, 3), "write": F(2, 3)}

    def score(i):
        z, p = i
        if p == "read":
            return {0: F(1)}
        return {1: F(4, 5), 0: F(1, 5)} if z == "high" else {1: F(1, 5), 0: F(4, 5)}

    def outcome(i):
        z, p, g = i
        if g != "EXEC" or p == "read":
            return {0: F(1)}
        return bern(HALF if z == "high" else F(1, 10))

    steps = [
        rand("Z", lambda h: (), lambda i: {"low": F(2, 3), "high": F(1, 3)}),
        rand("P0", lambda h: (h["Z"],), proposal0),
        rand("S0", lambda h: (h["Z"], h["P0"]), score),
        dec("G0", "gate", 0, lambda h: (h["P0"], h["S0"])),
        rand("O0", lambda h: (h["Z"], h["P0"], h["G0"]), outcome),
        rand("P1", lambda h: (h["Z"], h["O0"]), proposal1),
        rand("S1", lambda h: (h["Z"], h["P1"]), score),
        dec("G1", "gate", 1, lambda h: (h["P1"], h["S1"])),
        rand("O1", lambda h: (h["Z"], h["P1"], h["G1"]), outcome),
    ]

    def loss(h):
        return (h["O0"] + h["O1"], [h["G0"], h["G1"]].count("EXEC"))

    f0 = lambda t, i: "EXEC"
    fk = lambda t, i: ABORT if (i[0] == "write" and i[1] == 1) else "EXEC"

    def b(h):
        harm = h.get("O0", 0) if h.get("O0") is not None else 0
        cost = [h.get("G0"), h.get("G1")].count("EXEC")
        # Abort does not mean zero later damage: an unfinished transaction adds harm 1 w.p. 1/5.
        return {(harm, cost): F(4, 5), (harm + 1, cost): F(1, 5)}

    P0 = forward(steps, {"gate": f0}, loss, b)
    Pk = forward(steps, {"gate": fk}, loss, b)
    base_abort = sum(p for w, p in P0.items() if ABORT in [v for _, v in w])
    repl_abort = sum(p for w, p in Pk.items() if ABORT in [v for _, v in w])
    assert base_abort == 0 and repl_abort > 0
    for cond in (lambda w: True, lambda w: get(w, "Z") == "high", lambda w: get(w, "Z") == "low"):
        nu, eps, _ = replay(P0, steps, f0, fk, "gate", b, cond)
        assert eps == 0
        assert nu == law(Pk, cond)
    assert law(Pk) != law(P0)
    print("PASS T1-1 stopping law: two-step gate, baseline abort mass 0, replacement abort mass "
          f"{repl_abort}; replayed law equals forward law (unconditional and given Z), with a nonzero abort valuation.")


# ---------------------------------------------------------------------------
# Check 2: restricted-measure agreement under a seeded deterministic policy
# ---------------------------------------------------------------------------

def check_restricted_measure_seeded_policy():
    f0 = lambda t, i: (1 if i[1] == 0 else 0) if t == 0 else (i[1] + i[2]) % 2

    def fk(t, i):
        if t == 0:
            x, u = i
            return (1 if u <= 1 else 0) if x == 1 else (1 if u == 0 else 0)
        return i[1]  # ignore the seed at step 1

    envs = [
        ({(0, 0): F(1, 5), (0, 1): F(3, 5), (1, 0): HALF, (1, 1): F(1, 4)},
         {(0, 0): F(1, 3), (0, 1): F(2, 3), (1, 0): F(1, 10), (1, 1): F(9, 10)}),
        ({(0, 0): F(0), (0, 1): F(1), (1, 0): F(1, 7), (1, 1): F(5, 7)},
         {(0, 0): HALF, (0, 1): F(1, 4), (1, 0): F(3, 4), (1, 1): F(0)}),
        ({(0, 0): F(2, 9), (0, 1): F(4, 9), (1, 0): F(8, 9), (1, 1): F(1, 9)},
         {(0, 0): F(5, 6), (0, 1): F(1, 6), (1, 0): F(2, 5), (1, 1): F(3, 5)}),
    ]
    some_full_law_differs = False
    for e0, e1 in envs:
        steps = [
            rand("X", lambda h: (), lambda i: {0: HALF, 1: HALF}),
            rand("U0", lambda h: (), lambda i: uniform(range(3))),
            dec("A0", "policy", 0, lambda h: (h["X"], h["U0"])),
            rand("Y0", lambda h: (h["X"], h["A0"]), lambda i, e0=e0: bern(e0[i])),
            rand("U1", lambda h: (), lambda i: uniform(range(2))),
            dec("A1", "policy", 1, lambda h: (h["X"], h["Y0"], h["U1"])),
            rand("Y1", lambda h: (h["Y0"], h["A1"]), lambda i, e1=e1: bern(e1[i])),
        ]
        loss = lambda h: (h["Y0"] + h["Y1"], h["A0"] + h["A1"])
        P0 = forward(steps, {"policy": f0}, loss)
        Pk = forward(steps, {"policy": fk}, loss)
        for x in (0, 1):
            cond = lambda w, x=x: get(w, "X") == x
            r0 = restricted_agreement(P0, steps, f0, fk, "policy", cond)
            rk = restricted_agreement(Pk, steps, f0, fk, "policy", cond)
            assert r0 == rk and sum(r0.values()) > 0
            m = sum(r0.values())
            assert scale(r0, 1 / m) == scale(rk, 1 / sum(rk.values()))
            if law(Pk, cond) != law(P0, cond):
                some_full_law_differs = True
    assert some_full_law_differs
    print("PASS T1-2 restricted agreement: seeded deterministic policy over two steps and three environment "
          "kernels; P^k(L in B, A_k | r) = P^0(L in B, A_k | r) for every atom and record, full conditional laws differ.")


# ---------------------------------------------------------------------------
# Check 3: propensity-1/2 missing-outcome counterexample
# ---------------------------------------------------------------------------

def missing_outcome_mechanism(rule):
    steps = [
        rand("Prop", lambda h: (), lambda i: {"call": F(1)}),
        rand("U", lambda h: (), lambda i: uniform(range(4))),
        dec("G", "gate", 0, lambda h: (h["Prop"], h["U"])),
        rand("B", lambda h: (), lambda i: {0: F(1)}),  # boundary loss accrued at the gate
        rand("Y", lambda h: (h["G"],), lambda i: {rule(i[0]): F(1)}),
    ]
    loss = lambda h: h["B"] + h["Y"]
    return steps, loss


def check_missing_outcome():
    f0 = lambda t, i: "EXEC" if i[1] in (0, 1) else "SKIP"
    f_tight = lambda t, i: "EXEC" if i[1] == 0 else "SKIP"          # tightened gate, newly blocked calls skipped
    f_thm = lambda t, i: "EXEC" if i[1] == 0 else (ABORT if i[1] == 1 else "SKIP")  # newly blocked calls aborted
    b = lambda h: {0: F(1)}
    rules = {"M1": lambda g: 1 if g == "EXEC" else 0, "M2": lambda g: 1 if g == "SKIP" else 0}
    propensity = F(sum(1 for u in range(4) if f0(0, ("call", u)) == "EXEC"), 4)
    assert propensity == HALF

    def g_bdry(w):
        return (get(w, "Prop"), get(w, "U"), get(w, "G"), propensity, get(w, "B"))

    def g_full(w):
        return g_bdry(w) + (loss_of(w),)

    P0s, means_tight, means_thm = {}, {}, {}
    for m, rule in rules.items():
        steps, loss = missing_outcome_mechanism(rule)
        P0s[m] = forward(steps, {"gate": f0}, loss, b)
        means_tight[m] = mean(law(forward(steps, {"gate": f_tight}, loss, b)))
        Pk = forward(steps, {"gate": f_thm}, loss, b)
        means_thm[m] = mean(law(Pk))
        nu, eps, _ = replay(P0s[m], steps, f0, f_thm, "gate", b)
        assert eps == 0 and nu == law(Pk)
    gate_support = {get(w, "G") for w in P0s["M1"]}
    assert gate_support == {"EXEC", "SKIP"}
    assert push(P0s["M1"], g_bdry) == push(P0s["M2"], g_bdry)
    assert means_tight == {"M1": F(1, 4), "M2": F(3, 4)}
    assert means_thm == {"M1": F(1, 4), "M2": F(1, 2)}
    steps, _ = missing_outcome_mechanism(rules["M1"])
    omega0 = set(P0s["M1"]) | set(P0s["M2"])
    sig = lambda w: signature(w, steps, f0, f_thm, "gate", b)
    bad = [y for y, ws in fibres(omega0, g_bdry).items() if len({sig(w) for w in ws}) > 1]
    good = all(len({sig(w) for w in ws}) == 1 for ws in fibres(omega0, g_full).values())
    assert bad and good
    print("PASS T1-3 missing outcome: propensity 1/2 with full gate support and recorded boundary loss 0; "
          "boundary records equal under L=A and L=1-A, yet tightened-gate means are 1/4 vs 3/4 "
          "(1/4 vs 1/2 with abort valued at 0); the fibre condition fails without, and holds with, the terminal loss.")


# ---------------------------------------------------------------------------
# Incident mechanism shared by checks 4, 5, 9
# ---------------------------------------------------------------------------

def incident_mechanism(p, ql, qh):
    q = {"low": F(ql), "high": F(qh)}
    steps = [
        rand("Z", lambda h: (), lambda i: clean({"low": F(p), "high": 1 - F(p)})),
        dec("G", "gate", 0, lambda h: (h["Z"],)),
        rand("Y", lambda h: (h["Z"], h["G"]), lambda i: bern(q[i[0]]) if i[1] == "EXEC" else {0: F(1)}),
    ]
    return steps, (lambda h: h["Y"])


INC_F0 = lambda t, i: "EXEC"
INC_FK = lambda t, i: ABORT if i[0] == "high" else "EXEC"
INC_B = lambda h: {0: F(1)}  # declared: an aborted call causes no harm


def g_incident(w):
    return ("incident", get(w, "Z")) if loss_of(w) == 1 else "none"


def g_outcome(w):
    return loss_of(w)


def g_fullrec(w):
    return (get(w, "Z"), get(w, "G"), loss_of(w))


def g_const(w):
    return "const"


def incident_omega0():
    steps, loss = incident_mechanism(HALF, HALF, HALF)
    return set(forward(steps, {"gate": INC_F0}, loss, INC_B)), steps


# ---------------------------------------------------------------------------
# Check 4: equal contribution, different status, same fibre
# ---------------------------------------------------------------------------

def check_equal_contribution_different_status():
    omega0, steps = incident_omega0()
    sig = lambda w: signature(w, steps, INC_F0, INC_FK, "gate", INC_B)
    status = lambda w: first_deviation(w, steps, INC_F0, INC_FK, "gate")[1]
    fib = fibres(omega0, g_incident)
    none = fib["none"]
    assert {status(w) for w in none} == {"agree", "terminal"}
    assert len({sig(w) for w in none}) == 1
    # The tightened gate's status-plus-value map is not constant on this fibre.
    assert len({(status(w), sig(w)) for w in none}) == 2
    assert all(len({sig(w) for w in ws}) == 1 for ws in fib.values())
    sig_of = {y: dict(next(iter({sig(w) for w in ws}))[0]) for y, ws in fib.items()}
    grid = [F(0), F(1, 4), HALF, F(3, 4), F(1)]
    for p, ql, qh in product(grid, repeat=3):
        st, loss = incident_mechanism(p, ql, qh)
        P0 = forward(st, {"gate": INC_F0}, loss, INC_B)
        Pk = forward(st, {"gate": INC_FK}, loss, INC_B)
        phi_of_g = add(*[scale(sig_of[y], m) for y, m in push(P0, g_incident).items()])
        assert phi_of_g == law(Pk)
    stA, loss = incident_mechanism(HALF, HALF, 0)
    stB, _ = incident_mechanism(F(1, 4), 1, 0)
    PA, PB = forward(stA, {"gate": INC_F0}, loss, INC_B), forward(stB, {"gate": INC_F0}, loss, INC_B)
    agreeA = sum(p for w, p in PA.items() if status(w) == "agree")
    agreeB = sum(p for w, p in PB.items() if status(w) == "agree")
    assert push(PA, g_incident) == push(PB, g_incident) and agreeA != agreeB
    assert law(forward(stA, {"gate": INC_FK}, loss, INC_B)) == law(forward(stB, {"gate": INC_FK}, loss, INC_B))
    print("PASS T1-4 equal contribution: incident-only log merges agreement with loss 0 and abort valued "
          "delta_0; signatures are constant on fibres, replacement law is a function of the log law on a "
          f"125-point grid, and two laws with agreement mass {agreeA} vs {agreeB} give the same replacement law.")


# ---------------------------------------------------------------------------
# Coarse identified interval of the replacement mean (unrestricted law class)
# ---------------------------------------------------------------------------

def coarse_mean_interval(P_g, fib, m_lo, m_hi):
    """Identified interval of the replacement mean under a log whose observed law is P_g.

    The compatible full laws are all laws on Omega0 with this log law; the mean is
    linear in the full law, so its extremes are at vertices that put each fibre's
    mass on one member. m_lo/m_hi give each member's lowest/highest mean contribution.
    """
    ys = list(P_g)
    lows, highs = [], []
    for choice in product(*[fib[y] for y in ys]):
        lows.append(sum(P_g[y] * m_lo[w] for y, w in zip(ys, choice)))
        highs.append(sum(P_g[y] * m_hi[w] for y, w in zip(ys, choice)))
    return min(lows), max(highs)


# ---------------------------------------------------------------------------
# Check 5: strict logging example (outcome-only log)
# ---------------------------------------------------------------------------

def check_strict_logging():
    omega0, steps = incident_omega0()
    sig = lambda w: signature(w, steps, INC_F0, INC_FK, "gate", INC_B)
    fib = fibres(omega0, g_outcome)
    assert len({sig(w) for w in fib[1]}) == 2
    mechs = [(HALF, HALF, HALF), (F(1), HALF, F(0)), (F(0), F(0), HALF)]
    imgs, truths = [], []
    for prm in mechs:
        st, loss = incident_mechanism(*prm)
        P0 = forward(st, {"gate": INC_F0}, loss, INC_B)
        imgs.append(push(P0, g_outcome))
        truths.append(mean(law(forward(st, {"gate": INC_FK}, loss, INC_B))))
    assert imgs[0] == imgs[1] == imgs[2]
    assert truths == [F(1, 4), F(1, 2), F(0)]
    m = {w: mean(dict(sig(w)[0])) for w in omega0}
    lo, hi = coarse_mean_interval(imgs[0], fib, m, m)
    assert (lo, hi) == (F(0), F(1, 2)) and lo < truths[0] < hi
    print("PASS T1-5 strict logging: an outcome-only log (no gate input) violates the fibre condition; three "
          "mechanisms with one log law give replacement means 1/4, 1/2, 0, and the coarse identified interval "
          "[0, 1/2] strictly contains the full-record value {1/4}.")


# ---------------------------------------------------------------------------
# Retry mechanism shared by checks 6 and 7 (saturated continuation)
# ---------------------------------------------------------------------------

def retry_mechanism(n, c, q, after_retry=None):
    def outcome(i):
        if i[0] == "go":
            return bern(c)
        if after_retry is not None:
            return after_retry
        return bern(q)

    steps = [
        rand("U", lambda h: (), lambda i: uniform(range(n))),
        dec("A", "policy", 0, lambda h: (h["U"],)),
        rand("Y", lambda h: (h["A"],), outcome),
    ]
    return steps, (lambda h: h["Y"])


RT_F0 = lambda t, i: "go"
RT_FK = lambda t, i: "retry" if i[0] == 0 else "go"
NO_ABORT = lambda h: {}


def saturated_set(n, c, qs, after=None):
    """Replacement laws over the model class indexed by the unknown retry kernel."""
    base, laws = None, []
    for q in qs:
        st, loss = retry_mechanism(n, c, q, after)
        P0 = forward(st, {"policy": RT_F0}, loss)
        if base is None:
            base = P0
        assert P0 == base  # the retry input has zero baseline probability
        laws.append(law(forward(st, {"policy": RT_FK}, loss)))
    return base, laws


def pointwise_min(laws, alphabet):
    return {l: min(d.get(l, F(0)) for d in laws) for l in alphabet}


# ---------------------------------------------------------------------------
# Check 6: injectivity lemma
# ---------------------------------------------------------------------------

def check_injectivity():
    alphabet = (0, 1)
    sets, summaries = {}, {}
    for n, c in ((10, F(0)), (5, F(0)), (10, F(1, 3))):
        st, _ = retry_mechanism(n, c, 0)
        P0, laws = saturated_set(n, c, (F(0), F(1)))
        nu, eps, _ = replay(P0, st, RT_F0, RT_FK, "policy", NO_ABORT)
        # The class realizes exactly the segment {nu + eps Q}: endpoints and an affine midpoint.
        assert laws[0] == add(nu, {0: eps}) and laws[1] == add(nu, {1: eps})
        _, mid = saturated_set(n, c, (HALF,))
        assert mid[0] == add(scale(laws[0], HALF), scale(laws[1], HALF))
        assert pointwise_min(laws, alphabet) == {l: nu.get(l, F(0)) for l in alphabet}
        sets[(n, c)] = pointwise_min(laws, alphabet)
        summaries[(n, c)] = (tuple(sorted(nu.items())), eps)
    assert len(set(summaries.values())) == 3
    assert len({tuple(sorted(v.items())) for v in sets.values()}) == 3
    # |L| = 1: the hypothesis |L| >= 2 is needed.
    singles = set()
    for n in (10, 5):
        st, _ = retry_mechanism(n, F(0), 0, after_retry={0: F(1)})
        P0, laws = saturated_set(n, F(0), (F(0), F(1)), after={0: F(1)})
        nu, eps, _ = replay(P0, st, RT_F0, RT_FK, "policy", NO_ABORT)
        singles.add((tuple(sorted(nu.items())), eps, tuple(tuple(sorted(d.items())) for d in laws)))
    assert len({s[:2] for s in singles}) == 2 and len({s[2] for s in singles}) == 1
    # Prefix-dependent extension: a continuing prefix with singleton feasible set {1}
    # has the same identified set as a terminal deviation valued delta_1.
    st, loss = retry_mechanism(10, F(0), 0, after_retry={1: F(1)})
    P0, laws_cont = saturated_set(10, F(0), (F(0), F(1)), after={1: F(1)})
    nu_c, eps_c, _ = replay(P0, st, RT_F0, RT_FK, "policy", NO_ABORT)
    fk_abort = lambda t, i: ABORT if i[0] == 0 else "go"
    b1 = lambda h: {1: F(1)}
    nu_t, eps_t, _ = replay(P0, st, RT_F0, fk_abort, "policy", b1)
    law_t = law(forward(st, {"policy": fk_abort}, loss, b1))
    assert (nu_c, eps_c) != (nu_t, eps_t)
    assert all(d == law_t for d in laws_cont)
    # General reduced-summary recovery by minima over events and Moebius inversion.
    recovered, truth = moebius_recovery()
    assert recovered == truth
    print("PASS T1-6 injectivity: in the saturated retry class with |L|=2, the pointwise minimum recovers nu and "
          "three summaries give three sets; |L|=1 and a singleton feasible set break injectivity; event minima "
          "plus Moebius inversion recover the reduced summary in a two-prefix example.")


def moebius_recovery():
    """Two continuing prefixes with feasible sets {0,1} and {1,2}; recover their weights from the set."""
    def mech(qc, qp):
        steps = [
            rand("W", lambda h: (), lambda i: {"cheap": F(1, 4), "paid": F(1, 4), "done": HALF}),
            rand("C0", lambda h: (h["W"],), lambda i: {1 if i[0] == "paid" else 0: F(1)}),
            dec("A", "policy", 1, lambda h: (h["W"], h["C0"])),
            rand("C1", lambda h: (h["W"], h["A"]),
                 lambda i: {0: F(1)} if i[1] == "finish" else bern(qc if i[0] == "cheap" else qp)),
        ]
        return steps, (lambda h: h["C0"] + h["C1"])

    f0 = lambda t, i: "finish"
    fk = lambda t, i: "retry" if i[0] in ("cheap", "paid") else "finish"
    alphabet = (0, 1, 2)
    laws, feas, base = [], {}, None
    for qc, qp in product((F(0), HALF, F(1)), repeat=2):
        st, loss = mech(qc, qp)
        P0 = forward(st, {"policy": f0}, loss)
        base = P0 if base is None else base
        assert P0 == base
        Pk = forward(st, {"policy": fk}, loss)
        laws.append(law(Pk))
        for w in Pk:
            _, status, pre = first_deviation(w, st, f0, fk, "policy")
            if status == "continuing":
                feas.setdefault(pre, set()).add(loss_of(w))
    st, _ = mech(0, 0)
    nu, eps, cont = replay(base, st, f0, fk, "policy", NO_ABORT)
    truth = {}
    for pre, wgt in cont.items():
        S = frozenset(feas[pre])
        truth[S] = truth.get(S, F(0)) + wgt
    assert set(truth) == {frozenset({0, 1}), frozenset({1, 2})}
    subsets = [frozenset(s) for r in range(len(alphabet) + 1) for s in product(alphabet, repeat=r)
               if len(set(s)) == r and list(s) == sorted(s)]
    psi = {B: min(sum(d.get(l, F(0)) for l in B) for d in laws) for B in subsets}
    nu_red = {l: psi[frozenset({l})] for l in alphabet}
    assert clean(nu_red) == nu
    D = {B: (psi[B] - sum(nu_red[l] for l in B)) if len(B) >= 2 else F(0) for B in subsets}
    recovered = {}
    for S in subsets:
        if len(S) >= 2:
            val = sum((-1) ** (len(S) - len(B)) * D[B] for B in subsets if B <= S)
            if val:
                recovered[S] = val
    return recovered, truth


# ---------------------------------------------------------------------------
# Check 7: attained saturated endpoints (mean, CDF with endpoint correction, CVaR)
# ---------------------------------------------------------------------------

def check_saturated_endpoints():
    n, c = 10, F(0)
    st, _ = retry_mechanism(n, c, 0)
    qs = [F(0), F(1, 4), HALF, F(3, 4), F(1)]
    P0, laws = saturated_set(n, c, qs)
    nu, eps, _ = replay(P0, st, RT_F0, RT_FK, "policy", NO_ABORT)
    a, b = 0, 1  # feasible values after the retry: {0, 1}
    assert eps == F(1, 10) and nu == {0: F(9, 10)}
    lower, upper = add(nu, {a: eps}), add(nu, {b: eps})
    assert laws[0] == lower and laws[-1] == upper
    # Mean interval [int x dnu + eps a, int x dnu + eps b], attained.
    means = [mean(d) for d in laws]
    assert (min(means), max(means)) == (mean(nu) + eps * a, mean(nu) + eps * b) == (F(0), F(1, 10))
    # CDF envelopes at every x, including the endpoint correction at b.
    for x in (F(-1), F(0), HALF, F(1), F(2)):
        vals = [cdf(d, x) for d in laws]
        assert min(vals) == cdf(nu, x) + eps * (1 if x >= b else 0)
        assert max(vals) == cdf(nu, x) + eps * (1 if x >= a else 0)
    assert min(cdf(d, 1) for d in laws) == 1 and cdf(nu, 1) == F(9, 10)
    # FOSD: every law in the class lies between the two completions; monotone functionals.
    beta = F(9, 10)
    for d in laws:
        for x in (F(-1), F(0), HALF, F(1)):
            assert cdf(upper, x) <= cdf(d, x) <= cdf(lower, x)
        assert cvar(lower, beta) <= cvar(d, beta) <= cvar(upper, beta)
        assert var(lower, beta) <= var(d, beta) <= var(upper, beta)
    width = cvar(upper, beta) - cvar(lower, beta)
    assert cvar(lower, beta) == 0 and cvar(upper, beta) == 1
    assert width == 1 and eps * (b - a) == F(1, 10)
    # Proposition (Order Bounds)(c): CVaR along the segment is a minimum of affine functions of lambda,
    # so every intermediate value is attained; solve for one exactly and realize it in the class.
    lam = cvar_intermediate(lower, upper, beta)
    _, mid = saturated_set(n, c, (lam,))
    assert mid[0] == add(scale(lower, 1 - lam), scale(upper, lam))
    assert cvar(mid[0], beta) == (cvar(lower, beta) + cvar(upper, beta)) / 2
    # A segment on which CVaR is not affine in lambda: CVaR_{1/2} of (1-lam) delta_0 + lam delta_1 is min(2 lam, 1).
    lo2, hi2 = {0: F(1)}, {1: F(1)}
    assert cvar(segment(lo2, hi2, HALF), HALF) == 1 != (cvar(lo2, HALF) + cvar(hi2, HALF)) / 2
    assert cvar_intermediate(lo2, hi2, HALF) == F(1, 4)
    print("PASS T1-7 saturated endpoints: eps=1/10 on {0,1}; mean range [0,1/10] attained; CDF envelopes match "
          "nu+eps*1{x>=b} and nu+eps*1{x>=a} (lower CDF at b is 1, not 9/10); CVaR_0.9 endpoints 0 and 1, "
          f"width 1 vs eps(b-a)=1/10; the CVaR midpoint is attained exactly at lambda={lam} by a class member.")


def segment(lo, hi, lam):
    return add(scale(lo, 1 - lam), scale(hi, lam))


def cvar_intermediate(lo, hi, beta):
    """Exact intermediate-value check for CVaR on the segment (1-lam) lo + lam hi.

    CVaR(eta_lam) = min_c {A_c + B_c lam} over the joint support (Rockafellar-Uryasev), a concave
    piecewise-linear and hence continuous function of lam; solve CVaR = midpoint exactly.
    """
    V = sorted(set(lo) | set(hi))
    tail = lambda d, cc: sum(p * max(x - cc, 0) for x, p in d.items())
    A = {cc: cc + tail(lo, cc) / (1 - beta) for cc in V}
    B = {cc: (tail(hi, cc) - tail(lo, cc)) / (1 - beta) for cc in V}
    for k in range(9):
        lam = F(k, 8)
        assert cvar(segment(lo, hi, lam), beta) == min(A[cc] + B[cc] * lam for cc in V)
    v_lo, v_hi = cvar(lo, beta), cvar(hi, beta)
    assert v_lo < v_hi
    target = (v_lo + v_hi) / 2
    for cc in V:
        if B[cc] != 0:
            lam = (target - A[cc]) / B[cc]
            if 0 <= lam <= 1 and cvar(segment(lo, hi, lam), beta) == target:
                return lam
    raise AssertionError("no exact intermediate CVaR value found")


# ---------------------------------------------------------------------------
# Check 8: prefix-dependent feasibility witness
# ---------------------------------------------------------------------------

def check_prefix_feasibility():
    def mech(q):
        steps = [
            rand("W", lambda h: (), lambda i: {"cheap": HALF, "paid": HALF}),
            rand("C0", lambda h: (h["W"],), lambda i: {1 if i[0] == "paid" else 0: F(1)}),
            dec("A", "policy", 1, lambda h: (h["W"], h["C0"])),
            rand("C1", lambda h: (h["A"],), lambda i: {0: F(1)} if i[0] == "finish" else bern(q)),
        ]
        return steps, (lambda h: h["C0"] + h["C1"])

    f0 = lambda t, i: "finish"
    fk = lambda t, i: "retry" if i[0] == "paid" else "finish"
    qs = [F(0), F(1, 3), F(1)]
    base, laws_all, laws_paid, feas, global_vals = None, [], [], set(), set()
    paid = lambda w: get(w, "W") == "paid"
    for q in qs:
        st, loss = mech(q)
        P0 = forward(st, {"policy": f0}, loss)
        base = P0 if base is None else base
        assert P0 == base
        Pk = forward(st, {"policy": fk}, loss)
        laws_all.append(law(Pk))
        laws_paid.append(law(Pk, paid))
        global_vals |= {loss_of(w) for w in P0} | {loss_of(w) for w in Pk}
        feas |= {loss_of(w) for w in Pk if first_deviation(w, st, f0, fk, "policy")[1] == "continuing"}
    st, _ = mech(0)
    a_glob, b_glob = min(global_vals), max(global_vals)
    assert (a_glob, b_glob) == (0, 2) and feas == {1, 2}
    for cond, laws, sharp in ((paid, laws_paid, (F(1), F(2))), (lambda w: True, laws_all, (HALF, F(1)))):
        nu, eps, _ = replay(base, st, f0, fk, "policy", NO_ABORT, cond)
        attained = [mean(d) for d in laws]
        assert (min(attained), max(attained)) == sharp
        assert sharp == (mean(nu) + eps * min(feas), mean(nu) + eps * max(feas))
        naive = (mean(nu) + eps * a_glob, mean(nu) + eps * b_glob)
        assert naive[0] < sharp[0] and naive[1] == sharp[1]
    assert all(mean(d) >= 1 for d in laws_paid)
    print("PASS T1-8 prefix feasibility: accrued cost 1 before a retry; conditional mean range [1,2] (global "
          "formula gives [0,2]); unconditional sharp range [1/2,1] vs [0,1].")


# ---------------------------------------------------------------------------
# Check 9: garbling inclusion, strict and equal
# ---------------------------------------------------------------------------

def check_garbling():
    omega0, steps = incident_omega0()
    sig = lambda w: signature(w, steps, INC_F0, INC_FK, "gate", INC_B)
    m = {w: mean(dict(sig(w)[0])) for w in omega0}

    def is_garbling(g_fine, g_coarse):
        return all(len({g_coarse(w) for w in ws}) == 1 for ws in fibres(omega0, g_fine).values())

    chain = [g_fullrec, g_incident, g_outcome, g_const]
    for fine, coarse in zip(chain, chain[1:]):
        assert is_garbling(fine, coarse)
    assert not is_garbling(g_outcome, g_incident)
    st, loss = incident_mechanism(HALF, HALF, HALF)
    P0 = forward(st, {"gate": INC_F0}, loss, INC_B)
    truth = mean(law(forward(st, {"gate": INC_FK}, loss, INC_B)))
    sets = [coarse_mean_interval(push(P0, g), fibres(omega0, g), m, m) for g in chain]
    for (lo1, hi1), (lo2, hi2) in zip(sets, sets[1:]):
        assert lo2 <= lo1 and hi1 <= hi2
    assert sets[0] == sets[1] == (truth, truth) == (F(1, 4), F(1, 4))
    assert sets[2] == (F(0), HALF) and sets[3] == (F(0), F(1))
    print("PASS T1-9 garbling: full log -> incident log -> outcome log -> constant are known garblings; "
          "identified mean sets {1/4} = {1/4} (equality) within [0,1/2] within [0,1] (strict).")


# ---------------------------------------------------------------------------
# Check 10: conditional non-nesting for differently conditioned targets
# ---------------------------------------------------------------------------

def check_conditional_non_nesting():
    steps = [rand("H", lambda h: (), lambda i: {0: HALF, 1: HALF})]
    P = forward(steps, {}, lambda h: h["H"])
    rich = lambda w: get(w, "H")
    coarse = lambda w: "const"
    rich_target = {mean(law(P, lambda w: rich(w) == 1))}
    coarse_target = {mean(law(P, lambda w: coarse(w) == "const"))}
    assert rich_target == {F(1)} and coarse_target == {HALF}
    assert not rich_target <= coarse_target
    same_event_rich = {mean(law(P, lambda w: True))}
    assert same_event_rich <= coarse_target
    print("PASS T1-10 non-nesting (illustration with a fully specified model, identity replacement): L=H fair bit; "
          "E[L|H=1] is {1} under rich records and E[L] is {1/2} under constant records, so the sets are not nested; "
          "on a common coarse event both are {1/2}.")


# ---------------------------------------------------------------------------
# Check 11: anticipation counterexample
# ---------------------------------------------------------------------------

def check_anticipation():
    def proposal(announced):
        if announced:
            return lambda i: {"benign": HALF, "risky_unflagged": HALF}
        return lambda i: {"benign": HALF, "risky_flagged": HALF}

    def mech(announced):
        steps = [
            rand("P", lambda h: (), proposal(announced)),
            dec("G", "gate", 0, lambda h: (h["P"],)),
            rand("Y", lambda h: (h["P"], h["G"]),
                 lambda i: {1 if (i[1] == "EXEC" and i[0].startswith("risky")) else 0: F(1)}),
        ]
        return steps, (lambda h: h["Y"])

    f0 = lambda t, i: "EXEC"
    fk = lambda t, i: ABORT if i[0] == "risky_flagged" else "EXEC"
    b = lambda h: {0: F(1)}
    st0, loss = mech(False)
    P0 = forward(st0, {"gate": f0}, loss, b)
    nu, eps, _ = replay(P0, st0, f0, fk, "gate", b)
    assert eps == 0
    predicted = mean(nu)
    st_silent, _ = mech(False)
    st_announced, _ = mech(True)
    true_silent = mean(law(forward(st_silent, {"gate": fk}, loss, b)))
    true_announced = mean(law(forward(st_announced, {"gate": fk}, loss, b)))
    assert proposal(True)(()) != proposal(False)(())  # the proposal kernel changed: (A1) and (A2) fail
    assert predicted == true_silent == 0
    assert true_announced == HALF != predicted
    print("PASS T1-11 anticipation: announcing the gate changes the proposal kernel at the same declared input; "
          "replay predicts harm 0, the announced deployment has harm 1/2, the silent gate has harm 0 as replayed.")


# ---------------------------------------------------------------------------
# Check 12: one fixed scalar query may need a smaller statistic
# ---------------------------------------------------------------------------

def check_fixed_query_smaller_statistic():
    def mech(kcost):
        steps = [
            rand("Z", lambda h: (), lambda i: {"low": HALF, "high": HALF}),
            dec("G", "gate", 0, lambda h: (h["Z"],)),
            rand("Y", lambda h: (h["Z"], h["G"]),
                 lambda i: bern(F(1, 4) if i[0] == "low" else F(3, 4)) if i[1] == "EXEC" else {0: F(1)}),
            rand("K", lambda h: (h["G"],), lambda i: {1: F(kcost), 2: 1 - F(kcost)}),
        ]
        return steps, (lambda h: (h["Y"], h["K"]))

    b = lambda h: {(0, 0): F(1)}
    g_harm = lambda w: (get(w, "Z"), loss_of(w)[0])
    harm = lambda l: l[0]
    st, loss = mech(HALF)
    omega0 = set(forward(st, {"gate": INC_F0}, loss, b))
    sig = lambda w: signature(w, st, INC_F0, INC_FK, "gate", b)

    def push_sig(s):
        out = {}
        for l, p in s[0]:
            out[harm(l)] = out.get(harm(l), F(0)) + p
        return (tuple(sorted(out.items())), s[1])

    fib = fibres(omega0, g_harm)
    assert any(len({sig(w) for w in ws}) > 1 for ws in fib.values())
    assert all(len({push_sig(sig(w)) for w in ws}) == 1 for ws in fib.values())
    runs = []
    for kc in (F(1, 4), F(3, 4)):
        stk, lossk = mech(kc)
        P0 = forward(stk, {"gate": INC_F0}, lossk, b)
        Pk = forward(stk, {"gate": INC_FK}, lossk, b)
        runs.append((push(P0, g_harm), law(Pk), law(Pk, phi=harm)))
    assert runs[0][0] == runs[1][0] and runs[0][1] != runs[1][1] and runs[0][2] == runs[1][2]
    print("PASS T1-12 fixed query: a harm-only log violates the fibre condition for the vector (harm, cost) summary "
          "yet determines the harm law under the gate; two laws with one log law differ only in cost.")


# ---------------------------------------------------------------------------
# Check 13: without saturation the displayed set is conservative
# ---------------------------------------------------------------------------

def supported_mechanism(pu, qy, reads_tape):
    """U ~ Bern(pu); A = policy(U); Y ~ Bern(qy[input]) with input (A,) or (U, A)."""
    steps = [
        rand("U", lambda h: (), lambda i: bern(pu)),
        dec("A", "policy", 0, lambda h: (h["U"],)),
        rand("Y", (lambda h: (h["U"], h["A"])) if reads_tape else (lambda h: (h["A"],)),
             lambda i: bern(qy[i])),
    ]
    return steps, (lambda h: h["Y"])


def check_non_saturated_conservative():
    f0 = lambda t, i: i[0]
    fk = lambda t, i: 1 - i[0]
    true_q = {(0,): F(1, 4), (1,): F(3, 4)}
    steps, loss = supported_mechanism(HALF, true_q, False)
    P0 = forward(steps, {"policy": f0}, loss)
    Pk = forward(steps, {"policy": fk}, loss)
    nu, eps, _ = replay(P0, steps, f0, fk, "policy", NO_ABORT)
    assert nu == {} and eps == 1
    assert mean(law(Pk)) == HALF
    grid = [F(0), F(1, 4), F(1, 3), HALF, F(2, 3), F(3, 4), F(1)]
    # Restricted class: the outcome kernel reads only A. Enumerate every grid mechanism whose
    # baseline law equals P0; the replacement mean is the same for all of them.
    restricted = set()
    for pu, q0, q1 in product(grid, repeat=3):
        st, ls = supported_mechanism(pu, {(0,): q0, (1,): q1}, False)
        if forward(st, {"policy": f0}, ls) == P0:
            restricted.add(mean(law(forward(st, {"policy": fk}, ls))))
    assert restricted == {HALF}
    # Larger class: the outcome kernel may also read the tape U. The replacement reaches the inputs
    # (U, 1-U), which have zero baseline probability, and the identified mean fills [0, 1].
    larger = set()
    for pu, a, b_, c_, d_ in product([HALF], grid, grid, grid, grid):
        qy = {(0, 0): a, (0, 1): b_, (1, 0): c_, (1, 1): d_}
        st, ls = supported_mechanism(pu, qy, True)
        if forward(st, {"policy": f0}, ls) == P0:
            larger.add(mean(law(forward(st, {"policy": fk}, ls))))
    assert min(larger) == 0 and max(larger) == 1 and HALF in larger
    assert mean(nu) + eps * 0 <= HALF <= mean(nu) + eps * 1
    print("PASS T1-13 non-saturated: agreement mass 0 and eps=1 give the outer interval [0,1]; over every "
          "grid mechanism with the baseline law, the identified mean is {1/2} when the outcome kernel reads only "
          "A, and fills [0,1] when it may also read the tape.")


# ---------------------------------------------------------------------------
# Check 14: a shared continuation input breaks saturation (V2 of the verifier)
# ---------------------------------------------------------------------------

def shared_input_mechanism(qc, qp, flip, distinct):
    """Types c and p are both retried; the unknown kernel N reads the action only (shared input)
    or the action and the type (distinct inputs)."""
    def nker(i):
        if i[-1] == "go":
            return {0: F(1)}
        if distinct:
            return bern(qc if i[0] == "c" else qp)
        return bern(qc)

    steps = [
        rand("W", lambda h: (), lambda i: {"c": HALF, "p": HALF}),
        dec("A", "policy", 0, lambda h: (h["W"],)),
        rand("N", (lambda h: (h["W"], h["A"])) if distinct else (lambda h: (h["A"],)), nker),
    ]
    if flip:
        loss = lambda h: h["N"] if h["W"] == "c" else 1 - h["N"]
    else:
        loss = lambda h: h["N"] + (1 if h["W"] == "p" else 0)
    return steps, loss


def check_shared_input():
    f0 = lambda t, i: "go"
    fk = lambda t, i: "retry"
    grid = [F(k, 10) for k in range(11)]
    is_c = lambda w: get(w, "W") == "c"
    is_p = lambda w: get(w, "W") == "p"

    def run(flip, distinct, pairs):
        base, laws, cond_c, cond_p, feas = None, [], set(), set(), {}
        for qc, qp in pairs:
            st, ls = shared_input_mechanism(qc, qp, flip, distinct)
            P0 = forward(st, {"policy": f0}, ls)
            base = P0 if base is None else base
            assert P0 == base                                  # zero baseline support of the unknown input
            Pk = forward(st, {"policy": fk}, ls)
            laws.append(law(Pk))
            cond_c.add(tuple(sorted(law(Pk, is_c).items())))
            cond_p.add(tuple(sorted(law(Pk, is_p).items())))
            for w in Pk:
                _, s, pre = first_deviation(w, st, f0, fk, "policy")
                feas.setdefault(pre, set()).add(loss_of(w))
        st, _ = shared_input_mechanism(0, 0, flip, distinct)
        nu, eps, cont = replay(base, st, f0, fk, "policy", NO_ABORT)
        lo = add(nu, *[{min(feas[p]): w} for p, w in cont.items()])
        hi = add(nu, *[{max(feas[p]): w} for p, w in cont.items()])
        return laws, cond_c, cond_p, feas, lo, hi

    # Shared input, L = N for c and 1 - N for p: each Q_h alone is free, the joint law is not.
    laws, cond_c, cond_p, feas, lo, hi = run(True, False, [(q, q) for q in grid])
    assert all(v == {0, 1} for v in feas.values()) and len(feas) == 2
    assert len(cond_c) == len(cond_p) == len(grid)           # each prefix's law sweeps the grid
    assert all(d == {0: HALF, 1: HALF} for d in laws)         # every mechanism gives Bern(1/2)
    assert lo == {0: F(1)} and hi == {1: F(1)} and lo not in laws and hi not in laws
    # Shared input, L = N + 1{p}: mean endpoints attained, a product-form element of S is not.
    laws_s, _, _, _, lo_s, hi_s = run(False, False, [(q, q) for q in grid])
    means = [mean(d) for d in laws_s]
    assert (min(means), max(means)) == (mean(lo_s), mean(hi_s)) == (HALF, F(3, 2))
    assert {0: HALF, 2: HALF} not in laws_s
    # Control: N also reads the type, so the two prefixes reach distinct inputs and S is attained.
    laws_d, _, _, _, lo_d, hi_d = run(True, True, list(product((F(0), HALF, F(1)), repeat=2)))
    assert lo_d in laws_d and hi_d in laws_d and {0: HALF, 1: HALF} in laws_d
    laws_ds, _, _, _, _, _ = run(False, True, list(product((F(0), F(1)), repeat=2)))
    assert {0: HALF, 2: HALF} in laws_ds
    print("PASS T1-14 shared input: two retried prefixes reach one unknown input; with L=N for c and 1-N for p each "
          "Q_h sweeps Delta({0,1}) yet every mechanism gives Bern(1/2), so eta-/eta+ are not attained; with "
          "L=N+1{p} the mean endpoints 1/2, 3/2 are attained but 1/2 delta_0 + 1/2 delta_2 is not; with distinct "
          "inputs both sets are attained.")


# ---------------------------------------------------------------------------
# Check 15: prefix-dependent feasible sets need the reduced signature for logging
# ---------------------------------------------------------------------------

def check_prefix_dependent_logging():
    def mech(pc, qc, qp):
        steps = [
            rand("W", lambda h: (), lambda i: clean({"cheap": F(pc), "paid": 1 - F(pc)})),
            rand("C0", lambda h: (h["W"],), lambda i: {1 if i[0] == "paid" else 0: F(1)}),
            dec("A", "policy", 0, lambda h: (h["W"], h["C0"])),
            rand("C1", lambda h: (h["W"], h["A"]),
                 lambda i: {0: F(1)} if i[1] == "finish" else bern(qc if i[0] == "cheap" else qp)),
        ]
        return steps, (lambda h: h["C0"] + h["C1"])

    f0 = lambda t, i: "finish"
    fk = lambda t, i: "retry"
    st, ls = mech(HALF, 0, 0)
    omega0 = set(forward(st, {"policy": f0}, ls))
    # Feasible sets of each continuing prefix, from the replacement-consistent extensions.
    feas = {}
    for qc, qp in product((F(0), F(1)), repeat=2):
        stq, lsq = mech(HALF, qc, qp)
        for w in forward(stq, {"policy": fk}, lsq):
            _, s, pre = first_deviation(w, stq, f0, fk, "policy")
            feas.setdefault(pre, set()).add(loss_of(w))
    sig = lambda w: signature(w, st, f0, fk, "policy", NO_ABORT)

    def reduced(w):
        _, s, pre = first_deviation(w, st, f0, fk, "policy")
        if s != "continuing":
            return (sig(w)[0], frozenset())
        S = frozenset(feas[pre])
        return (frozenset({(min(S), F(1))}), frozenset()) if len(S) == 1 else (frozenset(), S)

    g_const = lambda w: "const"
    g_type = lambda w: get(w, "W")
    assert all(len({sig(w) for w in ws}) == 1 for ws in fibres(omega0, g_const).values())
    assert any(len({reduced(w) for w in ws}) > 1 for ws in fibres(omega0, g_const).values())
    assert all(len({reduced(w) for w in ws}) == 1 for ws in fibres(omega0, g_type).values())
    sets, logs, summaries = {}, set(), set()
    for pc in (F(0), HALF, F(1)):
        stp, lsp = mech(pc, 0, 0)
        P0 = forward(stp, {"policy": f0}, lsp)
        nu, eps, _ = replay(P0, stp, f0, fk, "policy", NO_ABORT)
        summaries.add((tuple(nu.items()), eps))
        logs.add(tuple(push(P0, g_const).items()))
        means = []
        for qc, qp in product((F(0), F(1)), repeat=2):
            stq, lsq = mech(pc, qc, qp)
            assert forward(stq, {"policy": f0}, lsq) == P0      # the retry inputs have zero baseline mass
            means.append(mean(law(forward(stq, {"policy": fk}, lsq))))
        sets[pc] = (min(means), max(means))
    assert len(logs) == 1 and summaries == {((), F(1))}
    assert sets[F(1)] == (F(0), F(1)) and sets[F(0)] == (F(1), F(2)) and sets[HALF] == (HALF, F(3, 2))
    print("PASS T1-15 prefix-dependent logging: a constant log keeps (C,u)=(0,1) constant and the summary (0,1) "
          "for every law, yet the saturated mean sets are [0,1] (all cheap), [1/2,3/2], and [1,2] (all paid); the "
          "reduced signature separates the fibre, and logging the task type restores its constancy.")


# ---------------------------------------------------------------------------
# Check 16: feasible sets follow later replaced decisions; declared later aborts break saturation
# ---------------------------------------------------------------------------

def check_feasible_set_and_later_abort():
    def mech(q, later):
        steps = [
            dec("A0", "gate", 0, lambda h: ()),
            rand("N", lambda h: (h["A0"],), lambda i: {0: F(1)} if i[0] == "go" else bern(q)),
            dec("A1", "gate", 1, lambda h: (h["A0"], h["N"])),
        ]
        if later == "skip":    # exec costs 2 after N=1 and 1 after N=0; skip costs 0
            loss = lambda h: 0 if h["A1"] == "skip" else (2 if h["N"] == 1 else 1)
        else:                  # exec costs 1; an abort is valued by the declared post-abort law
            loss = lambda h: 1
        return steps, loss

    f0 = lambda t, i: "go" if t == 0 else "exec"
    fk_skip = lambda t, i: "retry" if t == 0 else ("skip" if i[1] == 1 else "exec")
    fk_abort = lambda t, i: "retry" if t == 0 else (ABORT if i[1] == 1 else "exec")
    later_val = lambda h: {0: F(3, 5), 2: F(2, 5)}

    # (a) The replacement-consistent feasible set is {0,1}; all tree extensions give {0,1,2}.
    tree_vals = set()
    for nv, a1 in product((0, 1), ("exec", "skip")):
        tree_vals.add(0 if a1 == "skip" else (2 if nv == 1 else 1))
    feas, means, base = set(), [], None
    for q in (F(0), F(1, 3), F(1)):
        st, ls = mech(q, "skip")
        P0 = forward(st, {"gate": f0}, ls)
        base = P0 if base is None else base
        assert P0 == base
        Pk = forward(st, {"gate": fk_skip}, ls)
        feas |= {loss_of(w) for w in Pk}
        means.append(mean(law(Pk)))
    st, _ = mech(0, "skip")
    nu, eps, _ = replay(base, st, f0, fk_skip, "gate", NO_ABORT)
    assert nu == {} and eps == 1 and tree_vals == {0, 1, 2} and feas == {0, 1}
    assert (min(means), max(means)) == (F(0), F(1)) == (min(feas), max(feas))
    assert max(tree_vals) == 2 > max(means)
    # (b) A declared valuation of the later abort fixes part of the continuation law.
    laws, base = [], None
    for k in range(11):
        st, ls = mech(F(k, 10), "abort")
        P0 = forward(st, {"gate": f0}, ls, later_val)
        base = P0 if base is None else base
        assert P0 == base
        laws.append(law(forward(st, {"gate": fk_abort}, ls, later_val)))
    feas_b = set().union(*[set(d) for d in laws])
    m = [mean(d) for d in laws]
    assert feas_b == {0, 1, 2} and (min(m), max(m)) == (F(4, 5), F(1))
    assert {0: F(1)} not in laws and {2: F(1)} not in laws
    print("PASS T1-16 feasible set: after a retry a later replaced gate skips the costly branch, so X_h={0,1} and "
          "the sharp mean range is [0,1] (tree extensions give [0,2]); a declared later-abort valuation leaves "
          "X_h={0,1,2} but only means in [4/5,1], so eta-/eta+ are not attained and [0,2] is an outer bound.")


# ---------------------------------------------------------------------------
# Check 17: a restricted class (known tape law) has no necessity witness (V4 of the verifier)
# ---------------------------------------------------------------------------

def check_restricted_class():
    def mech(p):
        steps = [
            rand("X", lambda h: (), lambda i: clean({0: F(p), 1: 1 - F(p)})),
            rand("U", lambda h: (), lambda i: {0: HALF, 1: HALF}),    # known sampler law
            dec("G", "gate", 0, lambda h: (h["U"],)),
            rand("Y", lambda h: (h["X"], h["G"]), lambda i: {i[0] if i[1] == "EXEC" else 0: F(1)}),
        ]
        return steps, (lambda h: h["Y"])

    f0 = lambda t, i: "EXEC"
    fk = lambda t, i: ABORT if i[0] == 1 else "EXEC"
    b = lambda h: {0: F(1)}
    g = lambda w: (get(w, "X"), loss_of(w))                           # omits U
    st, ls = mech(HALF)
    omega0 = set(forward(st, {"gate": f0}, ls, b))
    sig = lambda w: signature(w, st, f0, fk, "gate", b)
    fib = fibres(omega0, g)
    bad = [y for y, ws in fib.items() if len({sig(w) for w in ws}) > 1]
    assert bad == [(1, 1)]
    # Unrestricted class: the two point masses in the bad fibre share a log law, summaries differ.
    w_agree, w_term = sorted(fib[(1, 1)], key=lambda w: get(w, "U"))
    assert push({w_agree: F(1)}, g) == push({w_term: F(1)}, g)
    assert replay({w_agree: F(1)}, st, f0, fk, "gate", b)[0] != replay({w_term: F(1)}, st, f0, fk, "gate", b)[0]
    # Class with the tape law fixed: the summary is the fixed function nu = 1/2 law(X) + 1/2 delta_0 of the log.
    for p in (F(0), F(1, 3), HALF, F(1)):
        stp, lsp = mech(p)
        P0 = forward(stp, {"gate": f0}, lsp, b)
        nu, eps, _ = replay(P0, stp, f0, fk, "gate", b)
        pred = add(*[scale({y[1]: F(1)}, m * HALF) for y, m in push(P0, g).items()], {0: HALF})
        assert nu == pred and eps == 0
    print("PASS T1-17 restricted class: a log of (X,L) omits a known fair tape; the fibre condition fails at "
          "{X=1,L=1} and two point masses there differ, yet with the tape law fixed the summary is a fixed function "
          "of the log law, so necessity needs admissible witnesses.")


if __name__ == "__main__":
    check_stopping_law_without_abort_support()
    check_restricted_measure_seeded_policy()
    check_missing_outcome()
    check_equal_contribution_different_status()
    check_strict_logging()
    check_injectivity()
    check_saturated_endpoints()
    check_prefix_feasibility()
    check_garbling()
    check_conditional_non_nesting()
    check_anticipation()
    check_fixed_query_smaller_statistic()
    check_non_saturated_conservative()
    check_shared_input()
    check_prefix_dependent_logging()
    check_feasible_set_and_later_abort()
    check_restricted_class()

"""Exact finite checks for Proposition A.5.1 (prop:supported; supported continuation) and
Corollary F.6.1 (cor:model-update; model updates).

Every check builds an explicit finite mechanism (named kernels with declared
inputs, in generation order), enumerates its law with fractions.Fraction, and
derives each asserted quantity from that enumeration.  Identification formulas
(the g-formula and likelihood weighting) receive only the baseline record law
and the declared input structure; they never read an unchanged kernel.

Run: python verify_supported.py


Paper map: Proposition A.5.1 (prop:supported) in its coordinate form, parts (i) to (v), and
Corollary F.6.1 (cor:model-update), parts (i) to (v). Supplementary checks:
check_hidden_baseline_input, check_hidden_replacement_input, and check_loss_value_not_enough
(hidden inputs: (G2) and (G4) cannot be dropped, and (G3) cannot be weakened to
the loss value); check_general_formula, check_propensity_conditioning, and
check_support_without_overlap (the likelihood-weighting identity, conditioning propensity on
later unaffected coordinates, and support without overlap); check_interface_includes_state
(the interface condition: comparing action alone vs action and memory); and
check_contribution_needs_recorded_loss (contribution without a recorded loss).
"""

from collections import defaultdict
from fractions import Fraction as F
from itertools import product


# --------------------------------------------------------------------------
# Finite mechanism engine
# --------------------------------------------------------------------------


class PositivityError(Exception):
    """A reached input of an unchanged descendant kernel has zero baseline probability."""


def bern(p):
    p = F(p)
    assert 0 <= p <= 1
    return {0: 1 - p, 1: p}


def point(value):
    return {value: F(1)}


def uniform(values):
    values = list(values)
    return {v: F(1, len(values)) for v in values}


class Mechanism:
    """Named kernels in declared generation order; kernel j reads declared earlier coordinates."""

    def __init__(self, nodes):
        self.nodes = [(name, tuple(ins), kern) for name, ins, kern in nodes]
        self.names = [name for name, _, _ in self.nodes]
        self.inputs = {name: ins for name, ins, _ in self.nodes}
        self.kernels = {name: kern for name, _, kern in self.nodes}
        seen = set()
        for name, ins, _ in self.nodes:
            assert set(ins) <= seen, f"{name} reads a coordinate generated later"
            seen.add(name)

    def replace(self, replacements):
        """Replace named kernels (and their declared inputs); every other kernel is unchanged."""
        nodes = []
        for name, ins, kern in self.nodes:
            if name in replacements:
                new_ins, new_kern = replacements[name]
                nodes.append((name, new_ins, new_kern))
            else:
                nodes.append((name, ins, kern))
        return Mechanism(nodes)

    def law(self):
        dist = {(): F(1)}
        for _, ins, kern in self.nodes:
            pos = [self.names.index(i) for i in ins]
            new = defaultdict(F)
            for path, p in dist.items():
                out = kern(*[path[j] for j in pos])
                assert sum(out.values()) == 1 and min(out.values()) >= 0
                for value, q in out.items():
                    if q:
                        new[path + (value,)] += p * q
            dist = dict(new)
        return dist

    def envs(self):
        return [(dict(zip(self.names, path)), p) for path, p in self.law().items()]


def marginal(names, dist, keep):
    pos = [names.index(n) for n in keep]
    out = defaultdict(F)
    for path, p in dist.items():
        out[tuple(path[j] for j in pos)] += p
    return {k: v for k, v in out.items() if v != 0}


def record_law(mech, record):
    """Baseline law of the recorded coordinates (the evaluator's population evidence)."""
    return marginal(mech.names, mech.law(), list(record))


def clean(d):
    return {k: v for k, v in d.items() if v != 0}


def prob(mech, event):
    return sum(p for env, p in mech.envs() if event(env))


def conditional_loss_law(mech, loss, cond=lambda env: True):
    num = defaultdict(F)
    den = F(0)
    for env, p in mech.envs():
        if cond(env):
            num[loss(env)] += p
            den += p
    assert den > 0
    return clean({b: v / den for b, v in num.items()})


def restricted_loss_measure(mech, loss, event, cond=lambda env: True):
    """B -> P(L in B, event | cond), as a measure on loss values."""
    num = defaultdict(F)
    den = F(0)
    for env, p in mech.envs():
        if cond(env):
            den += p
            if event(env):
                num[loss(env)] += p
    assert den > 0
    return clean({b: v / den for b, v in num.items()})


def descendants(base, D):
    """Dsc: replaced coordinates and every unchanged coordinate whose declared inputs meet Dsc."""
    dsc = set()
    for name in base.names:
        if name in D or any(i in dsc for i in base.inputs[name]):
            dsc.add(name)
    return dsc


def relevant_set(base, kappa, D, S_L, S_r):
    """Smallest J containing S_L and S_r and closed under kappa-inputs of its descendants."""
    dsc = descendants(base, D)
    J = set(S_L) | set(S_r)
    stack = list(J)
    while stack:
        name = stack.pop()
        if name in dsc:
            for i in kappa.inputs[name]:
                if i not in J:
                    J.add(i)
                    stack.append(i)
    return J, dsc


def conditions(base, kappa, D, S_L, record, S_r=(), r=(), known=None):
    """Mechanism-level truth values of (G1)-(G4) of Proposition prop:supported.

    known maps an unchanged kernel's name to (predicate, declared kernel): the
    kernel is declared at the inputs where the predicate holds (an inactive slot
    that emits a dummy value, or a post-abort valuation as in (A3) of Proposition A.4.1),
    and (G1) is required only at its other inputs.
    """
    known = known or {}
    J, dsc = relevant_set(base, kappa, D, S_L, S_r)
    rec = set(record)
    unchanged = [j for j in base.names if j in J and j in dsc and j not in D]
    cond_G2 = all(set(base.inputs[j]) <= rec for j in unchanged)
    cond_G3 = set(S_L) <= rec
    cond_G4 = all(set(kappa.inputs[j]) <= rec for j in D if j in J)
    law0, lawk = base.law(), kappa.law()
    cond_G1 = True
    for j in unchanged:
        ins = list(base.inputs[j])
        declared = known[j][0] if j in known else (lambda *z: False)
        reached = marginal(kappa.names, lawk, ins + list(S_r))
        seen = marginal(base.names, law0, ins)
        for key, p in reached.items():
            z = key[:len(ins)]
            if p > 0 and key[len(ins):] == tuple(r) and not declared(*z) and seen.get(z, 0) == 0:
                cond_G1 = False
    return {"G1": cond_G1, "G2": cond_G2, "G3": cond_G3, "G4": cond_G4}


def g_formula(record, rlaw, base, kappa, D, S_L, loss, S_r=(), r=(), known=None):
    """Proposition prop:supported(i): replacement loss law given R_t = r from the baseline record law.

    Uses only rlaw (the law of the recorded coordinates), the declared inputs in
    base, the declared kernel values in known, and the known replacement kernels
    in kappa.  Factors are evaluated in generation order and a path is dropped at
    its first zero factor; reaching an undeclared unchanged-kernel input with zero
    baseline probability raises PositivityError, which is the record-law test of
    (G1) in Proposition prop:supported(iii).
    """
    known = known or {}
    J, dsc = relevant_set(base, kappa, D, S_L, S_r)
    if not J <= set(record):
        raise ValueError("a relevant coordinate is unrecorded: (G2), (G3), or (G4) fails")
    order = [n for n in base.names if n in J]
    names = [n for n in order if n not in dsc]
    states = {}
    for vals, p in marginal(list(record), rlaw, names).items():
        env = dict(zip(names, vals))
        if p > 0 and tuple(env[n] for n in S_r) == tuple(r):
            states[vals] = p
    for j in [n for n in order if n in dsc]:
        new = defaultdict(F)
        if j in D:
            ins, kern = kappa.inputs[j], kappa.kernels[j]
            for vals, p in states.items():
                env = dict(zip(names, vals))
                for value, q in kern(*[env[i] for i in ins]).items():
                    if q:
                        new[vals + (value,)] += p * q
        else:
            ins = base.inputs[j]
            joint = marginal(list(record), rlaw, list(ins) + [j])
            marg = marginal(list(record), rlaw, list(ins))
            for vals, p in states.items():
                env = dict(zip(names, vals))
                pin = tuple(env[i] for i in ins)
                if j in known and known[j][0](*pin):
                    for value, q in known[j][1](*pin).items():
                        if q:
                            new[vals + (value,)] += p * q
                    continue
                if marg.get(pin, 0) == 0:
                    raise PositivityError(f"{j} reached at input {pin} with zero baseline probability")
                for key, q in joint.items():
                    if key[:-1] == pin:
                        new[vals + (key[-1],)] += p * q / marg[pin]
        names.append(j)
        states = dict(new)
    den = marginal(list(record), rlaw, list(S_r)).get(tuple(r), F(0))
    assert den > 0
    out = defaultdict(F)
    for vals, p in states.items():
        out[loss(dict(zip(names, vals)))] += p / den
    return clean(out)


def likelihood_weighting(record, rlaw, base, kappa, D, S_L, loss, S_r=(), r=()):
    """Proposition prop:supported(ii): B -> E^0[W 1{L in B} | R_t = r] with recorded propensities.

    The propensity of replaced decision j conditions on K_(j): the relevant
    unaffected coordinates (some generated after j) and the earlier relevant
    downstream ones.  The returned measure has total mass one when the overlap
    condition holds (a supplementary condition for likelihood weighting, not in the paper); otherwise replacement choices with zero propensity
    are lost and the total mass can fall below one.
    """
    J, dsc = relevant_set(base, kappa, D, S_L, S_r)
    assert J <= set(record)
    order = [n for n in base.names if n in J]
    JN = [n for n in order if n not in dsc]
    JD = [n for n in order if n in dsc]
    lawJ = marginal(list(record), rlaw, order)
    den = marginal(list(record), rlaw, list(S_r)).get(tuple(r), F(0))
    assert den > 0
    out = defaultdict(F)
    for vals, p in lawJ.items():
        env = dict(zip(order, vals))
        if tuple(env[n] for n in S_r) != tuple(r):
            continue
        weight = F(1)
        for idx, j in enumerate(JD):
            if j not in D:
                continue
            K = JN + JD[:idx]
            joint = marginal(list(record), rlaw, K + [j])
            marg = marginal(list(record), rlaw, K)
            u = tuple(env[i] for i in K)
            propensity = joint[u + (env[j],)] / marg[u]
            target = kappa.kernels[j](*[env[i] for i in kappa.inputs[j]]).get(env[j], F(0))
            weight *= target / propensity
        out[loss(env)] += p * weight / den
    return clean(out)


def agreement_event(f0, fk, inputs):
    """A_kappa = {f^kappa_k(I_k) = f^0_k(I_k) for every decision k}, a predicate on paths."""
    def event(env):
        return all(fk[a](*[env[i] for i in inputs[a]]) == f0[a](*[env[i] for i in inputs[a]]) for a in f0)
    return event


def recorded_agreement_event(fk, inputs):
    """Record version: f^kappa_k(logged I_k) equals the logged baseline output A_k."""
    def event(env):
        return all(fk[a](*[env[i] for i in inputs[a]]) == env[a] for a in fk)
    return event


def first_step_event(f0, fk, inputs, first):
    def event(env):
        return fk[first](*[env[i] for i in inputs[first]]) == f0[first](*[env[i] for i in inputs[first]])
    return event


# --------------------------------------------------------------------------
# P2(a): zero agreement, identified replacement risk
# --------------------------------------------------------------------------


def check_zero_agreement_identified():
    q = {0: F(1, 4), 1: F(3, 4)}

    def mech(pu, qq):
        return Mechanism([
            ("U", (), lambda: bern(pu)),                 # recorded exogenous tape
            ("A", ("U",), lambda u: point(u)),           # baseline A = f0(U) = U
            ("Y", ("A",), lambda a: bern(qq[a])),        # outcome kernel Q(Y | A), no U input
        ])

    f0 = {"A": lambda u: u}
    fk = {"A": lambda u: 1 - u}
    inputs = {"A": ("U",)}
    rep = {"A": (("U",), lambda u: point(fk["A"](u)))}
    record = ("U", "A", "Y")
    loss = lambda env: env["Y"]

    base = mech(F(1, 2), q)
    kappa = base.replace(rep)
    rlaw = record_law(base, record)
    assert conditions(base, kappa, {"A"}, ("Y",), record) == {"G1": True, "G2": True, "G3": True, "G4": True}

    mass = prob(base, agreement_event(f0, fk, inputs))
    mass_logged = prob(base, recorded_agreement_event(fk, inputs))
    assert mass == mass_logged == 0
    truth = conditional_loss_law(kappa, loss)
    g = g_formula(record, rlaw, base, kappa, {"A"}, ("Y",), loss)
    assert truth == g == {0: F(1, 2), 1: F(1, 2)}

    # Recorded-propensity weighting fails here: the propensity of the
    # replacement's choice A = 1 - U given U is zero, so the weighted measure is 0.
    propensity = {u: marginal(base.names, base.law(), ["U", "A"]).get((u, 1 - u), F(0)) for u in (0, 1)}
    assert propensity == {0: 0, 1: 0}
    assert sum(likelihood_weighting(record, rlaw, base, kappa, {"A"}, ("Y",), loss).values()) == 0

    # Saturated contrast: the same record law with the outcome kernel declared on
    # (U, A).  Inputs (u, 1-u) have zero baseline probability; the compatible
    # replacement risks fill [0, 1] and the g-formula raises the support error.
    def mech_sat(qq):
        return Mechanism([
            ("U", (), lambda: bern(F(1, 2))),
            ("A", ("U",), lambda u: point(u)),
            ("Y", ("U", "A"), lambda u, a: bern(qq[(u, a)])),
        ])

    grid = [F(i, 4) for i in range(5)]
    sat_risks = set()
    for x01, x10 in product(grid, repeat=2):
        m = mech_sat({(0, 0): q[0], (1, 1): q[1], (0, 1): x01, (1, 0): x10})
        assert record_law(m, record) == rlaw
        sat_risks.add(conditional_loss_law(m.replace(rep), loss).get(1, F(0)))
    assert min(sat_risks) == 0 and max(sat_risks) == 1
    sat = mech_sat({(0, 0): q[0], (1, 1): q[1], (0, 1): F(1, 2), (1, 0): F(1, 2)})
    assert conditions(sat, sat.replace(rep), {"A"}, ("Y",), record)["G1"] is False
    try:
        g_formula(record, rlaw, sat, sat.replace(rep), {"A"}, ("Y",), loss)
        raise AssertionError("support failure not detected")
    except PositivityError:
        pass

    # A biased tape separates baseline and replacement risk.
    biased = mech(F(1, 4), q)
    base_risk = conditional_loss_law(biased, loss)[1]
    g_biased = g_formula(record, record_law(biased, record), biased, biased.replace(rep), {"A"}, ("Y",), loss)
    assert base_risk == F(3, 8)
    assert g_biased == conditional_loss_law(biased.replace(rep), loss) and g_biased[1] == F(5, 8)

    print("PASS P2(a) zero-agreement identified case: agreement mass 0; (G1)-(G4) hold; g-formula and replacement "
          "law both give P(Y=1)=1/2; declaring Q on (U,A) instead gives [0,1] and a support error; "
          "recorded-propensity weighting has mass 0; tape P(U=1)=1/4 gives 3/8 -> 5/8.")


# --------------------------------------------------------------------------
# P2(b): hidden-input witnesses
# --------------------------------------------------------------------------


def check_hidden_baseline_input():
    """(G2) fails: the baseline assignment reads an unrecorded H that the outcome kernel also reads."""
    def mech(outcome):
        return Mechanism([
            ("H", (), lambda: bern(F(1, 2))),            # unrecorded context
            ("W", (), lambda: bern(F(1, 4))),            # unrecorded tape
            ("A", ("H", "W"), lambda h, w: point(h ^ w)),  # baseline assignment uses H
            ("Y", ("A", "H"), outcome),
        ])

    m1 = mech(lambda a, h: point(h))                     # Y = H
    m2 = mech(lambda a, h: bern(F(1, 4) + F(a, 2)))      # Y depends on A only
    record = ("A", "Y")
    rep = {"A": ((), lambda: point(1))}                  # replacement: always A = 1
    loss = lambda env: env["Y"]
    assert record_law(m1, record) == record_law(m2, record)
    for m in (m1, m2):
        assert conditions(m, m.replace(rep), {"A"}, ("Y",), record) == {"G1": True, "G2": False, "G3": True, "G4": True}
    law1 = conditional_loss_law(m1.replace(rep), loss)
    law2 = conditional_loss_law(m2.replace(rep), loss)
    assert law1 == {0: F(1, 2), 1: F(1, 2)} and law2 == {0: F(1, 4), 1: F(3, 4)}

    # Misdeclaring Y's input as A alone and applying the g-formula returns 3/4
    # for both: correct for m2, wrong for m1.
    misdeclared = Mechanism([("H", (), None), ("W", (), None), ("A", ("H", "W"), None), ("Y", ("A",), None)])
    for m in (m1, m2):
        naive = g_formula(record, record_law(m, record), misdeclared, misdeclared.replace(rep), {"A"}, ("Y",), loss)
        assert naive == law2
    print("PASS P2(b1) hidden baseline input: equal record laws of (A,Y); (G1),(G3),(G4) hold and (G2) fails; "
          "replacement A=1 gives P(Y=1)=1/2 vs 3/4; the g-formula with Y declared on A alone gives 3/4 for both.")


def check_hidden_replacement_input():
    """(G4) fails: the replacement assignment reads an unrecorded H."""
    def mech(ph):
        return Mechanism([
            ("H", (), lambda: bern(ph)),                 # unrecorded, read by nothing in the baseline
            ("U", (), lambda: bern(F(1, 2))),            # recorded tape
            ("A", ("U",), lambda u: point(u)),
            ("Y", ("A",), lambda a: bern(F(1, 4) + F(a, 2))),
        ])

    m1, m2 = mech(F(1, 4)), mech(F(3, 4))
    record = ("U", "A", "Y")
    rep = {"A": (("H",), lambda h: point(h))}
    loss = lambda env: env["Y"]
    assert record_law(m1, record) == record_law(m2, record)
    for m in (m1, m2):
        assert conditions(m, m.replace(rep), {"A"}, ("Y",), record) == {"G1": True, "G2": True, "G3": True, "G4": False}
    law1 = conditional_loss_law(m1.replace(rep), loss)
    law2 = conditional_loss_law(m2.replace(rep), loss)
    assert law1[1] == F(3, 8) and law2[1] == F(5, 8)
    print("PASS P2(b2) hidden replacement input: equal record laws of (U,A,Y); (G1),(G2),(G3) hold and (G4) fails; "
          "the replacement A=H gives P(Y=1)=3/8 vs 5/8.")


# --------------------------------------------------------------------------
# Corollary: shared seed versus fresh sample
# --------------------------------------------------------------------------


def check_shared_seed_vs_fresh_sample():
    tape = bern(F(1, 2))
    f0 = lambda u: u
    impls = {"f1": lambda u: u, "f2": lambda u: 1 - u}

    def sample_only_kernel(f):
        out = defaultdict(F)
        for u, p in tape.items():
            out[f(u)] += p
        return clean(out)

    kernels = {k: sample_only_kernel(f) for k, f in impls.items()}
    assert kernels["f1"] == kernels["f2"] == {0: F(1, 2), 1: F(1, 2)}
    shared = {k: sum(p for u, p in tape.items() if f(u) == f0(u)) for k, f in impls.items()}
    fresh = {k: sum(p * q for (u, p), (v, q) in product(tape.items(), repeat=2) if f(v) == f0(u))
             for k, f in impls.items()}
    assert shared == {"f1": 1, "f2": 0}
    assert fresh == {"f1": F(1, 2), "f2": F(1, 2)}

    # Inside a mechanism: identical replacement loss laws, agreement masses 1 and 0.
    q = {0: F(1, 4), 1: F(3, 4)}
    base = Mechanism([
        ("U", (), lambda: tape),
        ("A", ("U",), lambda u: point(f0(u))),
        ("Y", ("A",), lambda a: bern(q[a])),
    ])
    laws, masses = {}, {}
    for k, f in impls.items():
        kappa = base.replace({"A": (("U",), lambda u, f=f: point(f(u)))})
        laws[k] = conditional_loss_law(kappa, lambda env: env["Y"])
        masses[k] = prob(base, recorded_agreement_event({"A": f}, {"A": ("U",)}))
    assert laws["f1"] == laws["f2"] and masses == {"f1": 1, "f2": 0}
    print("PASS Cor shared-seed vs fresh-sample: f1(U)=U and f2(U)=1-U have the same sample-only kernel "
          "Bernoulli(1/2) and the same replacement loss law; shared-seed agreement with f0 is 1 and 0; "
          "fresh-sample agreement is 1/2 for both.")


# --------------------------------------------------------------------------
# Corollary: agreement restriction, first-step bound, estimation
# --------------------------------------------------------------------------


def check_first_step_bound():
    f0 = {
        "A1": lambda x, u: (x + u) % 2,
        "A2": lambda z, u: z if u < 2 else 1 - z,
    }
    fk = {
        "A1": lambda x, u: (x + u) % 2 if u != 3 else 1 - (x + u) % 2,
        "A2": lambda z, u: 1 - f0["A2"](z, u) if (u == 1 and z == 1) else f0["A2"](z, u),
    }
    inputs = {"A1": ("X", "U1"), "A2": ("Z", "U2")}
    base = Mechanism([
        ("X", (), lambda: uniform((0, 1))),                       # pre-intervention record
        ("U1", (), lambda: uniform(range(4))),                    # tape, unchanged random-input kernel
        ("A1", ("X", "U1"), lambda x, u: point(f0["A1"](x, u))),  # policy step 1
        ("Z", ("A1",), lambda a: bern(F(1, 3) + F(a, 3))),        # environment response
        ("U2", (), lambda: uniform(range(4))),                    # tape
        ("A2", ("Z", "U2"), lambda z, u: point(f0["A2"](z, u))),  # policy step 2
        ("Y", ("A2", "Z"), lambda a, z: bern(F(1 + a + z, 4))),   # outcome
    ])
    kappa = base.replace({a: (inputs[a], lambda *i, a=a: point(fk[a](*i))) for a in fk})
    loss = lambda env: (env["Y"], env["A1"] + env["A2"])
    agree = agreement_event(f0, fk, inputs)
    agree_logged = recorded_agreement_event(fk, inputs)
    first = first_step_event(f0, fk, inputs, "A1")

    # The logged agreement event equals A_kappa on the baseline support.
    assert all(agree(env) == agree_logged(env) for env, _ in base.envs())
    # Under the replacement law the path coordinate A_k equals f^kappa_k(I_k), so the
    # logged predicate is trivially true there; A_kappa must compare with f^0_k(I_k).
    assert prob(kappa, agree_logged) == 1 and prob(kappa, agree) == prob(base, agree) < 1

    for r in (0, 1):
        given = lambda env, r=r: env["X"] == r
        # Proposition A.4.1(i) restricted to agreement: equal restricted measures.
        restricted_kappa = restricted_loss_measure(kappa, loss, agree, given)
        restricted_base = restricted_loss_measure(base, loss, agree, given)
        assert restricted_kappa == restricted_base
        m_kappa = sum(restricted_kappa.values())
        m_logged = prob(base, lambda env: given(env) and agree_logged(env)) / prob(base, given)
        m_first = prob(base, lambda env: given(env) and first(env)) / prob(base, given)
        m_step = [m_first, m_kappa]
        assert m_kappa == m_logged
        assert m_first == F(3, 4)
        assert m_step[1] < m_step[0], (r, m_step)

    # Empirical agreement frequency among the episodes with X = r, from n = 3 i.i.d.
    # baseline episodes, conditioned on N_r.  Each episode enters only through
    # (X, logged agreement), whose law is read off the enumeration.
    n = 3
    summary = defaultdict(F)
    for env, p in base.envs():
        summary[(env["X"], agree_logged(env))] += p
    m_by_r = {}
    for r in (0, 1):
        m_r = summary[(r, True)] / (summary[(r, True)] + summary[(r, False)])
        m_by_r[r] = m_r
        for k in range(1, n + 1):
            p_k = mean = second = F(0)
            for draws in product(summary.items(), repeat=n):
                p = F(1)
                for _, q in draws:
                    p *= q
                selected = [ag for (x, ag), _ in draws if x == r]
                if len(selected) != k:
                    continue
                freq = F(sum(selected), k)
                p_k += p
                mean += p * freq
                second += p * freq * freq
            mean, second = mean / p_k, second / p_k
            assert mean == m_r and second - mean * mean == m_r * (1 - m_r) / k, (r, k)
    print(f"PASS Cor first-step agreement bound: two-step seeded policy; restricted measures agree under "
          f"replacement and baseline for X in {{0,1}}; agreement given X=0,1 is {m_by_r[0]}, {m_by_r[1]} "
          f"< first-decision agreement 3/4; with n=3 i.i.d. episodes, the frequency among those with X=r "
          f"has conditional mean m(r) and variance m(r)(1-m(r))/N_r for N_r=1,2,3.")


# --------------------------------------------------------------------------
# Corollary: the agreement test must compare the full interface value
# --------------------------------------------------------------------------


def check_interface_includes_state():
    f0 = {"A": lambda u: (u, 0)}          # (action, memory write)
    fk = {"A": lambda u: (u, u)}          # same action, different memory write
    inputs = {"A": ("U",)}
    base = Mechanism([
        ("U", (), lambda: bern(F(1, 2))),
        ("A", ("U",), lambda u: point(f0["A"](u))),
        ("Z", ("A",), lambda a: bern(F(1, 4) + F(a[0], 2))),    # environment reads the action
        ("Y", ("A", "Z"), lambda a, z: point(a[1] + z)),        # later kernel reads the memory
    ])
    kappa = base.replace({"A": (("U",), lambda u: point(fk["A"](u)))})
    loss = lambda env: env["Y"]
    action_only = lambda env: fk["A"](env["U"])[0] == f0["A"](env["U"])[0]
    full = agreement_event(f0, fk, inputs)
    assert prob(base, action_only) == 1
    assert restricted_loss_measure(kappa, loss, action_only) != restricted_loss_measure(base, loss, action_only)
    assert prob(base, full) == F(1, 2)
    assert restricted_loss_measure(kappa, loss, full) == restricted_loss_measure(base, loss, full)
    print("PASS Cor interface includes dynamic state: comparing only the action gives agreement mass 1 but "
          "unequal restricted measures; comparing (action, memory) gives mass 1/2 and equal restricted measures.")


# --------------------------------------------------------------------------
# Corollary (contract deviation): the contribution needs a recorded loss
# --------------------------------------------------------------------------


def check_contribution_needs_recorded_loss():
    def mech(q):
        return Mechanism([
            ("U", (), lambda: bern(F(1, 2))),
            ("A", ("U",), lambda u: point(u)),
            ("Z", ("A",), lambda a: bern(q[a])),       # unrecorded consequence; L = Z
        ])

    m1 = mech({0: F(1, 4), 1: F(3, 4)})
    m2 = mech({0: F(3, 4), 1: F(1, 4)})
    record = ("U", "A")
    f0, fk, inputs = {"A": lambda u: u}, {"A": lambda u: 0}, {"A": ("U",)}
    rep = {"A": (("U",), lambda u: point(0))}
    loss = lambda env: env["Z"]
    assert record_law(m1, record) == record_law(m2, record)
    masses = [prob(m, recorded_agreement_event(fk, inputs)) for m in (m1, m2)]
    assert masses == [F(1, 2), F(1, 2)]
    contributions = []
    for m in (m1, m2):
        agree = agreement_event(f0, fk, inputs)
        c_kappa = restricted_loss_measure(m.replace(rep), loss, agree)
        c_base = restricted_loss_measure(m, loss, agree)
        assert c_kappa == c_base
        contributions.append(c_base.get(1, F(0)))
    assert contributions == [F(1, 8), F(3, 8)]
    print("PASS Cor contribution needs a recorded loss: equal record laws of (U,A) and agreement mass 1/2, "
          "but P(L=1, agreement) is 1/8 vs 3/8 when L is unrecorded.")


# --------------------------------------------------------------------------
# P2(i)-(ii): general g-formula and weighting with hidden confounding
# --------------------------------------------------------------------------


GENERAL_SETTINGS = list(product((F(1, 3), F(1, 2)), (F(1, 5), F(1, 4))))
GENERAL_REP = {"A": (("X", "U"), lambda x, u: point(x & u))}


def general_mechanism(ph, pw):
    """Hidden H drives the baseline decision, the record X, and a post-decision unaffected Z."""
    return Mechanism([
        ("H", (), lambda: bern(ph)),                               # unrecorded
        ("X", ("H",), lambda h: bern(F(2, 3) if h else F(1, 4))),  # pre-intervention record
        ("U", (), lambda: bern(F(1, 2))),                          # recorded tape
        ("W", (), lambda: bern(pw)),                               # unrecorded tape
        ("A", ("H", "W"), lambda h, w: point(h ^ w)),              # baseline reads hidden H
        ("Z", ("H",), lambda h: bern(F(3, 4) if h else F(1, 5))),  # post-boundary non-descendant
        ("Y", ("A", "X", "Z"), lambda a, x, z: bern(F(1 + a + x + 2 * z, 6))),
    ])


def check_general_formula():
    naive_differs = 0
    settings = 0
    for ph, pw in GENERAL_SETTINGS:
        base = general_mechanism(ph, pw)
        rep = GENERAL_REP
        kappa = base.replace(rep)
        record = ("X", "U", "A", "Z", "Y")
        S_L = ("Y", "A")
        loss = lambda env: (env["Y"], env["A"])               # vector loss (harm, action cost)
        rlaw = record_law(base, record)
        J, dsc = relevant_set(base, kappa, {"A"}, S_L, ("X",))
        assert "Z" in J and "Z" not in dsc and "H" in base.inputs["Z"] and "H" not in record
        misdeclared = Mechanism([(n, ("A", "X") if n == "Y" else base.inputs[n], None) for n in base.names])
        for r in (0, 1):
            assert conditions(base, kappa, {"A"}, S_L, record, ("X",), (r,)) == \
                {"G1": True, "G2": True, "G3": True, "G4": True}
            given = lambda env, r=r: env["X"] == r
            truth = conditional_loss_law(kappa, loss, given)
            g = g_formula(record, rlaw, base, kappa, {"A"}, S_L, loss, ("X",), (r,))
            lw = likelihood_weighting(record, rlaw, base, kappa, {"A"}, S_L, loss, ("X",), (r,))
            assert truth == g == lw
            naive = g_formula(record, rlaw, misdeclared, misdeclared.replace(rep), {"A"}, S_L, loss, ("X",), (r,))
            naive_differs += naive != truth
            settings += 1
    assert naive_differs == settings
    print(f"PASS P2(i)-(ii) general formula: {settings} parameter/record settings with hidden H driving the baseline "
          f"decision, X, and a post-boundary non-descendant Z; vector-loss g-formula = likelihood weighting = "
          f"replacement law in all; omitting Z from Y's declared inputs is wrong in all {settings}.")


def check_support_without_overlap():
    """(G1) holds while overlap fails: the loss reads the replaced decision directly (abort)."""
    abort_value = F(1, 2)
    base = Mechanism([
        ("X", (), lambda: bern(F(1, 2))),                         # recorded risk signal
        ("A", ("X",), lambda x: point(0)),                        # baseline never aborts
        ("D", ("X",), lambda x: bern(F(1, 4) + F(x, 2))),         # damage if executed; reads X only
    ])
    rep = {"A": (("X",), lambda x: point(x))}                     # abort when the signal is 1
    kappa = base.replace(rep)
    record = ("X", "A", "D")
    loss = lambda env: env["D"] if env["A"] == 0 else abort_value
    rlaw = record_law(base, record)
    assert conditions(base, kappa, {"A"}, ("A", "D"), record) == {"G1": True, "G2": True, "G3": True, "G4": True}
    truth = conditional_loss_law(kappa, loss)
    g = g_formula(record, rlaw, base, kappa, {"A"}, ("A", "D"), loss)
    lw = likelihood_weighting(record, rlaw, base, kappa, {"A"}, ("A", "D"), loss)
    assert truth == g == {0: F(3, 8), 1: F(1, 8), abort_value: F(1, 2)}
    assert marginal(base.names, base.law(), ["A"]).get((1,), F(0)) == 0
    assert sum(lw.values()) == F(1, 2) and abort_value not in lw
    print("PASS P2 support without overlap: abort never occurs at baseline; the g-formula recovers the "
          "replacement law (3/8, 1/8, 1/2 on 0, 1, abort value) while recorded-propensity weighting keeps mass 1/2.")


def check_propensity_conditioning():
    """Remark: the propensity must condition on K_(j), not on the recorded past alone."""
    past_wrong = 0
    settings = 0
    for ph, pw in GENERAL_SETTINGS:
        base = general_mechanism(ph, pw)
        kappa = base.replace(GENERAL_REP)
        record = ("X", "U", "A", "Z", "Y")
        rlaw = record_law(base, record)
        loss = lambda env: (env["Y"], env["A"])
        J, dsc = relevant_set(base, kappa, {"A"}, ("Y", "A"), ("X",))
        K_A = [n for n in base.names if n in J and n not in dsc]
        assert K_A == ["X", "U", "Z"] and base.names.index("Z") > base.names.index("A")
        past_joint = marginal(list(record), rlaw, ["X", "U", "A"])
        past_marg = marginal(list(record), rlaw, ["X", "U"])
        for r in (0, 1):
            given = lambda env, r=r: env["X"] == r
            truth = conditional_loss_law(kappa, loss, given)
            lw = likelihood_weighting(record, rlaw, base, kappa, {"A"}, ("Y", "A"), loss, ("X",), (r,))
            assert lw == truth
            out = defaultdict(F)
            den = F(0)
            for env, p in base.envs():
                if not given(env):
                    continue
                den += p
                e_past = past_joint[(env["X"], env["U"], env["A"])] / past_marg[(env["X"], env["U"])]
                target = GENERAL_REP["A"][1](env["X"], env["U"]).get(env["A"], F(0))
                out[loss(env)] += p * target / e_past
            past = clean({k: v / den for k, v in out.items()})
            past_wrong += past != truth
            settings += 1
    assert past_wrong == settings
    print(f"PASS P2(ii) propensity conditioning: K_(A) = (X, U, Z) contains Z, generated after the decision; "
          f"weighting with e_A given K_(A) equals the replacement law in all {settings} settings, while "
          f"weighting with P0(A | X, U), the recorded past, is wrong in all {settings}.")


def check_loss_value_not_enough():
    """(G3) cannot be weakened to recording the loss value; the agreement contribution alone can use it."""
    def mech(d1_when_h0):
        return Mechanism([
            ("H", (), lambda: bern(F(1, 2))),                                # unrecorded context
            ("A", ("H",), lambda h: point(h)),                               # gate: 1 execute, 0 abort
            ("D1", ("H",), lambda h: point(0) if h else point(d1_when_h0)),  # damage if executed
        ])

    loss = lambda env: env["D1"] if env["A"] == 1 else F(1, 2)
    rep = {"A": ((), lambda: point(1))}                                      # always execute
    m1, m2 = mech(0), mech(1)

    def value_record_law(m):
        out = defaultdict(F)
        for env, p in m.envs():
            out[(env["A"], loss(env))] += p
        return dict(out)

    assert value_record_law(m1) == value_record_law(m2)
    law1 = conditional_loss_law(m1.replace(rep), loss)
    law2 = conditional_loss_law(m2.replace(rep), loss)
    assert law1 == {0: F(1)} and law2 == {0: F(1, 2), 1: F(1, 2)}
    for m in (m1, m2):
        assert conditions(m, m.replace(rep), {"A"}, ("A", "D1"), ("A",)) == \
            {"G1": True, "G2": True, "G3": False, "G4": True}
        # With the loss inputs (A, D1) recorded, (G1)-(G4) hold and the g-formula is exact.
        record = ("A", "D1")
        assert conditions(m, m.replace(rep), {"A"}, ("A", "D1"), record) == \
            {"G1": True, "G2": True, "G3": True, "G4": True}
        g = g_formula(record, record_law(m, record), m, m.replace(rep), {"A"}, ("A", "D1"), loss)
        assert g == conditional_loss_law(m.replace(rep), loss)
    # The agreement contribution needs only the loss value: agreement is {A = 1} on the
    # baseline support, and P0(L in ., A = 1) is a functional of the law of (A, L).
    agree = lambda env: env["A"] == 1
    contributions = []
    for m in (m1, m2):
        c_kappa = restricted_loss_measure(m.replace(rep), loss, lambda env: env["H"] == 1)
        c_base = restricted_loss_measure(m, loss, agree)
        assert c_kappa == c_base
        contributions.append(c_base)
    assert contributions[0] == contributions[1] == {0: F(1, 2)}
    print("PASS P2 (G3) needs the loss inputs: equal record laws of (A, loss value), (G1),(G2),(G4) hold, and "
          "'always execute' gives P(L=1) = 0 vs 1/2; recording (A, D1) makes the g-formula exact; the agreement "
          "contribution {0: 1/2} is the same in both, so the agreement contribution needs only the loss value.")


def check_abort_composition():
    """Declared inputs make (G1) compose with Proposition A.4.1(ii) when later kernels read the gate."""
    DUMMY = 9
    EXEC, ABORT, MODIFY = 0, 1, 2

    # (a) The baseline never aborts; the environment reads the gate and is inactive after an abort.
    base = Mechanism([
        ("X", (), lambda: bern(F(1, 2))),
        ("A", ("X",), lambda x: point(EXEC)),
        ("E", ("A", "X"), lambda a, x: point(DUMMY) if a == ABORT else bern(F(1, 4) + F(x, 2))),
    ])
    rep = {"A": (("X",), lambda x: point(ABORT if x else EXEC))}
    kappa = base.replace(rep)
    record = ("X", "A", "E")
    S_L = ("A", "E")
    loss = lambda env: F(1, 2) if env["A"] == ABORT else env["E"]
    rlaw = record_law(base, record)
    inactive = {"E": (lambda a, x: a == ABORT, lambda a, x: point(DUMMY))}
    assert conditions(base, kappa, {"A"}, S_L, record)["G1"] is False
    try:
        g_formula(record, rlaw, base, kappa, {"A"}, S_L, loss)
        raise AssertionError("support failure not detected")
    except PositivityError:
        pass
    assert conditions(base, kappa, {"A"}, S_L, record, known=inactive)["G1"] is True
    truth = conditional_loss_law(kappa, loss)
    g = g_formula(record, rlaw, base, kappa, {"A"}, S_L, loss, known=inactive)
    agree_a = lambda env: env["X"] == 0
    replay = restricted_loss_measure(base, loss, agree_a)
    terminal = {F(1, 2): prob(base, lambda env: env["X"] == 1)}
    assert truth == g == {0: F(3, 8), 1: F(1, 8), F(1, 2): F(1, 2)}
    assert all(g.get(k, 0) == replay.get(k, 0) + terminal.get(k, 0) for k in set(g) | set(replay) | set(terminal))

    # (b) Agreement, terminal, and continuing paths together, with a post-abort valuation slot C.
    f0 = {0: EXEC, 1: MODIFY, 2: ABORT}
    fk = {0: MODIFY, 1: ABORT, 2: ABORT}
    b1 = {F(1, 2): F(1, 2), 2: F(1, 2)}                    # declared valuation after the terminal abort

    def c_kernel(a, x):
        if a != ABORT:
            return point(DUMMY)                                # inactive slot
        return b1 if x == 1 else point(F(1, 2))                # x = 2: the baseline's own post-abort law

    base = Mechanism([
        ("X", (), lambda: uniform((0, 1, 2))),
        ("A", ("X",), lambda x: point(f0[x])),
        ("E", ("A",), lambda a: point(DUMMY) if a == ABORT else bern(F(1, 4) if a == EXEC else F(3, 4))),
        ("C", ("A", "X"), c_kernel),
    ])
    rep = {"A": (("X",), lambda x: point(fk[x]))}
    kappa = base.replace(rep)
    record = ("X", "A", "E", "C")
    S_L = ("A", "E", "C")
    loss = lambda env: env["C"] if env["A"] == ABORT else env["E"]
    rlaw = record_law(base, record)
    declared = {
        "E": (lambda a: a == ABORT, lambda a: point(DUMMY)),
        "C": (lambda a, x: a != ABORT or x == 1, lambda a, x: point(DUMMY) if a != ABORT else b1),
    }
    assert conditions(base, kappa, {"A"}, S_L, record)["G1"] is False
    assert conditions(base, kappa, {"A"}, S_L, record, known=declared) == \
        {"G1": True, "G2": True, "G3": True, "G4": True}
    truth = conditional_loss_law(kappa, loss)
    g = g_formula(record, rlaw, base, kappa, {"A"}, S_L, loss, known=declared)
    assert g == truth
    status = lambda env: ("agree" if fk[env["X"]] == f0[env["X"]] else
                          "terminal" if fk[env["X"]] == ABORT else "continuing")
    replay = restricted_loss_measure(base, loss, lambda env: status(env) == "agree")
    p_term = prob(base, lambda env: status(env) == "terminal")
    terminal = {k: p_term * v for k, v in b1.items()}
    continuing = clean({k: g.get(k, 0) - replay.get(k, 0) - terminal.get(k, 0) for k in g})
    assert continuing == restricted_loss_measure(kappa, loss, lambda env: status(env) == "continuing")
    assert sum(replay.values()) == sum(terminal.values()) == sum(continuing.values()) == F(1, 3)
    # Restricting to paths on which the replacement never aborts loses the agreement-with-abort mass.
    no_abort = prob(kappa, lambda env: env["A"] != ABORT)
    assert no_abort + p_term == F(2, 3)
    print("PASS P2 abort composition: (a) the environment reads the gate and the baseline never aborts; "
          "(G1) fails unless the inactive slot is declared, and then the g-formula gives (3/8, 1/8, 1/2) = "
          "agreement replay + Proposition A.4.1(ii); (b) agreement, terminal, and continuing paths each of mass 1/3: "
          "g-formula = replacement law, and its continuing part is the replacement law there; paths without a "
          "replacement abort plus terminal paths carry only 2/3.")


if __name__ == "__main__":
    check_zero_agreement_identified()
    check_hidden_baseline_input()
    check_hidden_replacement_input()
    check_shared_seed_vs_fresh_sample()
    check_first_step_bound()
    check_interface_includes_state()
    check_contribution_needs_recorded_loss()
    check_general_formula()
    check_propensity_conditioning()
    check_support_without_overlap()
    check_abort_composition()
    check_loss_value_not_enough()

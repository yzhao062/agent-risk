#!/usr/bin/env python
"""Exact finite checks for Proposition F.5.2 (prop:r4-unsupported; continuation from records
with unsupported inputs), a generalization of Proposition A.5.1 (prop:supported) that drops (G1).

Every check builds explicit finite mechanisms (named kernels with declared
inputs, in generation order) and computes every reported number in exact
rational arithmetic (fractions.Fraction).  The generalized g-formula
`g_lambda_formula` receives only the baseline record law, the declared inputs,
the declared kernel values, the known replacement kernels, and a family lambda
of laws at the null inputs; it never reads an unchanged kernel.  Brute-force
identified sets enumerate grid mechanisms, keep those whose baseline record law
equals the observed one, and drop those that leave a declared allowed set; they
never use the null-input characterization.

The only import from the paper is checks/verify_supported.py (mechanism engine,
record laws, relevant set, condition test, and the (G1) g-formula), loaded by
path for cross-checks.  The script looks for it next to itself first, so it can
be copied into checks/ unchanged.

Notation follows the paper: lambda (lam in code) is a family of free
laws at the null blocks; P["allowed"] holds the allowed sets E_{j,z}; X_h is the
prefix-dependent feasible set of Proposition E.1.1 (prop:replay-cont).

Run: python verify_unsupported.py


Paper map: Proposition F.5.2 (prop:r4-unsupported) has parts (a) to (c), and Remarks
F.5.3 (rem:r4-logs) and F.5.4 (rem:r4-scope) follow. Supplementary checks:
check_nonconvex_image (the non-convex image check, documented in checks/README.md),
check_shared_input_t114 (shared null inputs), check_saturated_extreme, check_part_f,
and check_pivot_without_onto (saturated extreme checks), and check_block_and_retry
(block and retry with mixed support: identified harm [43/192, 91/192] inside
the outer bound [9/64, 41/64]; treating the supported input (a, err) as null gives [1/8, 43/64]).
"Theorem thm:replay(iii)" in the messages below is Proposition E.1.1 (prop:replay-cont).
"""

import importlib.util
import random
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path

_VS_PATH = Path(__file__).resolve().parent / "verify_supported.py"
_spec = importlib.util.spec_from_file_location("verify_supported", _VS_PATH)
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)

Mechanism, marginal, record_law = vs.Mechanism, vs.marginal, vs.record_law
HALF = F(1, 2)
THIRD = F(1, 3)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def clean(d):
    return {k: v for k, v in d.items() if v != 0}


def point(value):
    return {value: F(1)}


def bern(p):
    p = F(p)
    return clean({0: 1 - p, 1: p})


def uniform_on(values):
    values = list(values)
    return {v: F(1, len(values)) for v in values}


def law_add(*laws):
    out = defaultdict(F)
    for d in laws:
        for k, v in d.items():
            out[k] += v
    return clean(out)


def law_scale(d, c):
    return clean({k: c * v for k, v in d.items()})


def push(d, f):
    out = defaultdict(F)
    for k, v in d.items():
        out[f(k)] += v
    return clean(out)


def prob_of(d, B):
    return sum((v for k, v in d.items() if k in B), F(0))


def mean_of(d):
    return sum((k * v for k, v in d.items()), F(0))


def frozen(d):
    return frozenset(clean(d).items())


def mech_from_tables(names, inputs, tables):
    return Mechanism([(n, inputs[n], (lambda *z, t=tables[n]: t[tuple(z)])) for n in names])


def known_from_declared(declared):
    """Declared kernel values in the (predicate, kernel) form of verify_supported."""
    return {j: (lambda *z, d=dec: tuple(z) in d, lambda *z, d=dec: d[tuple(z)]) for j, dec in declared.items()}


def given_r(P):
    S_r, r = tuple(P["S_r"]), tuple(P["r"])
    return lambda env: tuple(env[n] for n in S_r) == r


def modified_tables(tables, lam):
    """M^lambda: the tables with kernel j set to lambda_{j,z} at every null block (j, z)."""
    out = {n: dict(t) for n, t in tables.items()}
    for (j, z), law in lam.items():
        out[j][z] = dict(law)
    return out


# --------------------------------------------------------------------------
# The generalized g-formula G^lambda and the null blocks
# --------------------------------------------------------------------------


def relevant(P):
    return vs.relevant_set(P["base"], P["kappa"], set(P["D"]), P["S_L"], P["S_r"])


def null_blocks(P, rlaw):
    """B = {(j, z): j in U_kappa = J cap (Delta minus D), z undeclared, P0(V_{S_j} = z) = 0}, from the record law."""
    base, record, declared = P["base"], P["record"], P.get("declared", {})
    J, dsc = relevant(P)
    blocks = []
    for j in base.names:
        if j in J and j in dsc and j not in P["D"]:
            ins = base.inputs[j]
            assert set(ins) <= set(record), "(G2) fails"
            marg = marginal(list(record), rlaw, list(ins))
            for z in product(*[P["ranges"][i] for i in ins]):
                if z not in declared.get(j, {}) and marg.get(z, F(0)) == 0:
                    blocks.append((j, z))
    return blocks


def allowed(P, j, z):
    return tuple(P.get("allowed", {}).get((j, z), P["ranges"][j]))


def g_lambda_formula(P, rlaw, lam, force_free=()):
    """Q^lambda_t(. | r): eq:p2-factors with the fourth case c_j(v) = lambda_{j, v_{S_j}}(v_j) at null inputs.

    Uses only rlaw (the baseline record law), the declared inputs (base.inputs),
    the declared kernel values, the replacement kernels in kappa, and lambda.
    Blocks in force_free are treated as null even if the record law supports
    them (used for the conservative-enlargement check of Remark r4-logs).
    """
    base, kappa, D = P["base"], P["kappa"], set(P["D"])
    record, S_L, S_r, r = P["record"], P["S_L"], tuple(P["S_r"]), tuple(P["r"])
    declared = P.get("declared", {})
    J, dsc = vs.relevant_set(base, kappa, D, S_L, S_r)
    assert J <= set(record), "(G2), (G3), or (G4) fails"
    order = [n for n in base.names if n in J]
    names = [n for n in order if n not in dsc]
    states = {}
    for vals, p in marginal(list(record), rlaw, names).items():
        env = dict(zip(names, vals))
        if tuple(env[n] for n in S_r) == r:
            states[vals] = p
    for j in [n for n in order if n in dsc]:
        new = defaultdict(F)
        if j in D:
            ins, kern = kappa.inputs[j], kappa.kernels[j]

            def factor(env, ins=ins, kern=kern):
                return kern(*[env[i] for i in ins])
        else:
            ins = base.inputs[j]
            joint = marginal(list(record), rlaw, list(ins) + [j])
            marg = marginal(list(record), rlaw, list(ins))
            dec = declared.get(j, {})

            def factor(env, ins=ins, joint=joint, marg=marg, dec=dec, j=j):
                z = tuple(env[i] for i in ins)
                if z in dec:                                            # case 1: declared value
                    return dec[z]
                if marg.get(z, F(0)) > 0 and (j, z) not in force_free:  # case 2: supported input
                    return {key[-1]: w / marg[z] for key, w in joint.items() if key[:-1] == z}
                return lam[(j, z)]                                      # case 4: null input
        for vals, p in states.items():
            env = dict(zip(names, vals))
            for value, w in factor(env).items():
                if w:
                    new[vals + (value,)] += p * w
        names.append(j)
        states = dict(new)
    den = marginal(list(record), rlaw, list(S_r)).get(r, F(0))
    assert den > 0
    out = defaultdict(F)
    for vals, p in states.items():
        out[P["loss"](dict(zip(names, vals)))] += p / den
    return clean(out)


def vertex_families(P, blocks):
    """Every point-mass family lambda_{j,z} = delta_a with a in the allowed set E_{j,z}."""
    options = [[point(a) for a in allowed(P, j, z)] for (j, z) in blocks]
    for choice in product(*options):
        yield dict(zip(blocks, choice))


def random_family(P, blocks, rng, denom=7):
    lam = {}
    for (j, z) in blocks:
        vals = list(allowed(P, j, z))
        w = [rng.randint(0, denom) for _ in vals]
        if sum(w) == 0:
            w[0] = 1
        lam[(j, z)] = clean({a: F(x, sum(w)) for a, x in zip(vals, w)})
    return lam


def true_lam(P, tables, blocks):
    """lambda_M: the mechanism's own kernel values at the null blocks."""
    return {(j, z): clean(tables[j][z]) for (j, z) in blocks}


def replacement_law(P, base):
    kappa = base.replace(P["rep"])
    return vs.conditional_loss_law(kappa, P["loss"], given_r(P))


# --------------------------------------------------------------------------
# Brute-force identified set (grid class, prefix pruning on the record law)
# --------------------------------------------------------------------------


def brute_force(P, grids, target, ties=None):
    """All grid kernel tables, with the declared values fixed, whose baseline record law equals target.

    A table that puts mass outside an allowed set E_{j,z} is dropped (the class
    constraint).  ties maps a coordinate to an earlier coordinate whose table it
    copies (one named kernel hosted by two slots).  The pruning compares the
    marginal of the recorded coordinates among the first i coordinates, a
    necessary condition for equal record laws; the final list is filtered by the
    full record law.  No identification result is used.
    """
    names, inputs, ranges, record = P["names"], P["inputs"], P["ranges"], P["record"]
    declared = P.get("declared", {})
    allow = P.get("allowed", {})
    ties = ties or {}
    partial = [({}, {(): F(1)})]
    for idx, name in enumerate(names):
        ins = inputs[name]
        pos = [names.index(i) for i in ins]
        zs = list(product(*[ranges[i] for i in ins]))
        dec = declared.get(name, {})
        free = [z for z in zs if z not in dec]
        rec_prefix = [n for n in names[:idx + 1] if n in record]
        want = marginal(list(record), target, rec_prefix)
        out = []
        for tables, plaw in partial:
            if name in ties:
                choices = [tuple(tables[ties[name]][z] for z in free)]
            else:
                choices = product(grids[name], repeat=len(free))
            for choice in choices:
                if any((name, z) in allow and not set(clean(c)) <= set(allow[(name, z)])
                       for z, c in zip(free, choice)):
                    continue
                table = dict(dec)
                table.update(zip(free, choice))
                nl = defaultdict(F)
                for path, p in plaw.items():
                    for val, w in table[tuple(path[k] for k in pos)].items():
                        if w:
                            nl[path + (val,)] += p * w
                nl = dict(nl)
                if marginal(names[:idx + 1], nl, rec_prefix) == want:
                    out.append(({**tables, name: table}, nl))
        partial = out
    survivors = []
    for tables, _ in partial:
        m = mech_from_tables(names, inputs, tables)
        if record_law(m, record) == target:
            survivors.append(tables)
    return survivors


# --------------------------------------------------------------------------
# Theorem thm:replay(iii) quantities in the coordinate model (one replaced decision)
# --------------------------------------------------------------------------


def theorem41_outer(P, phi=lambda l: l):
    """nu, w_h, X_h, and the vertex set of S_phi(P0) from the full baseline law and the two maps.

    The replaced coordinate d is deterministic: baseline f0, replacement fk, both
    reading S_d.  Agreement paths contribute their baseline loss; every deviation
    is continuing (no abort in these examples).  X_h collects phi(loss) over all
    replacement-consistent completions of h^+ in the product space.
    """
    names, ranges = P["names"], P["ranges"]
    (dname,) = P["D"]
    d = names.index(dname)
    f0, fk, Sd = P["f0"], P["fk"], P["inputs"][dname]
    law0 = P["base"].law()
    cond = given_r(P)
    den = sum(p for path, p in law0.items() if cond(dict(zip(names, path))))
    nu, w = defaultdict(F), defaultdict(F)
    for path, p in law0.items():
        env = dict(zip(names, path))
        if not cond(env):
            continue
        z = tuple(env[i] for i in Sd)
        if fk(*z) == f0(*z):
            nu[phi(P["loss"](env))] += p / den
        else:
            w[path[:d]] += p / den
    X = {}
    for h in w:
        env_h = dict(zip(names[:d], h))
        hplus = h + (fk(*[env_h[i] for i in Sd]),)
        vals = set()
        for rest in product(*[ranges[n] for n in names[d + 1:]]):
            env = dict(zip(names, hplus + rest))
            vals.add(phi(P["loss"](env)))
        X[h] = vals
    hs = sorted(w, key=repr)
    vertices = set()
    for xs in product(*[sorted(X[h], key=repr) for h in hs]):
        vertices.add(frozen(law_add(dict(nu), *[{x: w[h]} for h, x in zip(hs, xs)])))
    return clean(nu), dict(w), X, vertices


def in_S(mu, nu, w, X):
    """Is mu in S_phi = {nu + sum_h w_h Q_h : Q_h in Delta(X_h)}?  Exact transport test (Hall's condition)."""
    d = {x: mu.get(x, F(0)) - nu.get(x, F(0)) for x in set(mu) | set(nu)}
    if any(v < 0 for v in d.values()) or sum(d.values()) != sum(w.values()):
        return False
    hs = list(w)
    for k in range(1, len(hs) + 1):
        for H in combinations(hs, k):
            reach = set().union(*[X[h] for h in H])
            if sum(w[h] for h in H) > sum(d.get(x, F(0)) for x in reach):
                return False
    return True


def pivot_check(P, phi=lambda l: l):
    """(P1)-(P3) of part (f) and the maps x_h, checked on the family M^lambda.

    (P1): the pivot is a null block read from the replayed prefix h^+.
    (P3): under every point-mass family lambda, phi(L) is a function of V_j on
    the paths of positive P^kappa_{M^lambda}-probability in [h]; x_h(a) is the
    value when lambda_{j,z} = delta_a.  (P2): distinct pivots.
    Returns (nu, w, X, pivots, maps, onto) or None, with a reason.
    """
    names, ranges, tables = P["names"], P["ranges"], P["tables"]
    rlaw = record_law(P["base"], P["record"])
    blocks = null_blocks(P, rlaw)
    (dname,) = P["D"]
    d = names.index(dname)
    fk, Sd = P["fk"], P["inputs"][dname]
    nu, w, X, _ = theorem41_outer(P, phi)
    fam_laws = []
    for lam in vertex_families(P, blocks):
        m = mech_from_tables(names, P["inputs"], modified_tables(tables, lam))
        assert record_law(m, P["record"]) == rlaw          # Lemma r4-zero: the record law is unchanged
        fam_laws.append((lam, m.replace(P["rep"]).law()))
    pivots, maps = {}, {}
    for h in w:
        env_h = dict(zip(names[:d], h))
        hplus = h + (fk(*[env_h[i] for i in Sd]),)
        env_hp = dict(zip(names[:d + 1], hplus))
        cands = [(j, z) for (j, z) in blocks
                 if set(P["inputs"][j]) <= set(names[:d + 1])
                 and tuple(env_hp[i] for i in P["inputs"][j]) == z]
        found = None
        for (j, z) in cands:
            jpos = names.index(j)
            xh, ok = {}, True
            for lam, lawk in fam_laws:
                (a,) = lam[(j, z)]
                hit = False
                for path, p in lawk.items():
                    if p and path[:d] == h:
                        assert path[jpos] == a and path[:d + 1] == hplus
                        hit = True
                        val = phi(P["loss"](dict(zip(names, path))))
                        if xh.setdefault(a, val) != val:
                            ok = False
                assert hit
            if ok:
                found = ((j, z), xh)
                break
        if found is None:
            return None, f"(P1) or (P3) fails at h={h}"
        pivots[h], maps[h] = found
        assert set(maps[h]) == set(allowed(P, *found[0]))
        assert set(maps[h].values()) <= X[h]                  # Step 4: x_h(E) is inside X_h
    if len(set(pivots.values())) != len(pivots):
        return None, "(P2) fails: two continuing histories share one null block"
    onto = all(set(maps[h].values()) == X[h] for h in w)
    return (nu, w, X, pivots, maps, onto), "ok"


def check_part_f(P, rng, phi=lambda l: l, n_random=10):
    """The conclusion of part (f) with or without the onto condition.

    Image vertex set (point-mass families) = {nu + sum_h w_h delta_{x_h(a_h)}: a_h in E_h};
    decomposition phi_# Q^lambda = nu + sum_h w_h (x_h)_# lambda_h for random lambda;
    realization: M^lambda has the observed record law and its replacement law is Q^lambda;
    with onto, the image vertex set equals that of S_phi(P0).
    """
    res, why = pivot_check(P, phi)
    rlaw = record_law(P["base"], P["record"])
    blocks = null_blocks(P, rlaw)
    _, _, _, S_vertices = theorem41_outer(P, phi)
    img = {frozen(push(g_lambda_formula(P, rlaw, lam), phi)) for lam in vertex_families(P, blocks)}
    if res is None:
        return None, why, img, S_vertices
    nu, w, X, pivots, maps, onto = res
    hs = sorted(w, key=repr)
    rhs = set()
    for xs in product(*[sorted(set(maps[h].values()), key=repr) for h in hs]):
        rhs.add(frozen(law_add(dict(nu), *[{x: w[h]} for h, x in zip(hs, xs)])))
    assert img == rhs
    for _ in range(n_random):
        lam = random_family(P, blocks, rng)
        parts = [law_scale(push(lam[pivots[h]], lambda a, h=h: maps[h][a]), w[h]) for h in w]
        Q = g_lambda_formula(P, rlaw, lam)
        assert push(Q, phi) == law_add(nu, *parts)
        m = mech_from_tables(P["names"], P["inputs"], modified_tables(P["tables"], lam))
        assert record_law(m, P["record"]) == rlaw and replacement_law(P, m) == Q
        assert in_S(push(Q, phi), nu, w, X)
    for L in img:
        assert in_S(dict(L), nu, w, X)
    if onto:
        assert img == S_vertices
    return res, why, img, S_vertices


# --------------------------------------------------------------------------
# (a), (b), (c), (d): random small instances against brute force
# --------------------------------------------------------------------------

GRID01 = [bern(0), bern(HALF), bern(1)]
GRID012 = [point(0), point(1), point(2), {0: HALF, 1: HALF}, {1: HALF, 2: HALF}, uniform_on((0, 1, 2))]


def random_instance(rng):
    n = rng.choice((4, 5))
    names = [f"V{i}" for i in range(n)]
    ranges = {v: (0, 1) for v in names}
    inputs = {}
    for i, v in enumerate(names):
        k = min(i, rng.choice((0, 1, 1, 2)))
        inputs[v] = tuple(sorted(rng.sample(names[:i], k), key=names.index))
    tables = {v: {z: rng.choice(GRID01) for z in product(*[ranges[i] for i in inputs[v]])} for v in names}
    d = rng.randrange(1, n - 1)
    dname = names[d]
    kins = tuple(sorted(rng.sample(names[:d], min(d, rng.choice((0, 1, 2)))), key=names.index))
    ktab = {z: rng.choice((point(0), point(1), point(0), point(1), bern(HALF)))
            for z in product(*[ranges[i] for i in kins])}
    rep = {dname: (kins, lambda *z, t=ktab: t[tuple(z)])}
    probe = mech_from_tables(names, inputs, tables)
    dsc = vs.descendants(probe, {dname})
    declared = {}
    cand = [v for v in names if v in dsc and v != dname]
    if cand and rng.random() < 0.5:
        j = rng.choice(cand)
        z = rng.choice(sorted(tables[j]))
        val = rng.choice((point(0), point(1)))
        tables[j][z] = val                      # the class fixes this value; the true mechanism has it
        declared[j] = {z: val}
    base = mech_from_tables(names, inputs, tables)
    kappa = base.replace(rep)
    S_r = tuple(sorted(rng.sample(names[:d], min(d, rng.choice((0, 0, 1)))), key=names.index))
    supp = sorted(marginal(base.names, base.law(), list(S_r)))
    r = rng.choice(supp)
    downstream = [v for v in names if v in dsc]
    first = rng.choice(downstream)
    others = [v for v in names if v != first]
    S_L = tuple(sorted({first} | set(rng.sample(others, rng.choice((0, 1)))), key=names.index))
    lmap = {vals: rng.choice((0, 1, 2)) for vals in product(*[ranges[v] for v in S_L])}
    loss = lambda env, S_L=S_L, lmap=lmap: lmap[tuple(env[v] for v in S_L)]
    P = dict(names=names, inputs=inputs, ranges=ranges, D=(dname,), rep=rep, base=base, kappa=kappa,
             S_L=S_L, loss=loss, S_r=S_r, r=r, declared=declared)
    J, _ = relevant(P)
    P["record"] = tuple(v for v in names if v in J or rng.random() < 0.5)
    P["tables"] = tables
    P["alphabet"] = sorted(set(lmap.values()))
    return P


def compare_with_brute_force(P, grids, rng, stats):
    """Parts (a), (b), (d) on one instance; returns whether some event has a nondegenerate interval."""
    base, record = P["base"], P["record"]
    rlaw = record_law(base, record)
    blocks = null_blocks(P, rlaw)
    alphabet = P["alphabet"]
    events = [frozenset(B) for k in range(1, len(alphabet)) for B in combinations(alphabet, k)]
    vertex_laws = {frozen(g_lambda_formula(P, rlaw, lam)) for lam in vertex_families(P, blocks)}
    survivors = brute_force(P, grids, rlaw)
    assert P["tables"] in survivors
    bf_laws = set()
    for tables in survivors:
        m = mech_from_tables(P["names"], P["inputs"], tables)
        Q_M = replacement_law(P, m)
        assert Q_M == g_lambda_formula(P, rlaw, true_lam(P, tables, blocks))      # (a)
        bf_laws.add(frozen(Q_M))
    assert vertex_laws <= bf_laws                                                 # (b): realized
    nondegenerate = False
    for B in events:
        img = [prob_of(dict(L), B) for L in vertex_laws]
        bf = [prob_of(dict(L), B) for L in bf_laws]
        assert (min(img), max(img)) == (min(bf), max(bf))                         # (b), (d)
        nondegenerate |= min(img) < max(img)
    for _ in range(3):
        if not blocks:
            break
        lam = random_family(P, blocks, rng)
        b = rng.choice(blocks)
        a1, a2 = random_family(P, [b], rng)[b], random_family(P, [b], rng)[b]
        t = F(rng.randint(1, 6), 7)
        mix = law_add(law_scale(a1, t), law_scale(a2, 1 - t))
        lhs = g_lambda_formula(P, rlaw, {**lam, b: mix})
        rhs = law_add(law_scale(g_lambda_formula(P, rlaw, {**lam, b: a1}), t),
                      law_scale(g_lambda_formula(P, rlaw, {**lam, b: a2}), 1 - t))
        assert lhs == rhs                                                         # (d): affine in a block
        Ql = g_lambda_formula(P, rlaw, lam)
        for B in events:
            img = [prob_of(dict(L), B) for L in vertex_laws]
            assert min(img) <= prob_of(Ql, B) <= max(img)
        stats["affine"] += 1
    stats["survivors"] += len(survivors)
    return blocks, vertex_laws, nondegenerate


def check_random_instances(n_instances=500, seed=20261003):
    rng = random.Random(seed)
    stats = defaultdict(int)
    for _ in range(n_instances):
        P = random_instance(rng)
        base, record = P["base"], P["record"]
        rlaw = record_law(base, record)
        blocks, vertex_laws, nondegenerate = compare_with_brute_force(P, {v: GRID01 for v in P["names"]}, rng,
                                                                      stats)
        cond = vs.conditions(base, P["kappa"], set(P["D"]), P["S_L"], record, P["S_r"], P["r"],
                             known_from_declared(P["declared"]))
        assert cond["G2"] and cond["G3"] and cond["G4"]
        stats["instances"] += 1
        stats["blocks"] += bool(blocks)
        if cond["G1"]:
            # (c): under (G1) the image is the singleton of Proposition prop:supported(i).
            g = vs.g_formula(record, rlaw, base, P["kappa"], set(P["D"]), P["S_L"], P["loss"], P["S_r"], P["r"],
                             known_from_declared(P["declared"]))
            assert vertex_laws == {frozen(g)}
            for _ in range(2):
                if blocks:
                    assert g_lambda_formula(P, rlaw, random_family(P, blocks, rng)) == g
            stats["G1"] += 1
            stats["G1_with_blocks"] += bool(blocks)
        else:
            stats["notG1"] += 1
            stats["notG1_interval"] += nondegenerate
            stats["notG1_singleton"] += len(vertex_laws) == 1
    return stats


def check_sharpness_random():
    s = check_random_instances()
    assert s["notG1_interval"] > 0 and s["G1_with_blocks"] > 0 and s["notG1_singleton"] > 0
    print(f"PASS (a),(b),(d) random instances: {s['instances']} instances with 4-5 binary coordinates, random "
          f"declared inputs, declared values, records, and pre-intervention records; brute force over grid "
          f"mechanisms with the observed record law ({s['survivors']} compatible mechanisms in all) gives, for "
          f"every event, the same min and max as the point-mass families of lambda; every compatible mechanism's "
          f"replacement law equals G^(lambda_M) exactly; every point-mass family is realized; {s['affine']} random "
          f"affine-in-a-block identities hold; (G1) fails in {s['notG1']} instances, {s['notG1_interval']} with a "
          f"nondegenerate interval and {s['notG1_singleton']} with a singleton image.")
    print(f"PASS (c) supported extreme: in all {s['G1']} random instances where (G1) holds, every family lambda "
          f"gives the law returned by verify_supported.g_formula (Proposition prop:supported(i)); {s['G1_with_blocks']} of "
          f"them have null blocks that the replacement never reaches.")


# --------------------------------------------------------------------------
# (a), (b), (d) with restricted allowed sets and ternary coordinates
# --------------------------------------------------------------------------


def random_instance_allowed(rng):
    """Random instance with binary and ternary coordinates and random allowed sets E_{j,z} at the null blocks.

    Input sets are kept small (at most four input values for a binary
    coordinate, at most three for a ternary one) so the grid brute force stays
    exact and fast.  Each allowed set contains the support of the true kernel.
    """
    while True:
        n = 4
        names = [f"V{i}" for i in range(n)]
        ranges = {v: rng.choice(((0, 1), (0, 1, 2))) for v in names}
        inputs = {}
        for i, v in enumerate(names):
            cands = names[:i]
            k = min(i, rng.choice((0, 1, 1, 2)))
            ins = tuple(sorted(rng.sample(cands, k), key=names.index))
            size = 1
            for x in ins:
                size *= len(ranges[x])
            limit = 3 if len(ranges[v]) == 3 else 4
            while size > limit:
                ins = ins[1:]
                size = 1
                for x in ins:
                    size *= len(ranges[x])
            inputs[v] = ins
        grid = {v: (GRID012 if len(ranges[v]) == 3 else GRID01) for v in names}
        tables = {v: {z: rng.choice(grid[v]) for z in product(*[ranges[i] for i in inputs[v]])} for v in names}
        d = rng.randrange(1, n - 1)
        dname = names[d]
        kins = tuple(sorted(rng.sample(names[:d], min(d, rng.choice((0, 1)))), key=names.index))
        ktab = {z: point(rng.choice(ranges[dname])) for z in product(*[ranges[i] for i in kins])}
        rep = {dname: (kins, lambda *z, t=ktab: t[tuple(z)])}
        base = mech_from_tables(names, inputs, tables)
        kappa = base.replace(rep)
        dsc = vs.descendants(base, {dname})
        downstream = [v for v in names if v in dsc and v != dname]
        if not downstream:
            continue
        S_r = ()
        r = ()
        first = rng.choice(downstream)
        others = [v for v in names if v != first]
        S_L = tuple(sorted({first} | set(rng.sample(others, rng.choice((0, 1)))), key=names.index))
        lmap = {vals: rng.choice((0, 1, 2)) for vals in product(*[ranges[v] for v in S_L])}
        loss = lambda env, S_L=S_L, lmap=lmap: lmap[tuple(env[v] for v in S_L)]
        P = dict(names=names, inputs=inputs, ranges=ranges, D=(dname,), rep=rep, base=base, kappa=kappa,
                 S_L=S_L, loss=loss, S_r=S_r, r=r, declared={})
        J, _ = relevant(P)
        P["record"] = tuple(v for v in names if v in J or rng.random() < 0.5)
        P["tables"] = tables
        P["alphabet"] = sorted(set(lmap.values()))
        rlaw = record_law(base, P["record"])
        blocks = null_blocks(P, rlaw)
        if not blocks:
            continue
        allow = {}
        for (j, z) in blocks:
            supp = set(clean(tables[j][z]))
            extra = [a for a in ranges[j] if a not in supp and rng.random() < 0.4]
            allow[(j, z)] = tuple(a for a in ranges[j] if a in supp or a in extra)
        P["allowed"] = allow
        P["grids"] = grid
        return P


def check_restricted_allowed_sets(n_instances=300, seed=31):
    rng = random.Random(seed)
    stats = defaultdict(int)
    for _ in range(n_instances):
        P = random_instance_allowed(rng)
        rlaw = record_law(P["base"], P["record"])
        blocks, vertex_laws, nondeg = compare_with_brute_force(P, P["grids"], rng, stats)
        strict = [b for b in blocks if len(allowed(P, *b)) < len(P["ranges"][b[0]])]
        stats["instances"] += 1
        stats["strict_blocks"] += len(strict)
        stats["with_strict"] += bool(strict)
        stats["ternary"] += any(len(P["ranges"][v]) == 3 for v in P["names"])
        stats["nondeg"] += nondeg
        # The allowed sets matter when the image with full allowed sets is different (larger).
        full = dict(P)
        full.pop("allowed")
        img_full = {frozen(g_lambda_formula(full, rlaw, lam)) for lam in vertex_families(full, blocks)}
        assert vertex_laws <= img_full
        stats["matters"] += vertex_laws != img_full
    assert stats["with_strict"] > 0 and stats["ternary"] > 0 and stats["matters"] > 0
    globals()["RA_STATS"] = dict(stats)
    print(f"PASS (a),(b),(d) restricted allowed sets: {stats['instances']} random instances with binary and "
          f"ternary coordinates ({stats['ternary']} with a ternary coordinate) and random allowed sets at the null "
          f"blocks ({stats['strict_blocks']} strict subsets of the range, in {stats['with_strict']} instances, "
          f"{stats['matters']} of which have a smaller point-mass image than with full allowed sets); "
          f"brute force over grid mechanisms inside the allowed sets ({stats['survivors']} compatible mechanisms) "
          f"gives the point-mass extremes for every event, every compatible mechanism's law is G^(lambda_M), "
          f"every point-mass family is realized, {stats['affine']} affine identities hold, and "
          f"{stats['nondeg']} instances have a nondegenerate interval.")


# --------------------------------------------------------------------------
# (d): the image need not be convex
# --------------------------------------------------------------------------


def check_nonconvex_image():
    """Two null blocks on one path, N2 not reading N1: product laws, a non-convex set."""
    names = ["A", "N1", "N2"]
    ranges = {"A": ("go", "retry"), "N1": (0, 1), "N2": (0, 1)}
    inputs = {"A": (), "N1": ("A",), "N2": ("A",)}
    tables = {"A": {(): point("go")},
              "N1": {("go",): point(0), ("retry",): bern(HALF)},
              "N2": {("go",): point(0), ("retry",): bern(HALF)}}
    base = mech_from_tables(names, inputs, tables)
    rep = {"A": ((), lambda: point("retry"))}
    loss = lambda env: env["N1"] + 2 * env["N2"]
    P = dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, kappa=base.replace(rep),
             S_L=("N1", "N2"), loss=loss, S_r=(), r=(), record=tuple(names),
             f0=lambda: "go", fk=lambda: "retry")
    rlaw = record_law(base, P["record"])
    blocks = null_blocks(P, rlaw)
    assert blocks == [("N1", ("retry",)), ("N2", ("retry",))]
    rng = random.Random(7)
    for _ in range(20):
        Q = g_lambda_formula(P, rlaw, random_family(P, blocks, rng))
        g = lambda x: Q.get(x, F(0))
        assert g(0) * g(3) == g(1) * g(2)                  # every image law is a product law
    lo = g_lambda_formula(P, rlaw, {blocks[0]: point(0), blocks[1]: point(0)})
    hi = g_lambda_formula(P, rlaw, {blocks[0]: point(1), blocks[1]: point(1)})
    mid = law_add(law_scale(lo, HALF), law_scale(hi, HALF))
    assert lo == {0: 1} and hi == {3: 1} and mid == {0: HALF, 3: HALF}
    assert mid.get(0, 0) * mid.get(3, 0) != mid.get(1, 0) * mid.get(2, 0)   # the midpoint is not in the image
    means = [mean_of(g_lambda_formula(P, rlaw, lam)) for lam in vertex_families(P, blocks)]
    assert (min(means), max(means)) == (0, 3)
    _, _, _, S_vertices = theorem41_outer(P)
    assert frozen(lo) in S_vertices and frozen(hi) in S_vertices   # S_phi is convex and contains mid
    print("PASS (d) non-convex image: after a retry two unknown kernels read only the retry, with X = N1 + 2 N2; "
          "every image law satisfies P(0)P(3) = P(1)P(2), so the midpoint 1/2 delta_0 + 1/2 delta_3 of two image "
          "points is not in the image, while the mean range [0,3] is an interval attained at point masses; "
          "Theorem thm:replay(iii) gives the full simplex.")


# --------------------------------------------------------------------------
# (e): shared null inputs, check T1-14 through the formula
# --------------------------------------------------------------------------


def t114_problem(distinct, flip, x=HALF, xc=HALF, xp=HALF):
    def nker(*i):
        if i[-1] == "go":
            return point(0)
        if distinct:
            return bern(xc if i[0] == "c" else xp)
        return bern(x)

    names = ["W", "A", "N"]
    ranges = {"W": ("c", "p"), "A": ("go", "retry"), "N": (0, 1)}
    inputs = {"W": (), "A": ("W",), "N": ("W", "A") if distinct else ("A",)}
    tables = {"W": {(): {"c": HALF, "p": HALF}}, "A": {(w,): point("go") for w in ranges["W"]},
              "N": {z: clean(nker(*z)) for z in product(*[ranges[i] for i in inputs["N"]])}}
    base = mech_from_tables(names, inputs, tables)
    rep = {"A": (("W",), lambda w: point("retry"))}
    if flip:
        loss = lambda env: env["N"] if env["W"] == "c" else 1 - env["N"]
    else:
        loss = lambda env: env["N"] + (1 if env["W"] == "p" else 0)
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("W", "N"), loss=loss, S_r=(), r=(), record=tuple(names),
                f0=lambda w: "go", fk=lambda w: "retry")


T114_GRIDS = {"W": [{"c": HALF, "p": HALF}, point("c"), point("p")],
              "A": [point("go"), point("retry"), {"go": HALF, "retry": HALF}],
              "N": [bern(F(k, 10)) for k in range(11)]}


def check_shared_input_t114():
    # Shared input, L = N for c and 1 - N for p.
    P = t114_problem(distinct=False, flip=True)
    rlaw = record_law(P["base"], P["record"])
    blocks = null_blocks(P, rlaw)
    assert blocks == [("N", ("retry",))]
    cond = vs.conditions(P["base"], P["kappa"], {"A"}, P["S_L"], P["record"])
    assert cond == {"G1": False, "G2": True, "G3": True, "G4": True}
    try:
        vs.g_formula(P["record"], rlaw, P["base"], P["kappa"], {"A"}, P["S_L"], P["loss"])
        raise AssertionError("support failure not detected")
    except vs.PositivityError:
        pass
    ends = [g_lambda_formula(P, rlaw, lam) for lam in vertex_families(P, blocks)]
    # One block, so Q^lambda is affine in lambda on a segment; equal values at both ends make it constant.
    assert ends == [{0: HALF, 1: HALF}, {0: HALF, 1: HALF}]
    survivors = brute_force(P, T114_GRIDS, rlaw)
    bf = {frozen(replacement_law(P, mech_from_tables(P["names"], P["inputs"], t))) for t in survivors}
    assert len(survivors) == 11 and bf == {frozen({0: HALF, 1: HALF})}
    nu, w, X, S_vertices = theorem41_outer(P)
    assert nu == {} and set(w.values()) == {HALF} and all(v == {0, 1} for v in X.values())
    assert S_vertices == {frozen({0: 1}), frozen({0: HALF, 1: HALF}), frozen({1: 1})}
    # Shared input, L = N + 1{p}: a segment of laws; the product element 1/2 delta_0 + 1/2 delta_2 is outside.
    P2 = t114_problem(distinct=False, flip=False)
    rlaw2 = record_law(P2["base"], P2["record"])
    b2 = null_blocks(P2, rlaw2)
    lo, hi = [g_lambda_formula(P2, rlaw2, lam) for lam in vertex_families(P2, b2)]
    assert lo == {0: HALF, 1: HALF} and hi == {1: HALF, 2: HALF}
    assert (mean_of(lo), mean_of(hi)) == (HALF, F(3, 2))
    assert lo.get(1) == hi.get(1) == HALF          # affine in lambda: every image law puts 1/2 on the value 1
    _, _, _, S2 = theorem41_outer(P2)
    assert frozen({0: HALF, 2: HALF}) in S2
    print("PASS (e) shared null input (T1-14): types c and p both reach the one null block (N, retry); (G1) fails "
          "and Proposition prop:supported's g-formula raises a support error; the generalized formula gives Bern(1/2) at "
          "both point masses of the single free argument, hence for every lambda, and 11 grid mechanisms with the "
          "baseline law all give Bern(1/2), while Theorem thm:replay(iii) gives the whole simplex on {0,1}; with "
          "L = N + 1{p} the image is a segment with mean range [1/2, 3/2], and 1/2 delta_0 + 1/2 delta_2 lies "
          "in the Theorem thm:replay(iii) set but not in the image.")


# --------------------------------------------------------------------------
# (f): the saturated extreme (pivot condition with onto) and its failures
# --------------------------------------------------------------------------


def feasibility_problem(three_types):
    names = ["W", "C0", "A", "C1"]
    if three_types:
        wlaw = {"cheap": F(1, 4), "paid": F(1, 4), "done": HALF}
        retried = ("cheap", "paid")
        c1_inputs = ("W", "A")                    # distinct null inputs (cheap, retry), (paid, retry)
    else:
        wlaw = {"cheap": HALF, "paid": HALF}
        retried = ("paid",)
        c1_inputs = ("A",)                        # one null input (retry), reached by paid tasks only
    ranges = {"W": tuple(wlaw), "C0": (0, 1), "A": ("finish", "retry"), "C1": (0, 1)}
    inputs = {"W": (), "C0": ("W",), "A": ("W", "C0"), "C1": c1_inputs}
    tables = {"W": {(): dict(wlaw)},
              "C0": {(w,): point(1 if w == "paid" else 0) for w in wlaw},
              "A": {(w, c): point("finish") for w in wlaw for c in (0, 1)},
              "C1": {z: (point(0) if z[-1] == "finish" else bern(THIRD))
                     for z in product(*[ranges[i] for i in c1_inputs])}}
    base = mech_from_tables(names, inputs, tables)
    fk = lambda w, c: "retry" if w in retried else "finish"
    rep = {"A": (("W", "C0"), lambda w, c: point(fk(w, c)))}
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("C0", "C1"), loss=lambda env: env["C0"] + env["C1"],
                S_r=(), r=(), record=tuple(names), f0=lambda w, c: "finish", fk=fk)


def followed_problem():
    """Retry reaches a null input of N, but a supported random kernel Y follows: (P3) fails."""
    names = ["U", "A", "N", "Y"]
    ranges = {"U": (0, 1), "A": ("go", "retry"), "N": (0, 1), "Y": (0, 1)}
    inputs = {"U": (), "A": ("U",), "N": ("A",), "Y": ("N",)}
    tables = {"U": {(): bern(HALF)}, "A": {(u,): point("go") for u in (0, 1)},
              "N": {(a,): bern(HALF) for a in ("go", "retry")},
              "Y": {(n,): bern(F(1, 4) + F(n, 2)) for n in (0, 1)}}
    base = mech_from_tables(names, inputs, tables)
    fk = lambda u: "retry" if u == 0 else "go"
    rep = {"A": (("U",), lambda u: point(fk(u)))}
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("Y",), loss=lambda env: env["Y"], S_r=(), r=(),
                record=tuple(names), f0=lambda u: "go", fk=fk)


def check_saturated_extreme():
    rng = random.Random(11)
    # (f1) T1-14 with distinct inputs: two continuing histories, two distinct pivots.
    out = {}
    for flip in (True, False):
        res, why, img, S = check_part_f(t114_problem(distinct=True, flip=flip), rng)
        assert res is not None and res[-1], why
        out[flip] = len(S)
    # (f2) T1-8: one retried history with accrued cost 1, X_h = {1, 2}; unconditional and given the paid record.
    P = feasibility_problem(three_types=False)
    res, why, img, S = check_part_f(P, rng)
    assert res is not None and res[-1], why
    means = sorted(mean_of(dict(L)) for L in img)
    assert (means[0], means[-1]) == (HALF, F(1))
    Pp = dict(P, S_r=("W",), r=("paid",))
    res, why, img_p, _ = check_part_f(Pp, rng)
    assert res is not None and res[-1], why
    means_p = sorted(mean_of(dict(L)) for L in img_p)
    assert (means_p[0], means_p[-1]) == (F(1), F(2))
    # (f3) Two retried types with prefix-dependent feasible sets X_h = {0,1} and {1,2} at distinct pivots.
    P3 = feasibility_problem(three_types=True)
    res, why, img3, S3 = check_part_f(P3, rng)
    assert res is not None and res[-1], why
    assert sorted(sorted(x) for x in res[2].values()) == [[0, 1], [1, 2]] and len(S3) == 4
    # Contrasts: the conditions of part (f) fail and the image is strictly inside S_phi.
    res, why, img_s, S_s = check_part_f(t114_problem(distinct=False, flip=True), rng)
    assert res is None and "(P2)" in why and img_s < S_s
    Pf = followed_problem()
    res, why, img_f, S_f = check_part_f(Pf, rng)
    assert res is None and "(P3)" in why
    p1 = sorted(prob_of(dict(L), {1}) for L in img_f)
    s1 = sorted(prob_of(dict(L), {1}) for L in S_f)
    assert (p1[0], p1[-1]) == (F(3, 8), F(5, 8)) and (s1[0], s1[-1]) == (F(1, 4), F(3, 4))
    rlaw_f = record_law(Pf["base"], Pf["record"])
    survivors = brute_force(Pf, {"U": [bern(HALF), bern(0), bern(1)], "A": [point("go"), point("retry")],
                                 "N": GRID01, "Y": [bern(F(1, 4)), bern(F(3, 4)), bern(0), bern(1)]}, rlaw_f)
    bf = sorted(replacement_law(Pf, mech_from_tables(Pf["names"], Pf["inputs"], t)).get(1, F(0)) for t in survivors)
    assert (bf[0], bf[-1]) == (F(3, 8), F(5, 8))
    print(f"PASS (f) saturated extreme: (P1)-(P3) (P3 on the paths of every point-mass family) and the onto "
          f"condition x_h(E) = X_h hold, and the image equals the Theorem thm:replay(iii) set S_phi(P0) (equal vertex "
          f"sets, Q^lambda = nu + sum_h w_h (x_h)_# lambda_h for random lambda, and M^lambda realizes it with the "
          f"observed record law), in T1-14 with distinct inputs ({out[True]} and {out[False]} vertices), in T1-8 "
          f"(mean range [1/2,1], and [1,2] given the paid record), and with prefix-dependent feasible sets "
          f"X_h = {{0,1}} and {{1,2}} at two distinct pivots; with a shared pivot (P2) fails and the image is a "
          f"strict subset; with a supported random kernel after the pivot (P3) fails and P(L=1) ranges over "
          f"[3/8,5/8] (brute force agrees) inside [1/4,3/4].")


# --------------------------------------------------------------------------
# (f) without the onto condition: image nu + sum_h w_h Delta(x_h(E))
# --------------------------------------------------------------------------


def copied_bit_problem():
    """A: go -> retry.  N reads A (Bern(1/2) at go, null at retry); a logged Y copies N; L = N + 2Y."""
    names = ["A", "N", "Y"]
    ranges = {"A": ("go", "retry"), "N": (0, 1), "Y": (0, 1)}
    inputs = {"A": (), "N": ("A",), "Y": ("N",)}
    tables = {"A": {(): point("go")}, "N": {("go",): bern(HALF), ("retry",): bern(THIRD)},
              "Y": {(0,): point(0), (1,): point(1)}}
    base = mech_from_tables(names, inputs, tables)
    rep = {"A": ((), lambda: point("retry"))}
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("N", "Y"), loss=lambda env: env["N"] + 2 * env["Y"], S_r=(), r=(),
                record=tuple(names), f0=lambda: "go", fk=lambda: "retry")


def restricted_pivots_problem():
    """T1-14 with distinct ternary pivots and allowed sets {0,1} at (c, retry) and {2} at (p, retry); L = N."""
    names = ["W", "A", "N"]
    ranges = {"W": ("c", "p"), "A": ("go", "retry"), "N": (0, 1, 2)}
    inputs = {"W": (), "A": ("W",), "N": ("W", "A")}
    tables = {"W": {(): {"c": HALF, "p": HALF}}, "A": {(w,): point("go") for w in ("c", "p")},
              "N": {("c", "go"): point(0), ("p", "go"): point(1), ("c", "retry"): {0: HALF, 1: HALF},
                    ("p", "retry"): point(2)}}
    base = mech_from_tables(names, inputs, tables)
    rep = {"A": (("W",), lambda w: point("retry"))}
    allow = {("N", ("c", "retry")): (0, 1), ("N", ("p", "retry")): (2,)}
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("N",), loss=lambda env: env["N"], S_r=(), r=(), record=tuple(names),
                f0=lambda w: "go", fk=lambda w: "retry", allowed=allow)


def random_pivot_instance(rng):
    """Random instance of the pivot structure: type W, accrued cost C0, gate A (go -> retry for some types),
    ternary pivot N reading (W, A) or only A (shared), a logged deterministic follower Y = g(N), L = l(C0, N, Y)."""
    k = rng.choice((2, 3))
    types = ("t0", "t1", "t2")[:k]
    wts = [rng.randint(1, 3) for _ in types]
    wlaw = {t: F(x, sum(wts)) for t, x in zip(types, wts)}
    c0 = {t: rng.choice((0, 1)) for t in types}
    retried = set(rng.sample(types, rng.randint(1, k)))
    shared = rng.random() < 0.3
    n_inputs = ("A",) if shared else ("W", "A")
    yrange = rng.choice(((0, 1), (0, 1, 2)))
    g = {a: rng.choice(yrange) for a in (0, 1, 2)}
    lmap = {(c, a, y): rng.choice((0, 1, 2, 3)) for c in (0, 1) for a in (0, 1, 2) for y in yrange}
    names = ["W", "C0", "A", "N", "Y"]
    ranges = {"W": types, "C0": (0, 1), "A": ("go", "retry"), "N": (0, 1, 2), "Y": yrange}
    inputs = {"W": (), "C0": ("W",), "A": ("W", "C0"), "N": n_inputs, "Y": ("N",)}
    tables = {"W": {(): dict(wlaw)}, "C0": {(t,): point(c0[t]) for t in types},
              "A": {(t, c): point("go") for t in types for c in (0, 1)},
              "N": {z: (uniform_on((0, 1, 2)) if z[-1] == "go" else rng.choice(GRID012))
                    for z in product(*[ranges[i] for i in n_inputs])},
              "Y": {(a,): point(g[a]) for a in (0, 1, 2)}}
    base = mech_from_tables(names, inputs, tables)
    fk = lambda w, c, R=frozenset(retried): "retry" if w in R else "go"
    rep = {"A": (("W", "C0"), lambda w, c, fk=fk: point(fk(w, c)))}
    if rng.random() < 0.3:
        S_r, r = ("W",), (rng.choice(types),)
    else:
        S_r, r = (), ()
    P = dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
             kappa=base.replace(rep), S_L=("C0", "N", "Y"),
             loss=lambda env, lmap=lmap: lmap[(env["C0"], env["N"], env["Y"])],
             S_r=S_r, r=r, record=tuple(names), f0=lambda w, c: "go", fk=fk, shared=shared)
    rlaw = record_law(base, P["record"])
    allow = {}
    for (j, z) in null_blocks(P, rlaw):
        supp = set(clean(tables[j][z]))
        allow[(j, z)] = tuple(a for a in ranges[j] if a in supp or rng.random() < 0.5)
    P["allowed"] = allow
    return P


PIVOT_GRIDS = {"C0": [point(0), point(1)], "A": [point("go"), point("retry")], "N": GRID012,
               "Y": [point(0), point(1), point(2)]}


def brute_force_part_f(P, res, phi=lambda l: l):
    """Brute-force identified set of a pivot instance: every compatible grid mechanism's law decomposes
    through its own pivot laws, and the per-event extremes equal the image's point-mass extremes."""
    rlaw = record_law(P["base"], P["record"])
    blocks = null_blocks(P, rlaw)
    grids = dict(PIVOT_GRIDS)
    grids["W"] = [P["tables"]["W"][()]] + [point(t) for t in P["ranges"]["W"]]
    grids["Y"] = [point(a) for a in P["ranges"]["Y"]]
    survivors = brute_force(P, grids, rlaw)
    assert P["tables"] in survivors
    nu, w, X, pivots, maps, onto = res
    img = [push(g_lambda_formula(P, rlaw, lam), phi) for lam in vertex_families(P, blocks)]
    bf = []
    for t in survivors:
        Q = push(replacement_law(P, mech_from_tables(P["names"], P["inputs"], t)), phi)
        parts = [law_scale(push(clean(t[pivots[h][0]][pivots[h][1]]), lambda a, h=h: maps[h][a]), w[h])
                 for h in w]
        assert Q == law_add(nu, *parts)                       # Steps 1-2 of the proof of (f), with P3
        bf.append(Q)
    values = sorted({x for L in img + bf for x in L})
    for k in range(1, len(values)):
        for B in combinations(values, k):
            assert (min(prob_of(L, B) for L in img), max(prob_of(L, B) for L in img)) == \
                   (min(prob_of(L, B) for L in bf), max(prob_of(L, B) for L in bf))
    return len(survivors)


def check_pivot_without_onto(n_instances=200, seed=5):
    rng = random.Random(seed)
    # Copied bit: x_h takes only the values 0 and 3, X_h = {0,1,2,3}.
    P = copied_bit_problem()
    res, why, img, S = check_part_f(P, rng)
    assert res is not None and not res[-1], why
    (h,) = res[1]
    assert res[2][h] == {0, 1, 2, 3} and set(res[4][h].values()) == {0, 3}
    assert img == {frozen({0: 1}), frozen({3: 1})} and img < S and frozen({1: 1}) in S
    rlaw = record_law(P["base"], P["record"])
    survivors = brute_force(P, {"A": [point("go"), point("retry"), {"go": HALF, "retry": HALF}],
                                "N": [bern(0), bern(THIRD), bern(HALF), bern(1)], "Y": GRID01}, rlaw)
    bf = [replacement_law(P, mech_from_tables(P["names"], P["inputs"], t)) for t in survivors]
    assert len(survivors) == 4 and all(set(Q) <= {0, 3} for Q in bf)
    # Restricted allowed sets {0,1} and {2} at two distinct pivots.
    P2 = restricted_pivots_problem()
    res2, why2, img2, S2 = check_part_f(P2, rng)
    assert res2 is not None and not res2[-1], why2
    assert img2 == {frozen({0: HALF, 2: HALF}), frozen({1: HALF, 2: HALF})} and img2 < S2
    rlaw2 = record_law(P2["base"], P2["record"])
    surv2 = brute_force(P2, {"W": [{"c": HALF, "p": HALF}, point("c")], "A": [point("go"), point("retry")],
                             "N": GRID012}, rlaw2)
    bf2 = {frozen(replacement_law(P2, mech_from_tables(P2["names"], P2["inputs"], t))) for t in surv2}
    assert all(dict(L).get(2) == HALF and set(dict(L)) <= {0, 1, 2} for L in bf2) and img2 <= bf2
    # Random pivot instances.
    stats = defaultdict(int)
    for i in range(n_instances):
        Pr = random_pivot_instance(rng)
        res_r, why_r, img_r, S_r = check_part_f(Pr, rng, n_random=4)
        stats["instances"] += 1
        if res_r is None:
            assert "(P2)" in why_r and Pr["shared"]
            # Shared pivot: the inclusion of Remark r4-extremes still holds.
            nu, w, X, _ = theorem41_outer(Pr)
            rl = record_law(Pr["base"], Pr["record"])
            bl = null_blocks(Pr, rl)
            for L in img_r:
                assert in_S(dict(L), nu, w, X)
            for _ in range(4):
                assert in_S(g_lambda_formula(Pr, rl, random_family(Pr, bl, rng)), nu, w, X)
            stats["shared_fail"] += 1
            continue
        stats["pivot"] += 1
        stats["histories"] += len(res_r[1])
        stats["no_history"] += not res_r[1]
        if res_r[-1]:
            stats["onto"] += 1
        else:
            stats["nononto"] += 1
            stats["strict"] += img_r < S_r
        if i % 4 == 0:
            stats["bf_instances"] += 1
            stats["bf_survivors"] += brute_force_part_f(Pr, res_r)
    assert stats["nononto"] > 0 and stats["onto"] > 0 and stats["shared_fail"] > 0
    globals()["PF_STATS"] = dict(stats)
    print(f"PASS (f) without the onto condition: in the copied-bit example x_h takes only 0 and 3, the image is "
          f"Delta({{0,3}}) inside Delta({{0,1,2,3}}) = S_phi(P0), and all {len(survivors)} compatible grid "
          f"mechanisms give laws on {{0,3}}; with allowed sets {{0,1}} and {{2}} at two distinct pivots the image "
          f"is 1/2 Delta({{0,1}}) + 1/2 delta_2, strictly inside S_phi, and brute force agrees; in "
          f"{stats['instances']} random pivot instances ({stats['pivot']} satisfying (P1)-(P3), "
          f"{stats['nononto']} of them without onto and {stats['strict']} with an image strictly inside S_phi, "
          f"{stats['onto']} with onto, {stats['no_history']} of these with no continuing history), the image equals nu + sum_h w_h Delta(x_h(E)), the decomposition holds for "
          f"random lambda, and each M^lambda has the observed record law; brute force over "
          f"{stats['bf_survivors']} grid mechanisms in {stats['bf_instances']} instances agrees; every image law "
          f"lies in S_phi(P0), including the {stats['shared_fail']} instances with a shared pivot.")


# --------------------------------------------------------------------------
# Remark r4-scope: slots that host one named kernel (ties)
# --------------------------------------------------------------------------


def tied_twice_problem():
    """One kernel N (input A) called at two positions on one path; L = N1 + N2."""
    names = ["A", "N1", "N2"]
    ranges = {"A": ("go", "retry"), "N1": (0, 1), "N2": (0, 1)}
    inputs = {"A": (), "N1": ("A",), "N2": ("A",)}
    tables = {"A": {(): point("go")}, "N1": {("go",): point(0), ("retry",): bern(HALF)},
              "N2": {("go",): point(0), ("retry",): bern(HALF)}}
    base = mech_from_tables(names, inputs, tables)
    rep = {"A": ((), lambda: point("retry"))}
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("N1", "N2"), loss=lambda env: env["N1"] + env["N2"], S_r=(), r=(),
                record=tuple(names), f0=lambda: "go", fk=lambda: "retry")


def tied_policy_problem():
    """An agent policy pi reads (type, last message) at two steps; the replacement blocks the second step.

    Type b sees an error at step 1 with probability 1/2, so pi(b, err) is supported at the first slot and
    null at the second; pi(a, err) is null at both.  L = 1{U2 risky}.
    """
    names = ["W", "Z1", "U1", "G", "Z2", "U2"]
    ranges = {"W": ("a", "b"), "Z1": ("ok", "err"), "U1": ("safe", "risky"), "G": ("exec", "block"),
              "Z2": ("ok", "err"), "U2": ("safe", "risky")}
    inputs = {"W": (), "Z1": ("W",), "U1": ("W", "Z1"), "G": (), "Z2": ("G",), "U2": ("W", "Z2")}
    pi = {("a", "ok"): point("safe"), ("b", "ok"): point("safe"),
          ("b", "err"): {"risky": THIRD, "safe": 1 - THIRD}, ("a", "err"): {"risky": HALF, "safe": HALF}}
    tables = {"W": {(): {"a": HALF, "b": HALF}}, "Z1": {("a",): point("ok"), ("b",): {"ok": HALF, "err": HALF}},
              "U1": dict(pi), "G": {(): point("exec")}, "Z2": {("exec",): point("ok"), ("block",): point("err")},
              "U2": dict(pi)}
    declared = {"Z2": {("block",): point("err")}}
    base = mech_from_tables(names, inputs, tables)
    rep = {"G": ((), lambda: point("block"))}
    return dict(names=names, inputs=inputs, ranges=ranges, D=("G",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("U2",), loss=lambda env: 1 if env["U2"] == "risky" else 0,
                S_r=(), r=(), record=tuple(names), declared=declared, f0=lambda: "exec", fk=lambda: "block")


def check_tied_slots():
    # (1) One law y at two positions on a path: P(L=1) = 2y(1-y); point masses give 0, maximum 1/2 at y = 1/2.
    P = tied_twice_problem()
    rlaw = record_law(P["base"], P["record"])
    blocks = null_blocks(P, rlaw)
    assert blocks == [("N1", ("retry",)), ("N2", ("retry",))]
    untied = sorted(g_lambda_formula(P, rlaw, lam).get(1, F(0)) for lam in vertex_families(P, blocks))
    assert (untied[0], untied[-1]) == (0, 1)
    for k in range(13):
        y = F(k, 12)
        q = g_lambda_formula(P, rlaw, {blocks[0]: bern(y), blocks[1]: bern(y)})
        assert q.get(1, F(0)) == 2 * y * (1 - y) <= HALF
    grid = {"A": [point("go"), point("retry")], "N1": [bern(F(k, 6)) for k in range(7)],
            "N2": [bern(F(k, 6)) for k in range(7)]}
    tied = brute_force(P, grid, rlaw, ties={"N2": "N1"})
    tv = sorted(replacement_law(P, mech_from_tables(P["names"], P["inputs"], t)).get(1, F(0)) for t in tied)
    untied_bf = brute_force(P, grid, rlaw)
    uv = sorted(replacement_law(P, mech_from_tables(P["names"], P["inputs"], t)).get(1, F(0)) for t in untied_bf)
    assert (tv[0], tv[-1]) == (0, HALF) and (uv[0], uv[-1]) == (0, 1)
    # Every tied mechanism is in the untied class with the same record law: the per-slot image is an outer set.
    assert all(t in untied_bf for t in tied)
    # (2) A tie pins the null argument (b, err) of the second slot to the supported law at the first slot.
    P2 = tied_policy_problem()
    rlaw2 = record_law(P2["base"], P2["record"])
    blocks2 = null_blocks(P2, rlaw2)
    assert sorted(blocks2) == [("U2", ("a", "err")), ("U2", ("b", "err"))]
    unt = sorted(g_lambda_formula(P2, rlaw2, lam).get(1, F(0)) for lam in vertex_families(P2, blocks2))
    assert (unt[0], unt[-1]) == (0, 1)
    gp = [point("safe"), {"safe": 1 - THIRD, "risky": THIRD}, {"safe": HALF, "risky": HALF}, point("risky")]
    grids2 = {"W": [{"a": HALF, "b": HALF}, point("a")], "Z1": [point("ok"), {"ok": HALF, "err": HALF}],
              "U1": gp, "G": [point("exec"), point("block")], "Z2": [point("ok"), point("err")], "U2": gp}
    tied2 = brute_force(P2, grids2, rlaw2, ties={"U2": "U1"})
    t2 = sorted(replacement_law(P2, mech_from_tables(P2["names"], P2["inputs"], t)).get(1, F(0)) for t in tied2)
    untied2 = brute_force(P2, grids2, rlaw2)
    u2 = sorted(replacement_law(P2, mech_from_tables(P2["names"], P2["inputs"], t)).get(1, F(0)) for t in untied2)
    assert (t2[0], t2[-1]) == (F(1, 6), F(2, 3)) and (u2[0], u2[-1]) == (0, 1)
    pinned = {g_lambda_formula(P2, rlaw2, {("U2", ("a", "err")): lam_a,
                                           ("U2", ("b", "err")): {"risky": THIRD, "safe": 1 - THIRD}}).get(1, F(0))
              for lam_a in (point("safe"), point("risky"))}
    assert pinned == {F(1, 6), F(2, 3)}
    print(f"PASS Remark r4-scope (tied slots): one kernel called twice on a path at one null input with "
          f"L = N1 + N2 gives P(L=1) = 2y(1-y) for the tied law, 0 at both point masses and maximum 1/2 at y = 1/2 "
          f"(brute force over the tied grid class: [0, 1/2]; untied slots: [0, 1]); a policy read at two steps "
          f"whose second-step error input is supported at the first step gives harm probability [1/6, 2/3] "
          f"under the tie ({len(tied2)} tied grid mechanisms) against [0, 1] for untied slots "
          f"({len(untied2)} grid mechanisms); every tied mechanism lies in the untied class.")


# --------------------------------------------------------------------------
# The class keeps the values the specification fixes
# --------------------------------------------------------------------------


def abort_problem(with_retry, keep_declared):
    """A stopping gate with a declared abort valuation V = 1; optionally also a retry into a null input."""
    us = (0, 1, 2) if with_retry else (0, 1)
    names = ["U", "A", "V"]
    ranges = {"U": us, "A": ("go", "abort", "retry"), "V": (0, 1)}
    inputs = {"U": (), "A": ("U",), "V": ("A",)}
    tables = {"U": {(): uniform_on(us)}, "A": {(u,): point("go") for u in us},
              "V": {("go",): point(0), ("abort",): point(1), ("retry",): bern(HALF)}}
    base = mech_from_tables(names, inputs, tables)
    fk = lambda u: {0: "go", 1: "abort", 2: "retry"}[u]
    rep = {"A": (("U",), lambda u: point(fk(u)))}
    declared = {"V": {("abort",): point(1)}} if keep_declared else {}
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base, tables=tables,
                kappa=base.replace(rep), S_L=("V",), loss=lambda env: env["V"], S_r=(), r=(),
                record=tuple(names), declared=declared)


def check_declared_values_kept():
    out = {}
    for with_retry in (False, True):
        for keep in (True, False):
            P = abort_problem(with_retry, keep)
            rlaw = record_law(P["base"], P["record"])
            blocks = null_blocks(P, rlaw)
            vals = sorted(g_lambda_formula(P, rlaw, lam).get(1, F(0)) for lam in vertex_families(P, blocks))
            grids = {"U": [P["tables"]["U"][()]], "A": [point("go"), point("abort"), point("retry")],
                     "V": [bern(F(k, 4)) for k in range(5)]}
            bf = sorted(replacement_law(P, mech_from_tables(P["names"], P["inputs"], t)).get(1, F(0))
                        for t in brute_force(P, grids, rlaw))
            assert (vals[0], vals[-1]) == (bf[0], bf[-1])
            out[(with_retry, keep)] = (vals[0], vals[-1])
    assert out[(False, True)] == (HALF, HALF) and out[(False, False)] == (0, HALF)
    assert out[(True, True)] == (THIRD, 2 * THIRD) and out[(True, False)] == (0, 2 * THIRD)
    print("PASS declared values kept: a stopping gate with a declared abort valuation has identified harm "
          "probability 1/2, and freeing the valuation at its input of baseline probability zero gives [0, 1/2]; "
          "with a retry that also fails (G1), the class that keeps the valuation gives [1/3, 2/3] and the class "
          "that frees it gives [0, 2/3]; brute force over grid mechanisms agrees in all four cases.")


# --------------------------------------------------------------------------
# Block and retry with mixed support
# --------------------------------------------------------------------------


def block_retry_problem(message, x_b=HALF, p_b=HALF):
    """Gate A blocks when the monitor score Y is high; the agent retries (U2 reads the type W and the message).

    message = "err": the block is reported through the tool-error message, which
    baseline operation shows for task a (flaky tool) but never for task b.
    message = "blk": the block has its own message, never seen in baseline.
    """
    names = ["W", "U1", "Y", "A", "Msg", "U2"]
    e_range = ("ok", "err") if message == "err" else ("ok", "err", "blk")
    ranges = {"W": ("a", "b"), "U1": ("safe", "risky"), "Y": ("lo", "hi"), "A": ("exec", "block"),
              "Msg": e_range, "U2": ("none", "safe", "risky")}
    inputs = {"W": (), "U1": ("W",), "Y": ("U1",), "A": ("Y",), "Msg": ("W", "U1", "A"), "U2": ("W", "Msg")}
    tables = {
        "W": {(): clean({"a": 1 - p_b, "b": p_b})},
        "U1": {(t,): {"safe": HALF, "risky": HALF} for t in ("a", "b")},
        "Y": {("risky",): {"hi": F(3, 4), "lo": F(1, 4)}, ("safe",): {"hi": F(1, 4), "lo": F(3, 4)}},
        "A": {(m,): point("exec") for m in ("lo", "hi")},
        "Msg": {},
        "U2": {},
    }
    for t, u in product(("a", "b"), ("safe", "risky")):
        tables["Msg"][(t, u, "block")] = point(message)               # declared by the specification
        tables["Msg"][(t, u, "exec")] = {"err": F(1, 4), "ok": F(3, 4)} if t == "a" else point("ok")
    for t, e in product(("a", "b"), e_range):
        if e == "ok":
            tables["U2"][(t, e)] = point("none")
        elif (t, e) == ("a", "err"):
            tables["U2"][(t, e)] = {"risky": THIRD, "safe": 1 - THIRD}
        else:
            tables["U2"][(t, e)] = clean({"risky": x_b, "safe": 1 - x_b})   # never reached at baseline
    declared = {"Msg": {(t, u, "block"): point(message) for t, u in product(("a", "b"), ("safe", "risky"))}}
    base = mech_from_tables(names, inputs, tables)
    fk = lambda m: "block" if m == "hi" else "exec"
    rep = {"A": (("Y",), lambda m: point(fk(m)))}
    harm = lambda env: 1 if (env["U1"] == "risky" and env["A"] == "exec") or env["U2"] == "risky" else 0
    return dict(names=names, inputs=inputs, ranges=ranges, D=("A",), rep=rep, base=base,
                kappa=base.replace(rep), S_L=("U1", "A", "U2"), loss=harm, S_r=(), r=(),
                record=tuple(names), declared=declared, tables=tables,
                f0=lambda m: "exec", fk=fk)


def check_block_and_retry():
    P = block_retry_problem("err")
    base, record = P["base"], P["record"]
    rlaw = record_law(base, record)
    known = known_from_declared(P["declared"])
    cond = vs.conditions(base, P["kappa"], {"A"}, P["S_L"], record, known=known)
    assert cond == {"G1": False, "G2": True, "G3": True, "G4": True}
    try:
        vs.g_formula(record, rlaw, base, P["kappa"], {"A"}, P["S_L"], P["loss"], known=known)
        raise AssertionError("support failure not detected")
    except vs.PositivityError:
        pass
    blocks = null_blocks(P, rlaw)
    assert blocks == [("U2", ("b", "err"))]
    harms = sorted(prob_of(g_lambda_formula(P, rlaw, lam), {1}) for lam in vertex_families(P, blocks))
    ident = (harms[0], harms[-1])
    assert ident == (F(43, 192), F(91, 192))
    # Brute force over grid mechanisms with the observed baseline law (true values in the grid).
    u2_grid = [point("none"), point("safe"), point("risky"), {"risky": THIRD, "safe": 1 - THIRD},
               {"risky": HALF, "safe": HALF}, {"none": THIRD, "safe": THIRD, "risky": THIRD}]
    grids = {"W": [{"a": HALF, "b": HALF}, point("a"), point("b")],
             "U1": [{"safe": HALF, "risky": HALF}, point("safe"), point("risky")],
             "Y": [{"hi": F(3, 4), "lo": F(1, 4)}, {"hi": F(1, 4), "lo": F(3, 4)}, point("hi"), point("lo")],
             "A": [point("exec"), point("block")],
             "Msg": [point("ok"), point("err"), {"err": F(1, 4), "ok": F(3, 4)}],
             "U2": u2_grid}
    survivors = brute_force(P, grids, rlaw)
    assert P["tables"] in survivors and len(survivors) == len(u2_grid)
    bf = []
    for t in survivors:
        Q_M = replacement_law(P, mech_from_tables(P["names"], P["inputs"], t))
        assert Q_M == g_lambda_formula(P, rlaw, true_lam(P, t, blocks))
        bf.append(Q_M.get(1, F(0)))
    assert (min(bf), max(bf)) == ident
    # Theorem thm:replay(iii) without (S): nu, the continuing mass, and X_h from the baseline law and the two maps.
    nu, w, X, S_vertices = theorem41_outer(P)
    eps = sum(w.values())
    assert nu.get(1) == F(27, 192) and eps == HALF and all(v == {0, 1} for v in X.values())
    outer = (nu.get(1), nu.get(1) + eps)
    assert outer == (F(27, 192), F(123, 192)) and outer[0] < ident[0] and ident[1] < outer[1]
    # The identified width is the blocked type-b mass P(W=b)/2; the outer width stays 1/2.
    widths = {}
    for p in (F(1, 4), HALF, F(3, 4)):
        Pp = block_retry_problem("err", p_b=p)
        rl = record_law(Pp["base"], Pp["record"])
        bl = null_blocks(Pp, rl)
        hs = sorted(prob_of(g_lambda_formula(Pp, rl, lam), {1}) for lam in vertex_families(Pp, bl))
        nu_p, w_p, X_p, _ = theorem41_outer(Pp)
        assert hs[-1] - hs[0] == p / 2 and sum(w_p.values()) == HALF
        assert nu_p.get(1) == F(1, 8) + (1 - p) / 32 and hs[0] == nu_p.get(1) + (1 - p) / 6
        widths[p] = hs[-1] - hs[0]
    # Treating the supported retry input (a, err) as unsupported is conservative: the set it gives contains the
    # identified interval.  It also discards the agreement evidence at that input, so it can exceed the outer bound.
    blocks_x = blocks + [("U2", ("a", "err"))]
    harms_x = sorted(prob_of(g_lambda_formula(P, rlaw, lam, force_free={("U2", ("a", "err"))}), {1})
                     for lam in vertex_families(P, blocks_x))
    enlarged = (harms_x[0], harms_x[-1])
    true_value = prob_of(g_lambda_formula(P, rlaw, {**true_lam(P, P["tables"], blocks),
                                                    ("U2", ("a", "err")): P["tables"]["U2"][("a", "err")]},
                                          force_free={("U2", ("a", "err"))}), {1})
    assert true_value == prob_of(replacement_law(P, base), {1})
    assert enlarged == (F(1, 8), F(43, 64)) and enlarged[0] < ident[0] and ident[1] < enlarged[1]
    # A distinct block message: every retry reaches an unseen input, and the interval is the outer bound.
    Pb = block_retry_problem("blk")
    rlaw_b = record_law(Pb["base"], Pb["record"])
    blocks_b = null_blocks(Pb, rlaw_b)
    # (b, err) is null too, but no path reaches it once the block has its own message.
    assert set(blocks_b) == {("U2", ("a", "blk")), ("U2", ("b", "err")), ("U2", ("b", "blk"))}
    harms_b = sorted(prob_of(g_lambda_formula(Pb, rlaw_b, lam), {1}) for lam in vertex_families(Pb, blocks_b))
    assert (harms_b[0], harms_b[-1]) == outer
    print(f"PASS block and retry with mixed support: the gate blocks high-score calls and reports the block "
          f"through the tool-error message; retries after a block reuse the logged retry-after-error behavior for "
          f"task a and reach an unseen input for task b; (G1) fails and Proposition prop:supported does not apply; the "
          f"identified harm probability is [{ident[0]}, {ident[1]}] (brute force over {len(survivors)} compatible "
          f"grid mechanisms agrees), strictly inside the Theorem thm:replay(iii) outer bound [{outer[0]}, {outer[1]}] "
          f"(continuing mass 1/2); the identified width is P(W=b)/2 "
          f"({', '.join(f'{widths[p]} at P(W=b)={p}' for p in sorted(widths))}) while the outer width stays 1/2; "
          f"giving the block its own message makes every retry input unseen and returns the outer bound; "
          f"treating the logged input (a, err) as unsupported gives the valid but wider set "
          f"[{enlarged[0]}, {enlarged[1]}], which contains the true value.")


if __name__ == "__main__":
    check_sharpness_random()
    check_restricted_allowed_sets()
    check_nonconvex_image()
    check_shared_input_t114()
    check_saturated_extreme()
    check_pivot_without_onto()
    check_tied_slots()
    check_declared_values_kept()
    check_block_and_retry()

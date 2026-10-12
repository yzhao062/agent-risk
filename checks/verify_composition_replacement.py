#!/usr/bin/env python
"""Exact finite checks for Proposition r4-composition: composition through shared
state as the admissible replacement kappa_conn of a joint specification whose
baseline runs the standalone deployments as independent sub-episodes.

Every check builds explicit finite mechanisms (named kernels with declared
inputs, in generation order) and computes every reported number with exact
rational arithmetic (fractions.Fraction); no floats are used.  Identification
formulas (the g-formula and its free-law form at null inputs) receive only the
joint baseline record law, the declared inputs, and the known replacement
kernels; they never read a component kernel.

Model used in the simple instances.  Component i has state coordinates D_i
(what it reads from outside itself: budget, cache, provider status, an incoming
handoff) and a component block C_i (private random input X_i, output O_i, loss
L_i, and the recorded output Y_i = (O_i, L_i)).  The joint baseline runs the
standalone deployments as independent sub-episodes in the composition's
declared order; kappa_conn replaces the state kernels by the composition's
state kernels, which may read other components' recorded outputs Y_k, and
leaves every component kernel unchanged.  The "collapsed" list replaces each
block C_i by one kernel from D_i to Y_i (Lemma r4-block).

check_general uses a more general generator: one or two interface coordinates
per component, binary or ternary ranges, component kernels that read random
subsets of the interface and of earlier block coordinates, random
interleavings, and replaced state kernels that read arbitrary earlier
coordinates.  There the output set Y_i is computed from the declared inputs, as
in the appendix (every block coordinate read from outside the block under
either mechanism, plus the loss).

Declared standalone state.  The appendix declares, for each component, a map
S^sa_i on the joint list evaluated under the joint baseline P^0.  The checks
name the declaration they use: the joint-baseline declaration (the
composition's map evaluated under P^0), the interface declaration (S read from
component i's own interface through a bijective psi_i), or a declared map such
as the scheduler's map applied to component i's own standalone outputs.

The only imports from the paper are checks/verify_supported.py (mechanism
engine, relevant set, (G1)-(G4) test, g-formula) and checks/verify_coupling.py
(exact LP over the coupling set Gamma), loaded by path.  The script looks for
them next to itself first, so it can be copied into checks/ unchanged.

Run: python verify_composition_replacement.py


Paper map: Proposition prop:r4-composition, Remark rem:r4-standalone-inv, and
Lemmas r4-block and r4-product are archived in supplement/appendix-G.md (not printed
in the paper). The paper assumes conditional-law invariance directly. Checks
referring to budget, stale-cache, and outage replacement examples document
supplementary material; checks/README.md summarizes them.
"""

import importlib.util
import random
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path



def _load(name):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vs = _load("verify_supported")
vc = _load("verify_coupling")
Mechanism = vs.Mechanism
_uncached_law = vs.Mechanism.law


def _cached_law(self):
    """Memoize the enumerated law; mechanisms are never mutated after construction."""
    if "_law_cache" not in self.__dict__:
        self._law_cache = _uncached_law(self)
    return self._law_cache


vs.Mechanism.law = _cached_law


def _cached_envs(self):
    if "_env_cache" not in self.__dict__:
        self._env_cache = [(dict(zip(self.names, path)), p) for path, p in self.law().items()]
    return self._env_cache


vs.Mechanism.envs = _cached_envs

BIN = (0, 1)
QV = (0, 1, 2)
YV = tuple(product(BIN, BIN))


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def clean(d):
    return {k: v for k, v in d.items() if v != 0}


def point(v):
    return {v: F(1)}


def bern(p):
    p = F(p)
    return clean({0: 1 - p, 1: p})


def from_table(table):
    return lambda *z: table[tuple(z)]


def law_of(mech, f, cond=lambda env: True):
    out = defaultdict(F)
    for env, p in mech.envs():
        if cond(env):
            out[f(env)] += p
    return clean(out)


def conditional(mech, f_given, f_val):
    """{g: {v: P(val = v | given = g)}} for every g of positive probability."""
    joint = law_of(mech, lambda e: (f_given(e), f_val(e)))
    marg = defaultdict(F)
    for (g, _), p in joint.items():
        marg[g] += p
    out = defaultdict(dict)
    for (g, v), p in joint.items():
        out[g][v] = p / marg[g]
    return dict(out)


def rec_law(mech, record):
    return vs.marginal(mech.names, mech.law(), list(record))


def rand_law(rng, values, zero=True):
    while True:
        w = [rng.choice((0, 1, 2, 3)) if zero else rng.choice((1, 2, 3)) for _ in values]
        if sum(w):
            break
    t = sum(w)
    return clean({v: F(x, t) for v, x in zip(values, w)})


def rand_table(rng, in_ranges, values, zero=True):
    return {z: rand_law(rng, values, zero) for z in product(*in_ranges)}


def func_map(mech, f_from, f_to):
    """If f_to is mech-almost surely a function of f_from, return that map on the
    support of f_from; otherwise None."""
    out = {}
    for env, p in mech.envs():
        a, b = f_from(env), f_to(env)
        if out.setdefault(a, b) != b:
            return None
    return out


def ci_holds(mech, S, losses):
    joint = conditional(mech, S, lambda e: tuple(e[x] for x in losses))
    margs = [conditional(mech, S, lambda e, x=x: e[x]) for x in losses]
    for s, law in joint.items():
        for ls in product(*[list(m[s]) for m in margs]):
            prod_p = F(1)
            for m, l in zip(margs, ls):
                prod_p *= m[s][l]
            if law.get(ls, F(0)) != prod_p:
                return False
    return True


# --------------------------------------------------------------------------
# Joint specification: standalone sub-episodes, kappa_conn, and collapse
# --------------------------------------------------------------------------


def comp_nodes(i, c):
    """Component block C_i: private X_i, output O_i, loss L_i, recorded Y_i."""
    return [(f"X{i}", (f"q{i}",), from_table(c["X"])),
            (f"O{i}", (f"q{i}", f"X{i}"), from_table(c["O"])),
            (f"L{i}", (f"q{i}", f"O{i}", f"X{i}"), from_table(c["L"])),
            (f"Y{i}", (f"O{i}", f"L{i}"), lambda o, l: {(o, l): F(1)})]


def block_names(i):
    return [f"X{i}", f"O{i}", f"L{i}", f"Y{i}"]


def make_instance(rng, n, sched):
    """Random joint specification with n components.

    sched 'seq': q1, C1, q2, C2, ...; kappa_conn sets q_k from Y_{k-1}.
    sched 'par': q1..qn first (q_k reads q_{k-1}), then the blocks, interleaved.
    sched 'seq2': as 'seq', but q3 reads Y1 and Y2 (two components).
    """
    comps = {}
    for i in range(1, n + 1):
        comps[i] = dict(q_sa=rand_law(rng, QV),
                        X=rand_table(rng, [QV], BIN),
                        O=rand_table(rng, [QV, BIN], BIN),
                        L=rand_table(rng, [QV, BIN, BIN], BIN))
    repl = {}
    for k in range(1, n + 1):
        if sched == "par":
            ins, rin = ((), []) if k == 1 else ((f"q{k-1}",), [QV])
        elif sched == "seq" or (sched == "seq2" and k < 3):
            ins, rin = ((), []) if k == 1 else ((f"Y{k-1}",), [YV])
        else:
            ins, rin = ((f"Y{k-2}", f"Y{k-1}"), [YV, YV])
        repl[f"q{k}"] = (ins, from_table(rand_table(rng, rin, QV)))
    if sched == "par":
        order = [f"q{k}" for k in range(1, n + 1)]
        queues = {i: block_names(i) for i in range(1, n + 1)}
        while any(queues.values()):
            i = rng.choice([i for i in queues if queues[i]])
            order.append(queues[i].pop(0))
    else:
        order = []
        for i in range(1, n + 1):
            order += [f"q{i}"] + block_names(i)
    nodes = {}
    for i in range(1, n + 1):
        nodes[f"q{i}"] = (f"q{i}", (), from_table({(): comps[i]["q_sa"]}))
        for node in comp_nodes(i, comps[i]):
            nodes[node[0]] = node
    base = Mechanism([nodes[name] for name in order])
    kappa = base.replace(repl)
    sa = {i: Mechanism([nodes[f"q{i}"]] + comp_nodes(i, comps[i])) for i in range(1, n + 1)}
    return dict(n=n, sched=sched, comps=comps, repl=repl, base=base, kappa=kappa, sa=sa,
                D={f"q{k}" for k in range(1, n + 1)})


def block_kernel(mech, block, d_names, y_name):
    """K^Y(y | d): the block's product of kernels summed over its internal coordinates."""
    cache = {}

    def K(*d):
        if d in cache:
            return cache[d]
        dist = {(): F(1)}
        for name in block:
            ins, kern = mech.inputs[name], mech.kernels[name]
            assert set(ins) <= set(d_names) | set(block), f"{name} reads outside its block"
            new = defaultdict(F)
            for path, p in dist.items():
                env = dict(zip(d_names, d))
                env.update(zip(block, path))
                for v, q in kern(*[env[x] for x in ins]).items():
                    if q:
                        new[path + (v,)] += p * q
            dist = new
        pos = block.index(y_name)
        out = defaultdict(F)
        for path, p in dist.items():
            out[path[pos]] += p
        cache[d] = clean(out)
        return cache[d]
    return K


def collapse(mech, blocks):
    """Lemma r4-block: replace each block (C_i, D_i, Y_i) by one kernel from D_i to Y_i,
    placed at the position of the first coordinate of C_i."""
    first = {}
    kern = {}
    for block, d_names, y in blocks:
        first[block[0]] = (y, tuple(d_names))
        kern[y] = block_kernel(mech, block, list(d_names), y)
    skip = {name for block, _, _ in blocks for name in block}
    nodes = []
    for name in mech.names:
        if name in first:
            y, d_names = first[name]
            nodes.append((y, d_names, kern[y]))
        elif name not in skip:
            nodes.append((name, mech.inputs[name], mech.kernels[name]))
    return Mechanism(nodes)


def collapsed_pair(inst):
    blocks = [(block_names(i), [f"q{i}"], f"Y{i}") for i in range(1, inst["n"] + 1)]
    base_c = collapse(inst["base"], blocks)
    kappa_c = base_c.replace(inst["repl"])
    return base_c, kappa_c


def gq_formula(record, rlaw, base, kappa, D, S_L, loss, free):
    """Generalized g-formula (Proposition r4-unsupported): the factor of an unchanged
    kernel at an input of baseline probability zero is the free law free(j, input)."""
    J, dsc = vs.relevant_set(base, kappa, D, S_L, ())
    assert J <= set(record)
    order = [n for n in base.names if n in J]
    names = [n for n in order if n not in dsc]
    states = dict(vs.marginal(list(record), rlaw, names))
    for j in [n for n in order if n in dsc]:
        new = defaultdict(F)
        if j in D:
            ins, kern = kappa.inputs[j], kappa.kernels[j]
            factor = lambda env, ins=ins, kern=kern: kern(*[env[i] for i in ins])
        else:
            ins = base.inputs[j]
            joint = vs.marginal(list(record), rlaw, list(ins) + [j])
            marg = vs.marginal(list(record), rlaw, list(ins))

            def factor(env, ins=ins, joint=joint, marg=marg, j=j):
                z = tuple(env[i] for i in ins)
                if marg.get(z, F(0)) > 0:
                    return {key[-1]: w / marg[z] for key, w in joint.items() if key[:-1] == z}
                return free(j, z)
        for vals, p in states.items():
            env = dict(zip(names, vals))
            for v, w in factor(env).items():
                if w:
                    new[vals + (v,)] += p * w
        names.append(j)
        states = dict(new)
    out = defaultdict(F)
    for vals, p in states.items():
        out[loss(dict(zip(names, vals)))] += p
    return clean(out)


# --------------------------------------------------------------------------
# Lemma r4-product and Lemma r4-block
# --------------------------------------------------------------------------


def check_product_and_collapse(n_inst=150, seed=20261004):
    rng = random.Random(seed)
    n_paths = 0
    for t in range(n_inst):
        inst = make_instance(rng, rng.choice((2, 3)), rng.choice(("seq", "par", "seq2")))
        base, n = inst["base"], inst["n"]
        sa_laws = {i: inst["sa"][i].law() for i in range(1, n + 1)}
        sa_names = {i: inst["sa"][i].names for i in range(1, n + 1)}
        total = F(0)
        for env, p in base.envs():                         # Lemma r4-product (a)
            prod_p = F(1)
            for i in range(1, n + 1):
                prod_p *= sa_laws[i].get(tuple(env[x] for x in sa_names[i]), F(0))
            assert p == prod_p
            total += prod_p
            n_paths += 1
        assert total == 1
        record = [x for i in range(1, n + 1) for x in (f"q{i}", f"Y{i}")]
        rl = rec_law(base, record)                          # Lemma r4-product (b)
        sa_rec = {i: rec_law(inst["sa"][i], [f"q{i}", f"Y{i}"]) for i in range(1, n + 1)}
        for key, p in rl.items():
            prod_p = F(1)
            for i in range(1, n + 1):
                prod_p *= sa_rec[i].get((key[2 * i - 2], key[2 * i - 1]), F(0))
            assert p == prod_p
        assert sum(rl.values()) == 1
        # Interleaving does not matter: the sequential order gives the same law.
        if inst["sched"] == "par":
            seq_order = [x for i in range(1, n + 1) for x in [f"q{i}"] + block_names(i)]
            seq_base = Mechanism([(x, base.inputs[x], base.kernels[x]) for x in seq_order])
            assert law_of(seq_base, lambda e: tuple(sorted(e.items()))) == \
                law_of(base, lambda e: tuple(sorted(e.items())))
        # Lemma r4-block: marginal of the full list on (D, Y) equals the collapsed list,
        # under the baseline and under kappa_conn.
        base_c, kappa_c = collapsed_pair(inst)
        keep = [x for x in base_c.names]
        for full, col in ((base, base_c), (inst["kappa"], kappa_c)):
            assert vs.marginal(full.names, full.law(), keep) == vs.marginal(col.names, col.law(), keep)
    print(f"PASS product baseline and block collapse: in {n_inst} random joint specifications "
          f"(2-3 components; sequential, parallel interleaved, and two-reader schedulers; "
          f"{n_paths} baseline paths) the joint baseline law is the product of the standalone "
          f"laws, its record law is the product of the standalone record laws, the interleaving "
          f"does not change it, and collapsing each component to one kernel from its state to "
          f"its recorded output preserves the law of (state, recorded outputs) under both the "
          f"baseline and kappa_conn.")


# --------------------------------------------------------------------------
# Claim (i): (Inv_i) from kernel invariance, and when S is read
# --------------------------------------------------------------------------


def check_inv(n_inst=150, seed=7):
    rng = random.Random(seed)
    checked = full_par = lit_par = seq_fail = seq_cand = seq_T = 0
    for t in range(n_inst):
        sched = ("seq", "par")[t % 2]
        inst = make_instance(rng, 3, sched)
        n, base, kappa = inst["n"], inst["base"], inst["kappa"]
        for i in range(1, n + 1):
            K = block_kernel(inst["sa"][i], block_names(i), [f"q{i}"], f"Y{i}")

            def G(s, K=K):
                out = defaultdict(F)
                for (o, l), p in K(s).items():
                    out[l] += p
                return clean(out)
            # (i)(a) and (b) with S = q_i (psi_i the identity; the interface declaration).
            Fi = conditional(kappa, lambda e, i=i: e[f"q{i}"], lambda e, i=i: e[f"L{i}"])
            Qi = conditional(inst["sa"][i], lambda e, i=i: e[f"q{i}"], lambda e, i=i: e[f"L{i}"])
            for s, law in Fi.items():
                assert law == G(s)
                checked += 1
            for s, law in Qi.items():
                assert law == G(s)
            # Full shared state S = (q_1, ..., q_n), psi_i the projection to q_i.
            Sfull = lambda e: tuple(e[f"q{k}"] for k in range(1, n + 1))
            Ffull = conditional(kappa, Sfull, lambda e, i=i: e[f"L{i}"])
            if sched == "par":
                # Every state coordinate precedes every block, so (T_i) holds; (i)(a).
                for s, law in Ffull.items():
                    assert law == G(s[i - 1])
                    full_par += 1
                # (i)(c) with the joint-baseline declaration: S evaluated under P^0 is a
                # function of the coordinates before C_i and psi_i(S) = q_i identically,
                # so the literal (Inv_i) of def:p3-assumptions holds with G(psi_i(.)).
                Q0 = conditional(base, Sfull, lambda e, i=i: e[f"L{i}"])
                for s, law in Q0.items():
                    assert law == G(s[i - 1])
                    lit_par += 1
            elif i < n:
                # S also holds q_{i+1}, written from Y_i after component i acts.
                W = [x for x in base.names[:base.names.index(f"X{i}")]]
                T = func_map(kappa, lambda e, W=W: tuple(e[x] for x in W), Sfull) is not None
                broken = any(law != G(s[i - 1]) for s, law in Ffull.items())
                if T:
                    assert not broken
                    seq_T += 1
                else:
                    seq_cand += 1
                    seq_fail += broken
    assert seq_fail > 0

    # (i)(e): the state is generated after component 2 acts and reads it.
    # Component 2: private X ~ Bern(1/2), L2 = X; its own state q2 is constant 0.
    # kappa_conn writes q1 = X and B (B a fresh fair bit drawn by the state kernel);
    # the joint baseline (joint-baseline declaration) has q1 an independent fair bit.
    def mech(q1_kernel, q1_inputs):
        return Mechanism([("q2", (), lambda: point(0)),
                          ("X2", ("q2",), lambda q: bern(F(1, 2))),
                          ("L2", ("X2",), lambda x: point(x)),
                          ("q1", q1_inputs, q1_kernel)])
    sa_c = mech(lambda: bern(F(1, 2)), ())
    comp = mech(lambda x: {0: F(1) - F(x, 2), 1: F(x, 2)} if x else point(0), ("X2",))
    S = lambda e: (e["q1"], e["q2"])
    Fc = conditional(comp, S, lambda e: e["L2"])
    Fs = conditional(sa_c, S, lambda e: e["L2"])
    psi2 = func_map(comp, S, lambda e: e["q2"])
    assert psi2 == {(0, 0): 0, (1, 0): 0}                     # (Rd_2), psi_2 not injective
    assert set(law_of(comp, lambda e: e["q2"])) <= set(law_of(sa_c, lambda e: e["q2"]))  # (Supp^Sigma_2)
    assert set(Fc) == set(Fs) == {(0, 0), (1, 0)}           # (Supp_2), joint-baseline declaration
    G20 = law_of(sa_c, lambda e: e["L2"])                    # G_2(.|0)
    assert G20[1] == F(1, 2)
    assert Fc[(0, 0)][1] == F(1, 3) and Fs[(0, 0)][1] == F(1, 2)   # identity of (a) and (Inv_2) fail
    assert Fc[(1, 0)][1] == F(1) and Fs[(1, 0)][1] == F(1, 2)

    # A bijective psi_i with S held by another component: S = q1, kappa sets q2 := q1, and
    # component 2 reads q2 only.  The joint-baseline declaration (S = standalone q1, which is
    # independent of component 2's run) breaks the literal (Inv_2); the interface declaration
    # S^sa_2 = psi_2^{-1}(q2) = q2 satisfies (i)(c).
    g = {0: F(1, 10), 1: F(3, 5)}
    bj = Mechanism([("q1", (), lambda: bern(F(1, 4))),
                    ("q2", (), lambda: bern(F(1, 2))),
                    ("L2", ("q2",), lambda q: bern(g[q]))])
    bk = bj.replace({"q2": (("q1",), lambda a: point(a))})
    Fb = conditional(bk, lambda e: e["q1"], lambda e: e["L2"])
    Q_joint = conditional(bj, lambda e: e["q1"], lambda e: e["L2"])
    Q_iface = conditional(bj, lambda e: e["q2"], lambda e: e["L2"])
    assert Fb == Q_iface and Fb != Q_joint
    assert Q_joint[0][1] == Q_joint[1][1] == F(7, 20)
    print(f"PASS (i) invariance: with S read before the component acts and holding everything the "
          f"component reads from outside, the composed law of L_i given S equals G_i(.|psi_i(S)) and "
          f"the standalone law given the interface value equals G_i, both computed from the "
          f"component kernels ({checked} (component, state) pairs over {n_inst} random "
          f"3-component instances, plus {full_par} pairs with the full state tuple in parallel "
          f"schedulers); under the joint-baseline declaration the literal (Inv_i) holds on "
          f"{lit_par} standalone (component, state) pairs (part (i)(c)); with the full state under "
          f"a sequential scheduler, (T_i) fails in {seq_cand} components and the identity of (a) "
          f"fails in {seq_fail} of them ({seq_T} components where (T_i) still holds keep it); a "
          f"state written from component 2's private bit after it acts gives P(L2=1|S) = 1/3 and 1 "
          f"against G_2 = 1/2, with (Rd_2), (Supp^Sigma_2), and joint-baseline (Supp_2) holding and "
          f"(Inv_2) failing; with S held by another component (bijective psi_2) the joint-baseline "
          f"declaration gives 7/20 at both states against composed 1/10 and 3/5, while the "
          f"interface declaration gives (Inv_2).")


# --------------------------------------------------------------------------
# General random joint specifications: multi-coordinate interfaces, partial
# reads, random interleavings, state kernels reading arbitrary earlier coordinates
# --------------------------------------------------------------------------


def gen_general(rng, n):
    """Component i: interface s{i}0[, s{i}1] (binary or ternary), block x{i}, o{i}, l{i}.
    Standalone kernels read random subsets of earlier coordinates of the same
    deployment, so (Blk_i) holds and block kernels may read part of the interface.
    kappa_conn replaces a random nonempty set of state kernels by known kernels that
    read up to two arbitrary earlier coordinates of the joint list."""
    ranges, nodes, comp = {}, {}, {}
    for i in range(1, n + 1):
        m = rng.choice((1, 2)) if n == 2 else 1
        sig = [f"s{i}{a}" for a in range(m)]
        blk = [f"x{i}", f"o{i}", f"l{i}"]
        for nm in sig:
            ranges[nm] = rng.choice((BIN, QV))
        ranges[f"x{i}"], ranges[f"o{i}"] = BIN, BIN
        ranges[f"l{i}"] = rng.choice((BIN, QV)) if n == 2 else BIN
        order = sig + blk
        for pos, nm in enumerate(order):
            ins = tuple(x for x in order[:pos] if rng.randrange(2) == 0)
            nodes[nm] = (ins, from_table(rand_table(rng, [ranges[x] for x in ins], ranges[nm])))
        comp[i] = dict(sig=sig, blk=blk, order=order)
    queues = {i: list(comp[i]["order"]) for i in comp}
    gorder = []
    while any(queues.values()):
        i = rng.choice([i for i in queues if queues[i]])
        gorder.append(queues[i].pop(0))
    Sigma = [x for i in comp for x in comp[i]["sig"]]
    D = [x for x in Sigma if rng.randrange(5) < 3] or [rng.choice(Sigma)]
    repl = {}
    for j in D:
        earlier = gorder[:gorder.index(j)]
        ins = tuple(rng.sample(earlier, min(rng.choice((0, 1, 2)), len(earlier))))
        repl[j] = (ins, from_table(rand_table(rng, [ranges[x] for x in ins], ranges[j])))
    base = Mechanism([(nm,) + nodes[nm] for nm in gorder])
    kappa = base.replace(repl)
    sa = {i: Mechanism([(nm,) + nodes[nm] for nm in comp[i]["order"]]) for i in comp}
    Y = {}
    for i in comp:
        C = set(comp[i]["blk"])
        readers = set()
        for mech in (base, kappa):
            for nm in mech.names:
                if nm not in C:
                    readers |= set(mech.inputs[nm]) & C
        Y[i] = [x for x in comp[i]["blk"] if x in readers or x == f"l{i}"]
    inst = dict(n=n, comp=comp, gorder=gorder, Sigma=Sigma, D=set(D), repl=repl, base=base,
                kappa=kappa, sa=sa, Y=Y, ranges=ranges, nodes=nodes)
    inst["Kblk"] = {i: general_block_kernel(inst, i, Y[i]) for i in comp}
    return inst


def general_block_kernel(inst, i, outset):
    """k^blk_i(y | u) from component i's standalone kernels only (eq:r4-block-kernel)."""
    sig, blk, nodes = inst["comp"][i]["sig"], inst["comp"][i]["blk"], inst["nodes"]
    cache = {}

    def K(u):
        if u not in cache:
            dist = {(): F(1)}
            for nm in blk:
                ins, kern = nodes[nm]
                new = defaultdict(F)
                for path, p in dist.items():
                    env = dict(zip(sig, u))
                    env.update(zip(blk, path))
                    for v, q in kern(*[env[x] for x in ins]).items():
                        if q:
                            new[path + (v,)] += p * q
                dist = new
            out = defaultdict(F)
            for path, p in dist.items():
                env = dict(zip(blk, path))
                out[tuple(env[x] for x in outset)] += p
            cache[u] = clean(out)
        return cache[u]
    return K


def general_collapse(inst, mech):
    """Lemma r4-block(b): each block becomes one coordinate Y{i} with values in V_{Y_i},
    declared input Sigma_i, at the position of the block's first coordinate; a kernel that
    reads a coordinate of Y_i reads the tuple instead."""
    where = {}
    for i in inst["comp"]:
        for pos, y in enumerate(inst["Y"][i]):
            where[y] = (f"Y{i}", pos)
    first = {inst["comp"][i]["blk"][0]: i for i in inst["comp"]}
    blocknames = {x for i in inst["comp"] for x in inst["comp"][i]["blk"]}
    nodes = []
    for nm in mech.names:
        if nm in first:
            i = first[nm]
            K = inst["Kblk"][i]
            nodes.append((f"Y{i}", tuple(inst["comp"][i]["sig"]), lambda *u, K=K: K(tuple(u))))
        elif nm not in blocknames:
            new_ins, getters = [], []
            for x in mech.inputs[nm]:
                src, pos = where.get(x, (x, None))
                if src not in new_ins:
                    new_ins.append(src)
                getters.append((new_ins.index(src), pos))

            def k2(*vals, getters=tuple(getters), kern=mech.kernels[nm]):
                return kern(*[vals[a] if p is None else vals[a][p] for a, p in getters])
            nodes.append((nm, tuple(new_ins), k2))
    return Mechanism(nodes)


def general_retained_law(inst, mech, names):
    """Law of the full list on the retained coordinates of the collapsed list."""
    def key(e):
        out = []
        for x in names:
            if x.startswith("Y"):
                i = int(x[1:])
                out.append(tuple(e[y] for y in inst["Y"][i]))
            else:
                out.append(e[x])
        return tuple(out)
    return law_of(mech, key)


def check_general(n_inst=400, seed=31):
    rng = random.Random(seed)
    c = defaultdict(int)
    for t in range(n_inst):
        inst = gen_general(rng, 2 if t % 2 == 0 else 3)
        base, kappa, comp = inst["base"], inst["kappa"], inst["comp"]
        dsc = vs.descendants(base, inst["D"])
        # Lemma r4-block (b): the collapsed lists have the retained marginals.
        base_c, kappa_c = general_collapse(inst, base), general_collapse(inst, kappa)
        for full, col in ((base, base_c), (kappa, kappa_c)):
            assert general_retained_law(inst, full, col.names) == \
                law_of(col, lambda e, col=col: tuple(e[x] for x in col.names))
        # A random shared state: a nonempty subset of Sigma, read as a tuple or through a
        # random (often non-injective) map varsigma.
        SS = [x for x in inst["Sigma"] if rng.randrange(5) < 3] or [rng.choice(inst["Sigma"])]
        if rng.randrange(2) == 0:
            vsig = lambda z: z
        else:
            tab = {z: rng.choice(QV) for z in product(*[inst["ranges"][x] for x in SS])}
            vsig = lambda z, tab=tab: tab[z]
        S = lambda e, SS=tuple(SS), vsig=vsig: vsig(tuple(e[x] for x in SS))
        for i in comp:
            sig, blk = comp[i]["sig"], comp[i]["blk"]
            W = inst["gorder"][:inst["gorder"].index(blk[0])]
            K = inst["Kblk"][i]
            lpos = inst["Y"][i].index(f"l{i}")

            def G(u, K=K, lpos=lpos):
                out = defaultdict(F)
                for y, p in K(u).items():
                    out[y[lpos]] += p
                return clean(out)
            Vs = lambda e, sig=tuple(sig): tuple(e[x] for x in sig)
            Vw = lambda e, W=tuple(W): tuple(e[x] for x in W)
            Li = lambda e, i=i: e[f"l{i}"]
            # Lemma r4-block (c) under both laws.
            for mech in (base, kappa):
                jw = law_of(mech, lambda e, i=i: (Vw(e), tuple(e[y] for y in inst["Y"][i])))
                mw = law_of(mech, Vw)
                for (w, y), p in jw.items():
                    assert p == mw[w] * K(tuple(w[W.index(x)] for x in sig)).get(y, F(0))
            # (i)(b): standalone law of L_i given the interface value is G_i.
            for u, law in conditional(inst["sa"][i], Vs, Li).items():
                assert law == G(u)
            psi = func_map(kappa, S, Vs)                       # (Rd_i)
            T = func_map(kappa, Vw, S) is not None             # (T_i)
            if psi is not None:
                c["Rd"] += 1
                Fi = conditional(kappa, S, Li)
                injective = len(set(psi.values())) == len(psi)
                in_W = set(SS) <= set(W)
                if injective:
                    assert T                                   # injective psi_i gives (T_i)
                    c["inj_out_W"] += not in_W
                if T:                                          # (i)(a)
                    for s, law in Fi.items():
                        assert law == G(psi[s])
                    c["a"] += 1
                    if not in_W:
                        c["a_by_inj"] += 1
                else:
                    c["T_fail"] += 1
                    c["T_fail_break"] += any(law != G(psi[s]) for s, law in Fi.items())
                # (i)(c), joint-baseline declaration: check the two counterparts under P^0.
                psi0 = func_map(base, S, Vs)
                T0 = func_map(base, Vw, S) is not None
                if T and psi0 is not None and T0 and all(psi0[s] == psi[s] for s in set(psi0) & set(psi)):
                    for s, law in conditional(base, S, Li).items():
                        assert law == G(psi0[s])               # literal (Inv_i) holds
                    c["c"] += 1
                # Support relation under psi_i(S^sa_i) = S^(i) P^0-a.s.
                supp_sig = set(law_of(kappa, Vs)) <= set(law_of(inst["sa"][i], Vs))
                if psi0 is not None and all(psi0[s] == psi[s] for s in set(psi0) & set(psi)):
                    lit = set(law_of(kappa, S)) <= set(law_of(base, S))
                    if lit:
                        assert supp_sig                        # (Supp_i) implies (Supp^Sigma_i)
                    c["sig_not_lit"] += supp_sig and not lit
            # (i)(f): S = g(V_Sigma_i, V_C_i) in both deployments, Sigma_i not downstream.
            if not set(sig) & dsc:
                gt = {}
                g = lambda e, gt=gt: gt.setdefault(tuple(e[x] for x in sig + blk), rng.choice(QV))
                for e, _ in base.envs() + kappa.envs():
                    g(e)
                lk = law_of(kappa, lambda e: (g(e), Li(e)))
                l0 = law_of(base, lambda e: (g(e), Li(e)))
                assert lk == l0
                c["f"] += 1
            # (ii): interface support, (G1) at the kernels of C_i (full list) and at the
            # collapsed block kernel.
            supp_sig = set(law_of(kappa, Vs)) <= set(law_of(base, Vs))
            assert law_of(base, Vs) == law_of(inst["sa"][i], Vs)
            g1_all = True
            for j in blk:
                ins = list(base.inputs[j])
                reached = vs.marginal(kappa.names, kappa.law(), ins)
                seen = vs.marginal(base.names, base.law(), ins)
                g1_all &= all(z in seen for z in reached)
            if supp_sig:
                assert g1_all
                c["supp"] += 1
            else:
                c["unsupp"] += 1
                c["g1_without_supp"] += g1_all
            reached_c = set(law_of(kappa_c, Vs))
            seen_c = set(law_of(base_c, Vs))
            assert (reached_c <= seen_c) == supp_sig
    assert c["a_by_inj"] > 0 and c["inj_out_W"] > 0 and c["T_fail_break"] > 0 and c["g1_without_supp"] > 0
    assert c["sig_not_lit"] > 0 and c["f"] > 0 and c["c"] > 0
    print(f"PASS general specifications: in {n_inst} random joint specifications with one or two "
          f"interface coordinates, ternary ranges, partial reads, random interleavings, and state "
          f"kernels reading arbitrary earlier coordinates, Lemma r4-block (b) and (c) hold under "
          f"both laws; (i)(a) holds in all {c['a']} (component, state map) pairs with (Rd_i) and "
          f"(T_i), {c['a_by_inj']} of them with a state coordinate written after the block starts, "
          f"{c['inj_out_W']} of these through an injective psi_i; without (T_i) the identity breaks in {c['T_fail_break']} of "
          f"{c['T_fail']}; (i)(c) gives the literal (Inv_i) under the joint-baseline declaration "
          f"in {c['c']} pairs; (Supp_i) implies (Supp^Sigma_i) wherever psi_i(S^sa_i) = S^(i), and "
          f"(Supp^Sigma_i) holds without (Supp_i) in {c['sig_not_lit']} pairs; (i)(f) holds in "
          f"{c['f']} components; (Supp^Sigma_i) implies (G1) at every component kernel in all "
          f"{c['supp']} supported components, and (G1) holds at every component kernel without "
          f"(Supp^Sigma_i) in {c['g1_without_supp']} of {c['unsupp']} unsupported components; "
          f"(G1) at the collapsed block kernel is exactly (Supp^Sigma_i).")


# --------------------------------------------------------------------------
# Claims (ii) and (iii): (Supp) is (G1); (Known P_S) by the g-formula
# --------------------------------------------------------------------------


def supp(inst, kappa_c, k):
    reached = law_of(kappa_c, lambda e: e[f"q{k}"])
    return all(inst["comps"][k]["q_sa"].get(s, 0) > 0 for s in reached)


def check_supp_and_state_law(n_inst=300, seed=11):
    rng = random.Random(seed)
    g1_fail = ps_ok = ps_err = 0
    for t in range(n_inst):
        inst = make_instance(rng, rng.choice((2, 3)), "seq")
        n, base, kappa, D = inst["n"], inst["base"], inst["kappa"], inst["D"]
        base_c, kappa_c = collapsed_pair(inst)
        rec_c = list(base_c.names)
        # (ii): (G1) at the block kernels relevant to L_n is (Supp_k) for those k.
        S_L = [f"Y{n}"]
        cond = vs.conditions(base_c, kappa_c, D, S_L, rec_c)
        J, _ = vs.relevant_set(base_c, kappa_c, D, S_L, ())
        rel = [k for k in range(1, n + 1) if f"Y{k}" in J]
        all_supp = all(supp(inst, kappa_c, k) for k in rel)
        assert cond["G1"] == all_supp and cond["G2"] and cond["G3"] and cond["G4"]
        # In these instances every block kernel reads exactly its one interface coordinate,
        # so (G1) in the full list coincides with (Supp^Sigma) by construction.  The general
        # one-directional statement of part (ii) is tested in check_general.
        cond_full = vs.conditions(base, kappa, D, [f"L{n}"], list(base.names))
        assert cond_full["G1"] == all_supp
        g1_fail += not all_supp
        # (iii): the state law from the g-formula on the collapsed list, using only the
        # product record law of the standalone runs (internal coordinates unrecorded).
        record = [x for k in range(1, n + 1) for x in (f"q{k}", f"Y{k}")]
        rlaw = rec_law(base, record)
        S_names = [f"q{k}" for k in range(1, n + 1)]
        Sfun = lambda e: tuple(e[x] for x in S_names)
        J_S, _ = vs.relevant_set(base_c, kappa_c, D, S_names, ())
        rel_S = [k for k in range(1, n + 1) if f"Y{k}" in J_S]
        try:
            ps = vs.g_formula(record, rlaw, base_c, kappa_c, D, S_names, Sfun)
            assert ps == law_of(kappa, Sfun)
            assert all(supp(inst, kappa_c, k) for k in rel_S)
            ps_ok += 1
        except vs.PositivityError:
            assert not all(supp(inst, kappa_c, k) for k in rel_S)
            ps_err += 1
    assert g1_fail > 0 and ps_ok > 0 and ps_err > 0
    print(f"PASS (ii) support is (G1) at the collapsed kernel and (iii) state law: over {n_inst} "
          f"random sequential compositions, (G1) at the collapsed component kernels holds exactly "
          f"when every relevant component's composed interface values occur in its standalone runs "
          f"(fails in {g1_fail}), and (G2)-(G4) hold with only states and recorded outputs logged; the "
          f"g-formula on the collapsed list, fed the product of the standalone record laws, "
          f"returns the composed state law exactly in {ps_ok} instances and raises a support "
          f"error in the other {ps_err}, exactly where an upstream component is unsupported.")


# --------------------------------------------------------------------------
# Claim (iv): the interval of Proposition prop:component-joint(iv)(b) is the M1 image
# --------------------------------------------------------------------------


def check_interval(n_inst=200, seed=13):
    rng = random.Random(seed)
    done = nondeg = 0
    while done < n_inst:
        inst = make_instance(rng, 2, "seq")
        base, D = inst["base"], inst["D"]
        base_c, kappa_c = collapsed_pair(inst)
        if not supp(inst, kappa_c, 1):
            continue                      # (Known P_S) via (iii) needs component 1 supported
        done += 1
        record = ["q1", "Y1", "q2", "Y2"]
        rlaw = rec_law(base, record)
        P_S = law_of(kappa_c, lambda e: e["q2"])
        qsa = inst["comps"][2]["q_sa"]
        U = sorted(s for s in P_S if qsa.get(s, 0) == 0)
        Qcond = conditional(inst["sa"][2], lambda e: e["q2"], lambda e: e["L2"])
        c_B = sum(P_S[s] * Qcond[s].get(1, F(0)) for s in P_S if s not in U)
        width = sum(P_S[s] for s in U)
        loss = lambda e: e["Y2"][1]
        vals = []
        for choice in product(YV, repeat=len(U)):
            fam = dict(zip(U, choice))
            free = lambda j, z, fam=fam: point(fam[z[0]])
            q = gq_formula(record, rlaw, base_c, kappa_c, D, ["Y2"], loss, free)
            vals.append(q.get(1, F(0)))
            # Realize the family: change the block kernel of component 2 at U only.
            K2 = base_c.kernels["Y2"]
            newK = lambda s, fam=fam, K2=K2: point(fam[s]) if s in fam else K2(s)
            b2 = base_c.replace({"Y2": (("q2",), newK)})
            k2 = b2.replace(inst["repl"])
            assert rec_law(b2, record) == rlaw                    # Lemma r4-zero
            assert law_of(k2, loss).get(1, F(0)) == vals[-1]
        assert (min(vals), max(vals)) == (c_B, c_B + width)
        if U:
            nondeg += 1
            mix = {y: F(1, 4) for y in YV}                         # a non-point free law
            free = lambda j, z: mix
            q = gq_formula(record, rlaw, base_c, kappa_c, D, ["Y2"], loss, free)
            assert c_B <= q.get(1, F(0)) <= c_B + width
        else:
            assert len(set(vals)) == 1
    assert nondeg > 0
    print(f"PASS (iv) interval as an image: in {n_inst} random two-component compositions with "
          f"component 1 supported, the free-law g-formula over point-mass families at the "
          f"unsupported states of component 2 has range exactly [c_B, c_B + P_S(U_2)] "
          f"(nondegenerate in {nondeg}); every family is realized by a mechanism with the same "
          f"standalone record law whose composed failure probability equals the formula.")


def interval_full_mech(c1_tab, s_repl, t_repl, s_sa, t_sa, a2, l_tab, l_ins=("s2", "t2", "x2")):
    """Full list for part (iv): component 1 (q1 = 2, private X1, spend C1 in {0,1,2});
    component 2 with interface (s2, t2), private x2 ~ Bern(a2), ternary loss l2 reading
    l_ins.  kappa_conn sets s2 and t2 from C1."""
    nodes = [("q1", (), lambda: point(2)),
             ("X1", ("q1",), lambda q: bern(F(1, 2))),
             ("C1", ("X1",), lambda x: c1_tab[x]),
             ("s2", (), lambda: dict(s_sa)),
             ("t2", ("s2",), lambda s: t_sa[s]),
             ("x2", (), lambda: bern(a2)),
             ("l2", l_ins, lambda *z: l_tab[z])]
    base = Mechanism(nodes)
    kappa = base.replace({"s2": (("C1",), lambda c: s_repl[c]), "t2": (("C1",), lambda c: t_repl[c])})
    return base, kappa


def check_interval_full(n_inst=6, seed=41):
    """Part (iv) by brute force over full-list members: every member of a grid of component
    kernels with the true standalone record law lies in [c_B, c_B + P_S(U_2)], and the
    endpoints are realized by changing the component kernel l2 at null inputs only.  The
    interface has two coordinates and the loss is ternary.  A partial-read control shows
    that the interval needs the family to be free at the unseen interface values."""
    rng = random.Random(seed)
    record = ["q1", "C1", "s2", "t2", "l2"]
    grid_a = (F(0), F(1, 2), F(1))
    pts = [point(v) for v in QV]
    inputs = list(product(BIN, BIN, BIN))
    events = [B for r in (1, 2) for B in combinations(QV, r)]
    members_total = nondeg = 0
    done = 0
    while done < n_inst:
        c1_tab = {x: rand_law(rng, QV) for x in BIN}
        s_repl = {c: rand_law(rng, BIN) for c in QV}
        t_repl = {c: rand_law(rng, BIN) for c in QV}
        s_sa = rand_law(rng, BIN, zero=False)
        t_sa = {s: rand_law(rng, BIN) for s in BIN}
        a_true = rng.choice(grid_a)
        l_true = {z: rng.choice(pts) for z in inputs}
        base_t, kappa_t = interval_full_mech(c1_tab, s_repl, t_repl, s_sa, t_sa, a_true, l_true)
        P_S = law_of(kappa_t, lambda e: (e["s2"], e["t2"]))
        seen = law_of(base_t, lambda e: (e["s2"], e["t2"]))
        U = [u for u in P_S if u not in seen]
        if not U:
            continue
        done += 1
        nondeg += 1
        rl_true = rec_law(base_t, record)
        Qc = conditional(base_t, lambda e: (e["s2"], e["t2"]), lambda e: e["l2"])
        width = sum(P_S[u] for u in U)
        cB = {B: sum(P_S[u] * sum(Qc[u].get(v, F(0)) for v in B) for u in P_S if u not in U)
              for B in events}
        vals = {B: [] for B in events}
        for a in grid_a:
            for choice in product(range(3), repeat=len(inputs)):
                l_tab = {z: pts[k] for z, k in zip(inputs, choice)}
                b, k = interval_full_mech(c1_tab, s_repl, t_repl, s_sa, t_sa, a, l_tab)
                if rec_law(b, record) != rl_true:
                    continue
                members_total += 1
                assert law_of(k, lambda e: (e["s2"], e["t2"])) == P_S
                lk = law_of(k, lambda e: e["l2"])
                for B in events:
                    vals[B].append(sum(lk.get(v, F(0)) for v in B))
        for B in events:
            assert (min(vals[B]), max(vals[B])) == (cB[B], cB[B] + width)
        # Realization through the component kernel l2 at null inputs only (Lemma r4-zero).
        for B in events:
            for target in (0, 1):
                v = next(x for x in QV if (x in B) == bool(target))
                l_tab = {z: (point(v) if z[:2] in U else l_true[z]) for z in inputs}
                b, k = interval_full_mech(c1_tab, s_repl, t_repl, s_sa, t_sa, a_true, l_tab)
                assert b.law() == base_t.law()
                got = sum(law_of(k, lambda e: e["l2"]).get(x, F(0)) for x in B)
                assert got == cB[B] + target * width
    # Partial-read control: l2 reads (s2, x2) only, kappa moves t2 to an unseen value.
    s_law = {0: F(1, 2), 1: F(1, 2)}
    t0 = {s: point(0) for s in BIN}
    tk = {c: bern(F(1, 2)) for c in QV}
    c1 = {x: point(0) for x in BIN}
    s_id = {c: dict(s_law) for c in QV}
    pin = list(product(BIN, BIN))
    l_true = {z: (point(z[0] + z[1])) for z in pin}
    bt, kt = interval_full_mech(c1, s_id, tk, s_law, t0, F(1, 2), l_true, l_ins=("s2", "x2"))
    rl = rec_law(bt, record)
    P_S = law_of(kt, lambda e: (e["s2"], e["t2"]))
    U = [u for u in P_S if u not in law_of(bt, lambda e: (e["s2"], e["t2"]))]
    assert sum(P_S[u] for u in U) == F(1, 2)
    partial_vals = set()
    for a in grid_a:
        for choice in product(range(3), repeat=len(pin)):
            l_tab = {z: pts[k] for z, k in zip(pin, choice)}
            b, k = interval_full_mech(c1, s_id, tk, s_law, t0, a, l_tab, l_ins=("s2", "x2"))
            if rec_law(b, record) == rl:
                partial_vals.add(tuple(sorted(law_of(k, lambda e: e["l2"]).items())))
    assert len(partial_vals) == 1
    print(f"PASS (iv) full list: over {n_inst} instances with a two-coordinate interface and a "
          f"ternary loss, and {members_total} grid members of component kernels with the true "
          f"standalone record law, P(L_2 in B) ranges exactly over [c_B, c_B + P_S(U_2)] for "
          f"every nonempty proper B, P_S is the same for every member, and both endpoints are "
          f"realized by changing the component kernel l2 at null inputs only; when l2 does not "
          f"read the interface coordinate that leaves the standalone support, (Supp^Sigma_2) "
          f"fails with P_S(U_2) = 1/2 yet the law of L_2 is identified.")


# --------------------------------------------------------------------------
# Claim (v): conditional independence under kappa_conn
# --------------------------------------------------------------------------


def check_ci_rereading(n_inst=120, seed=5):
    """Part (v)(a) needs only (K1) and (K2).  Component k reads its state s_k at the start,
    acts (a_k), reads a second state r_k after acting, and fails (l_k).  So (Blk_k) and
    (T_k) fail.  The replaced state kernels read one block plus other state coordinates,
    and S holds every state coordinate."""
    rng = random.Random(seed)
    ok = 0
    for _ in range(n_inst):
        tabs = {k: dict(a=rand_table(rng, [BIN], BIN), l=rand_table(rng, [BIN, BIN, BIN], BIN),
                        s=rand_law(rng, BIN)) for k in (1, 2, 3)}
        nodes = {}
        for k in (1, 2, 3):
            nodes[f"s{k}"] = (f"s{k}", (), lambda k=k: dict(tabs[k]["s"]))
            nodes[f"a{k}"] = (f"a{k}", (f"s{k}",), from_table(tabs[k]["a"]))
            nodes[f"r{k}"] = (f"r{k}", (f"a{k}",), lambda a: point(a))
            nodes[f"l{k}"] = (f"l{k}", (f"s{k}", f"a{k}", f"r{k}"), from_table(tabs[k]["l"]))
        order = ["s1", "a1", "s2", "r1", "l1", "a2", "r2", "l2", "s3", "a3", "r3", "l3"]
        base = Mechanism([nodes[x] for x in order])
        repl = {"s2": (("a1", "s1"), from_table(rand_table(rng, [BIN, BIN], BIN))),   # block 1 + state
                "r1": (("a1", "s2"), from_table(rand_table(rng, [BIN, BIN], BIN))),   # own block + state
                "s3": (("a2", "r1"), from_table(rand_table(rng, [BIN, BIN], BIN))),   # block 2 + state
                "r3": (("a3", "s2"), from_table(rand_table(rng, [BIN, BIN], BIN)))}   # own block + state
        kappa = base.replace(repl)
        S = lambda e: tuple(e[x] for x in ("s1", "r1", "s2", "r2", "s3", "r3"))
        assert ci_holds(kappa, S, ["l1", "l2", "l3"])
        ok += 1
    return ok


def check_ci(n_inst=150, seed=17):
    rng = random.Random(seed)
    ok = fails_k2 = 0
    for t in range(n_inst):
        for sched in ("seq", "par"):                    # (K1) with S = all states, (K2)
            inst = make_instance(rng, 3, sched)
            S = lambda e: (e["q1"], e["q2"], e["q3"])
            assert ci_holds(inst["kappa"], S, ["L1", "L2", "L3"])
            ok += 1
        inst = make_instance(rng, 3, "seq2")            # q3 reads Y1 and Y2: (K2) fails
        S = lambda e: (e["q1"], e["q2"], e["q3"])
        fails_k2 += not ci_holds(inst["kappa"], S, ["L1", "L2", "L3"])
    assert fails_k2 > 0

    # Common effect: q3 = L1 + L2 read by component 3; (K1) holds, (K2) fails.
    ce = Mechanism([("q1", (), lambda: point(0)), ("L1", ("q1",), lambda q: bern(F(1, 2))),
                    ("q2", (), lambda: point(0)), ("L2", ("q2",), lambda q: bern(F(1, 2))),
                    ("q3", ("L1", "L2"), lambda a, b: point(a + b))])
    both = conditional(ce, lambda e: e["q3"], lambda e: (e["L1"], e["L2"]))
    assert both[1].get((1, 1), F(0)) == 0 and not ci_holds(ce, lambda e: e["q3"], ["L1", "L2"])

    # Unrecorded tape: q2 = Y1 xor T and q3 = Y2 xor T with T unlogged; (K2) holds per
    # kernel, (K1) fails because T is a replaced coordinate that S does not record.
    tape = Mechanism([("T", (), lambda: bern(F(1, 2))),
                      ("q1", (), lambda: point(0)), ("L1", ("q1",), lambda q: bern(F(1, 2))),
                      ("q2", ("L1", "T"), lambda a, t: point(a ^ t)),
                      ("L2", ("q2",), lambda q: bern(F(1, 2))),
                      ("q3", ("L2", "T"), lambda a, t: point(a ^ t)),
                      ("L3", ("q3",), lambda q: bern(F(1, 2)))])
    S = lambda e: (e["q1"], e["q2"], e["q3"])
    assert not ci_holds(tape, S, ["L1", "L2", "L3"])
    St = lambda e: (e["T"], e["q1"], e["q2"], e["q3"])
    assert ci_holds(tape, St, ["L1", "L2", "L3"])
    n_reread = check_ci_rereading()
    print(f"PASS (v)(a) conditional independence: in {2 * n_inst} random 3-component "
          f"compositions where S records every replaced coordinate and each composed state "
          f"kernel reads at most one component, the losses are mutually independent given S; "
          f"the same holds in {n_reread} compositions whose components read a second state after "
          f"acting, so (Blk_k) and (T_k) fail, with state kernels reading one block plus other "
          f"state coordinates; "
          f"with one state kernel reading two components (CI) fails in {fails_k2} of {n_inst}; "
          f"a budget spent by two components gives P(L1=L2=1 | S=1) = 0 against 1/4; an "
          f"unlogged tape shared by two state kernels breaks (CI), and logging it restores it.")


# --------------------------------------------------------------------------
# Counterexamples behind the hypotheses of parts (ii) and (iii), and the target of
# Proposition prop:component-joint(iii)
# --------------------------------------------------------------------------


def check_counterexamples():
    # (ii) without (Blk_2): component 2 reads its budget b0 at the start and again (b1)
    # after its own spend a2.  Standalone b1 = 2 - a2; composed b1 = 2 - max(a2, c1).
    def reread(g):
        base = Mechanism([("q1", (), lambda: point(0)),
                          ("c1", ("q1",), lambda q: bern(F(1, 2))),
                          ("b0", (), lambda: point(2)),
                          ("a2", ("b0",), lambda b: bern(F(1, 2))),
                          ("b1", ("a2",), lambda a: point(2 - a)),
                          ("L2", ("a2", "b1"), lambda a, b: bern(g[(a, b)]))])
        return base, base.replace({"b1": (("a2", "c1"), lambda a, c: point(2 - max(a, c)))})
    g1 = {(a, b): F(0) for a in BIN for b in QV}
    g1[(0, 2)], g1[(1, 1)] = F(1, 10), F(1, 2)
    g2 = dict(g1)
    g2[(0, 1)] = F(1)
    (b1, k1), (b2, k2) = reread(g1), reread(g2)
    rec = ["q1", "c1", "b0", "a2", "b1", "L2"]
    assert rec_law(b1, rec) == rec_law(b2, rec)
    Vs = lambda e: (e["b0"], e["b1"])
    assert set(law_of(k1, Vs)) == {(2, 1), (2, 2)} <= set(law_of(b1, Vs))        # (Supp^Sigma_2)
    assert law_of(k1, lambda e: (e["a2"], e["b1"]))[(0, 1)] == F(1, 4)
    assert (0, 1) not in law_of(b1, lambda e: (e["a2"], e["b1"]))                 # (G1) fails
    r1, r2 = law_of(k1, lambda e: e["L2"])[1], law_of(k2, lambda e: e["L2"])[1]
    assert (r1, r2) == (F(11, 40), F(21, 40))

    # (iii) needs (G4) and the records: upstream component 1 is supported in both pairs.
    # (a) The state kernel reads component 1's unlogged internal coordinate X1.
    def via_internal(px, linked):
        nodes = [("q1", (), lambda: point(2)),
                 ("X1", ("q1",), lambda q: bern(px)),
                 ("C1", ("X1",), (lambda x: point(x)) if linked else (lambda x: bern(F(1, 2)))),
                 ("q2", (), lambda: point(2))]
        base = Mechanism(nodes)
        return base, base.replace({"q2": (("X1",), lambda x: point(2 - x))})
    (ba, ka), (bb, kb) = via_internal(F(1, 2), True), via_internal(F(1, 4), False)
    rec = ["q1", "C1", "q2"]
    assert rec_law(ba, rec) == rec_law(bb, rec)
    Pa, Pb = law_of(ka, lambda e: e["q2"]), law_of(kb, lambda e: e["q2"])
    assert Pa == {1: F(1, 2), 2: F(1, 2)} and Pb == {1: F(1, 4), 2: F(3, 4)}

    # (b) The state kernel reads the logged output C1, but component 1's interface q1 is
    # replaced and unlogged in standalone runs, so (G2) fails at its block.
    def via_unlogged_state(linked):
        base = Mechanism([("q1", (), lambda: bern(F(1, 2))),
                          ("C1", ("q1",), (lambda q: point(q)) if linked else (lambda q: bern(F(1, 2)))),
                          ("q2", (), lambda: point(2))])
        return base, base.replace({"q1": ((), lambda: point(1)),
                                   "q2": (("C1",), lambda c: point(2 - c))})
    (bc, kc), (bd, kd) = via_unlogged_state(True), via_unlogged_state(False)
    rec = ["C1", "q2"]
    assert rec_law(bc, rec) == rec_law(bd, rec)
    assert set(law_of(kc, lambda e: e["q1"])) <= set(law_of(bc, lambda e: e["q1"]))   # supported
    Pc, Pd = law_of(kc, lambda e: e["q2"]), law_of(kd, lambda e: e["q2"])
    assert Pc == {1: F(1)} and Pd == {1: F(1, 2), 2: F(1, 2)}

    # Proposition prop:component-joint(iii): the "if and only if" concerns the law of (S, L_1, ..., L_n).
    # Two states; at s = 0 both component laws are fair bits, at s = 1 F_2 is a point mass;
    # h(s, l1, l2) = l1 + l2 [s = 1].  Gamma has five listed members; the law of L is the same.
    P_S = {0: F(1, 2), 1: F(1, 2)}
    p_s = [P_S[0], P_S[1]]
    fs = [[[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]], [[F(1, 2), F(1, 2)], [F(1), F(0)]]]
    cells, A, b = vc.gamma_constraints(p_s, fs, [2, 2])
    c_cell = [F(1) if (cell[0], cell[1], cell[2]) == (0, 1, 1) else F(0) for cell in cells]
    assert (vc.lp_min(c_cell, A, b), vc.lp_max(c_cell, A, b)) == (F(0), F(1, 4))   # Gamma not a point
    laws_L, lams = set(), set()
    for beta in [F(k, 8) for k in range(5)]:
        lam = {}
        for x, y in product(BIN, BIN):
            n00 = {(1, 1): beta, (1, 0): F(1, 2) - beta, (0, 1): F(1, 2) - beta, (0, 0): beta}[(x, y)]
            lam[(0, x, y)] = P_S[0] * n00
            lam[(1, x, y)] = P_S[1] * fs[0][1][x] * fs[1][1][y]
        assert all(sum(row[k] * lam.get(cell, F(0)) for k, cell in enumerate(cells)) == rhs
                   for row, rhs in zip(A, b))                                        # lam in Gamma
        lams.add(tuple(sorted(clean(lam).items())))
        Lh = defaultdict(F)
        for (s, x, y), p in lam.items():
            Lh[x + y * (s == 1)] += p
        laws_L.add(tuple(sorted(clean(Lh).items())))
    assert len(lams) == 5 and len(laws_L) == 1
    print("PASS counterexamples: a component that reads its budget again after its own spend has "
          "(Supp^Sigma_2) but not (G1) (composed input (0,1) of mass 1/4 never seen alone), and two "
          "loss kernels with identical standalone records give composed risks 11/40 and 21/40; "
          "with an upstream component supported, P_S differs between mechanisms with identical "
          "standalone records when the state kernel reads an unlogged internal coordinate (1/2 "
          "against 1/4) or when the upstream interface is unlogged (point mass against 1/2); under "
          "paired records Gamma has five distinct listed members while the law of L = h(S, L1, L2) "
          "is the same for all, so the if-and-only-if of Prop prop:component-joint(iii) concerns the law of "
          "(S, L_1, ..., L_n).")


# --------------------------------------------------------------------------
# Reproduced examples
# --------------------------------------------------------------------------


def budget_spec(g2, with_l1=True, start2=None):
    """Shared budget of two paid calls.  Component 1: private X1, spend C1, loss L1.
    Component 2: state q2 (remaining budget), loss L2 ~ Bern(g2[q2])."""
    start2 = start2 or {2: F(1)}
    c1 = [("q1", (), lambda: point(2)),
          ("X1", ("q1",), lambda q: bern(F(1, 2))),
          ("C1", ("q1", "X1"), lambda q, x: bern(F(1, 4) + F(x, 2))),
          ("L1", ("X1",), lambda x: bern(F(1, 5) + F(2 * x, 5))),
          ("Y1", ("C1", "L1"), lambda c, l: {(c, l): F(1)})]
    if not with_l1:                        # Section app:p3-budget: component 1 reports its spend only
        c1 = c1[:3] + [("Y1", ("C1",), lambda c: {(c, None): F(1)})]
    c2 = [("q2", (), lambda: dict(start2)),
          ("L2", ("q2",), lambda s: bern(g2[s]))]
    base = Mechanism(c1 + c2)
    kappa = base.replace({"q2": (("Y1",), lambda y: point(2 - y[0]))})
    return base, kappa, Mechanism(c1), Mechanism(c2)


def check_budget():
    g_a = {2: F(1, 10), 1: F(1, 10), 0: F(1, 10)}     # budget-insensitive
    g_b = {2: F(1, 10), 1: F(1), 0: F(1)}             # needs two calls
    D = {"q1", "q2"}
    record = ["q1", "Y1", "q2", "L2"]
    res = {}
    for tag, g in (("a", g_a), ("b", g_b)):
        base, kappa, sa1, sa2 = budget_spec(g)
        res[tag] = dict(base=base, kappa=kappa, rlaw=rec_law(base, record),
                        risk=law_of(kappa, lambda e: e["L2"]).get(1, F(0)))
    assert res["a"]["rlaw"] == res["b"]["rlaw"]                 # same standalone evidence
    assert (res["a"]["risk"], res["b"]["risk"]) == (F(1, 10), F(11, 20))
    base, kappa = res["b"]["base"], res["b"]["kappa"]
    rlaw = res["b"]["rlaw"]
    # Collapsed list: component 1 is one kernel from q1 to Y1; component 2 is L2 itself.
    base_c = collapse(base, [(["X1", "C1", "L1", "Y1"], ["q1"], "Y1")])
    kappa_c = base_c.replace({"q2": (("Y1",), lambda y: point(2 - y[0]))})
    # (iii) P_S from the g-formula with component 1 supported (q1 = 2 in both).
    P_S = vs.g_formula(record, rlaw, base_c, kappa_c, D, ["q2"], lambda e: e["q2"])
    assert P_S == {1: F(1, 2), 2: F(1, 2)} == law_of(kappa, lambda e: e["q2"])
    # (ii) (G1) fails at component 2 exactly at the unseen state S = 1; Proposition prop:supported
    # does not apply.
    assert not vs.conditions(base_c, kappa_c, D, ["L2"], record)["G1"]
    try:
        vs.g_formula(record, rlaw, base_c, kappa_c, D, ["L2"], lambda e: e["L2"])
        raise AssertionError("support error expected")
    except vs.PositivityError:
        pass
    # (iv) M1 image over the free law of L2 at q2 = 1.
    vals = [gq_formula(record, rlaw, base_c, kappa_c, D, ["L2"], lambda e: e["L2"],
                       lambda j, z, y=y: point(y)).get(1, F(0)) for y in BIN]
    assert (min(vals), max(vals)) == (F(1, 20), F(11, 20))
    naive = law_of(Mechanism([("q2", (), lambda: point(2)), ("L2", ("q2",), lambda s: bern(g_a[s]))]),
                   lambda e: e["L2"]).get(1, F(0))
    transferred = gq_formula(record, rlaw, base_c, kappa_c, D, ["L2"], lambda e: e["L2"],
                             lambda j, z: bern(F(1, 10))).get(1, F(0))
    assert naive == transferred == F(1, 10)
    # Theorem thm:replay(iii): q2 is a deterministic replaced decision, f0 = 2, fk(c) = 2 - c.
    agree = lambda e: e["Y1"][0] == 0
    nu1 = law_of(base, lambda e: e["L2"], agree).get(1, F(0))  # P0(L2=1, agreement)
    eps = sum(p for e, p in base.envs() if not agree(e))
    assert (nu1, nu1 + eps) == (F(1, 20), F(11, 20))
    for tag in ("a", "b"):                                     # Theorem thm:replay(i)
        k, b = res[tag]["kappa"], res[tag]["base"]
        assert law_of(k, lambda e: e["L2"], agree) == law_of(b, lambda e: e["L2"], agree)
    # Section app:p3-budget: component 1 reports only its spend, so exactly one continuing
    # history (q1, Y1) reaches the null state q2 = 1, the pivot of M1 part (f); the
    # free-law image and the Theorem thm:replay(iii) set of laws of L2 coincide.
    b0, k0, _, _ = budget_spec(g_b, with_l1=False)
    rec0 = ["q1", "Y1", "q2", "L2"]
    rl0 = rec_law(b0, rec0)
    hist = {(e["q1"], e["Y1"]) for e, p in b0.envs() if not agree(e)}
    assert len(hist) == 1
    bc0 = collapse(b0, [(["X1", "C1", "Y1"], ["q1"], "Y1")])
    kc0 = bc0.replace({"q2": (("Y1",), lambda y: point(2 - y[0]))})
    img0 = [gq_formula(rec0, rl0, bc0, kc0, D, ["L2"], lambda e: e["L2"],
                       lambda j, z, y=y: point(y)).get(1, F(0)) for y in BIN]
    nu0 = law_of(b0, lambda e: e["L2"], agree).get(1, F(0))
    eps0 = sum(p for e, p in b0.envs() if not agree(e))
    assert (min(img0), max(img0)) == (nu0, nu0 + eps0) == (F(1, 20), F(11, 20))
    # Positive control: standalone runs that also start at budget 1 satisfy (Supp).
    for tag, g in (("a", g_a), ("b", g_b)):
        b2, k2, _, _ = budget_spec(g, start2={1: F(1, 2), 2: F(1, 2)})
        rl2 = rec_law(b2, record)
        bc = collapse(b2, [(["X1", "C1", "L1", "Y1"], ["q1"], "Y1")])
        kc = bc.replace({"q2": (("Y1",), lambda y: point(2 - y[0]))})
        assert vs.g_formula(record, rl2, bc, kc, D, ["L2"], lambda e: e["L2"]).get(1, F(0)) == res[tag]["risk"]
    # (v) The scheduler makes S a function of component 1's spend.  (CI) holds because the
    # only state kernel that reads a component reads one; (Inv_1) relative to S fails.
    S = lambda e: e["q2"]
    assert ci_holds(kappa, S, ["L1", "L2"])
    F1 = conditional(kappa, S, lambda e: e["L1"])
    _, _, sa1, _ = budget_spec(g_b)
    Q1 = law_of(sa1, lambda e: e["L1"])                        # standalone 1 sees state 2 only
    assert F1[1][1] == F(1, 2) and F1[2][1] == F(3, 10) and Q1[1] == F(2, 5)
    # F_1 is identified from component 1's recorded (spend, loss) by the g-formula.
    for s in (1, 2):
        num = gq_formula(record, rlaw, base_c, kappa_c, D, ["q2", "Y1"],
                         lambda e: (e["q2"], e["Y1"][1]), lambda j, z: None)
        assert num[(s, 1)] / (num[(s, 0)] + num[(s, 1)]) == F1[s][1]
    # The joint composed law at the supported state factorizes as Prop prop:component-joint(i) says.
    joint = conditional(kappa, S, lambda e: (e["L1"], e["L2"]))
    Q2 = conditional(Mechanism([("q2", (), lambda: point(2)), ("L2", ("q2",), lambda s: bern(g_b[s]))]),
                     lambda e: e["q2"], lambda e: e["L2"])
    assert joint[2][(1, 1)] == F1[2][1] * Q2[2][1]
    # (Inv_1) depends on the declared standalone state.  psi_1 is constant (q1 = 2), so
    # (T_1) fails and the identity of part (i)(a) fails (G_1(1|2) = 2/5).
    assert func_map(kappa, S, lambda e: e["q1"]) == {1: 2, 2: 2}
    G1_2 = law_of(sa1, lambda e: e["L1"])
    assert G1_2[1] == F(2, 5) and all(F1[s] != G1_2 for s in F1)
    # Joint-baseline declaration: S = q2 evaluated under P^0 is the constant 2, so
    # (Supp_1) fails at S = 1 and (Inv_1) fails at S = 2 (3/10 against 2/5).
    Q1_joint = conditional(base, S, lambda e: e["L1"])
    assert set(Q1_joint) == {2} and Q1_joint[2][1] == F(2, 5) and F1[2][1] == F(3, 10)
    # Scheduler-map declaration S^sa_1 = 2 - C1 on component 1's standalone trajectory: part
    # (i)(f) gives Law_kappa(S, L1) = Q^(1), hence (Inv_1) and (Supp_1), and the standalone
    # conditional law returns F_1 (Prop prop:component-joint(iv)(a)).
    Q1_sched = law_of(sa1, lambda e: (2 - e["C1"], e["L1"]))
    assert law_of(kappa, lambda e: (e["q2"], e["L1"])) == Q1_sched
    assert conditional(sa1, lambda e: 2 - e["C1"], lambda e: e["L1"]) == F1
    # Remark r4-conn-replay: standalone budget 2 - T for a recorded fair coin T.  The replay
    # bound for the component that needs two calls is [11/40, 31/40]; (G1) holds, and the
    # g-formula identifies 11/20.
    def tape_spec(g2):
        nodes = [("q1", (), lambda: point(2)),
                 ("X1", ("q1",), lambda q: bern(F(1, 2))),
                 ("C1", ("q1", "X1"), lambda q, x: bern(F(1, 4) + F(x, 2))),
                 ("L1", ("X1",), lambda x: bern(F(1, 5) + F(2 * x, 5))),
                 ("Y1", ("C1", "L1"), lambda c, l: {(c, l): F(1)}),
                 ("T", (), lambda: bern(F(1, 2))),
                 ("q2", ("T",), lambda t: point(2 - t)),
                 ("L2", ("q2",), lambda s: bern(g2[s]))]
        b = Mechanism(nodes)
        return b, b.replace({"q2": (("Y1",), lambda y: point(2 - y[0]))})
    bt, kt = tape_spec(g_b)
    agree_t = lambda e: e["Y1"][0] == e["T"]
    nu_t = law_of(bt, lambda e: e["L2"], agree_t).get(1, F(0))
    eps_t = sum(p for e, p in bt.envs() if not agree_t(e))
    assert (nu_t, nu_t + eps_t) == (F(11, 40), F(31, 40))
    assert law_of(kt, lambda e: e["L2"], agree_t) == law_of(bt, lambda e: e["L2"], agree_t)
    rec_t = ["q1", "Y1", "T", "q2", "L2"]
    assert vs.conditions(bt, kt, {"q2"}, ["L2"], rec_t)["G1"]
    ident = vs.g_formula(rec_t, rec_law(bt, rec_t), bt, kt, {"q2"}, ["L2"], lambda e: e["L2"])
    assert ident.get(1, F(0)) == F(11, 20) == law_of(kt, lambda e: e["L2"])[1]
    print("PASS budget: two components with identical standalone evidence give composed "
          "failure 1/10 and 11/20; the g-formula on the collapsed list returns P_S = (1/2, 1/2) "
          "from component 1's records; (G1) fails at S=1 and the free-law image is [1/20, 11/20], "
          "equal to the Theorem thm:replay(iii) set (agreement 1/20, continuing mass 1/2, one continuing "
          "history when component 1 reports only its spend); the "
          "transferred standalone rate 1/10 is the point with the S=2 law at S=1; standalone runs "
          "that also start at budget 1 identify both values; (CI) given S holds although S is "
          "written from component 1's spend; F_1 = (1/2, 3/10) differs from G_1 = 2/5, so "
          "(Inv_1) fails under the joint-baseline declaration and holds, with (Supp_1), under "
          "the scheduler-map declaration S = 2 - C1 (part (i)(f)); the g-formula recovers F_1 "
          "under either; with a standalone budget 2 - T for a recorded coin T the replay bound "
          "is [11/40, 31/40] while the identified value is 11/20.")


def check_stale_cache():
    def spec(g_stale, log_z):
        c1 = [("q1", (), lambda: point(2)),
              ("Z1", ("q1",), lambda q: bern(F(1, 2))),         # component 1's cache write
              ("Y1", ("Z1",), lambda z: point(z))]
        c2 = [("b2", (), lambda: point(2)), ("z2", (), lambda: point(0)),
              ("L2", ("b2", "z2"), lambda b, z: bern(F(1, 10) if z == 0 else g_stale))]
        base = Mechanism(c1 + c2)
        kappa = base.replace({"z2": (("Y1",), lambda y: point(y))})
        record = ["q1", "Y1", "b2", "L2"] + (["z2"] if log_z else [])
        return base, kappa, record
    risks, rlaws = {}, {}
    for g in (F(1, 10), F(1)):
        base, kappa, record = spec(g, log_z=False)
        rlaws[g] = rec_law(base, record)
        risks[g] = conditional(kappa, lambda e: e["b2"], lambda e: e["L2"])[2].get(1, F(0))
        sa = conditional(base, lambda e: e["b2"], lambda e: e["L2"])[2].get(1, F(0))
        assert sa == F(1, 10)
        cond = vs.conditions(base, kappa, {"z2"}, ["L2"], record)
        assert not cond["G2"] and cond["G4"]                    # (G2) fails: z2 unrecorded
    assert rlaws[F(1, 10)] == rlaws[F(1)]                      # same evidence
    assert (risks[F(1, 10)], risks[F(1)]) == (F(1, 10), F(11, 20))
    # Logging the flag: S' = (budget, Z); (Inv) holds, (Supp) fails at (2, stale).
    base, kappa, record = spec(F(1), log_z=True)
    D = {"z2"}
    rlaw = rec_law(base, record)
    cond = vs.conditions(base, kappa, D, ["L2"], record)
    assert not cond["G1"] and cond["G2"] and cond["G3"] and cond["G4"]
    P_S = law_of(kappa, lambda e: (e["b2"], e["z2"]))
    assert P_S == {(2, 0): F(1, 2), (2, 1): F(1, 2)}
    vals = [gq_formula(record, rlaw, base, kappa, D, ["L2"], lambda e: e["L2"],
                       lambda j, z, y=y: point(y)).get(1, F(0)) for y in BIN]
    assert (min(vals), max(vals)) == (F(1, 20), F(11, 20))
    # A standalone cache declared always fresh is a recorded coordinate: adding the constant
    # z2 = 0 to the record adds no information, and (G2) holds, so the logged case applies.
    base, kappa, rec_plain = spec(F(1), log_z=False)
    rl_plain, rl_decl = rec_law(base, rec_plain), rec_law(base, rec_plain + ["z2"])
    assert {k[:-1]: v for k, v in rl_decl.items()} == rl_plain and {k[-1] for k in rl_decl} == {0}
    assert vs.conditions(base, kappa, {"z2"}, ["L2"], rec_plain + ["z2"])["G2"]
    # Standalone cache neither logged nor declared: its stale probability pi is free.  The
    # standalone record fixes (1 - pi) g0 + pi g1 = 1/10; the composed rate (g0 + g1)/2 fills
    # exactly [1/20, 11/20] over a grid, and pi = 0 attains every grid point of it.
    grid = [F(k, 20) for k in range(21)]
    rec_u = ["q1", "Y1", "b2", "L2"]
    vals_free, vals_pi0 = set(), set()
    for pi in grid:
        for g1 in grid:
            if pi == 1:
                cands = grid if g1 == F(1, 10) else []
            else:
                g0 = (F(1, 10) - pi * g1) / (1 - pi)
                cands = [g0] if 0 <= g0 <= 1 else []
            for g0 in cands:
                b = Mechanism([("q1", (), lambda: point(2)),
                               ("Z1", ("q1",), lambda q: bern(F(1, 2))),
                               ("Y1", ("Z1",), lambda z: point(z)),
                               ("b2", (), lambda: point(2)),
                               ("z2", (), lambda pi=pi: bern(pi)),
                               ("L2", ("b2", "z2"),
                                lambda bb, z, g0=g0, g1=g1: bern(g0 if z == 0 else g1))])
                k = b.replace({"z2": (("Y1",), lambda y: point(y))})
                assert rec_law(b, rec_u) == rlaws[F(1, 10)]
                v = law_of(k, lambda e: e["L2"]).get(1, F(0))
                assert F(1, 20) <= v <= F(11, 20)
                vals_free.add(v)
                if pi == 0:
                    vals_pi0.add(v)
    assert min(vals_free) == F(1, 20) and max(vals_free) == F(11, 20)
    assert vals_pi0 == {F(1, 20) + g / 2 for g in grid}
    # Component 1's write also unlogged: its stale probability p is free.  With the standalone
    # cache declared fresh (g0 = 1/10), the composed rate (1 - p)/10 + p g1 fills [0, 1].
    rec_w = ["q1", "b2", "z2", "L2"]
    ref, vals_w = None, set()
    for p in grid:
        for g1 in grid:
            b = Mechanism([("q1", (), lambda: point(2)),
                           ("Z1", ("q1",), lambda q, p=p: bern(p)),
                           ("b2", (), lambda: point(2)),
                           ("z2", (), lambda: point(0)),
                           ("L2", ("b2", "z2"), lambda bb, z, g1=g1: bern(F(1, 10) if z == 0 else g1))])
            k = b.replace({"z2": (("Z1",), lambda y: point(y))})
            ref = ref or rec_law(b, rec_w)
            assert rec_law(b, rec_w) == ref
            vals_w.add(law_of(k, lambda e: e["L2"]).get(1, F(0)))
    assert min(vals_w) == 0 and max(vals_w) == 1
    print("PASS stale cache: with the staleness flag unlogged and the standalone cache neither "
          "logged nor declared, (G2) fails, and two components with identical standalone and "
          "component-1 records give composed failure 1/10 and 11/20 at the logged state S=2 "
          "(standalone rate 1/10); over a grid of the free standalone stale probability the "
          "composed rate fills exactly [1/20, 11/20]; a standalone cache declared fresh is a "
          "recorded coordinate and (G2) holds; logging the flag restores (G2), (G1) then fails at "
          "(2, stale), and the identified set is [1/20, 11/20]; if component 1's write is also "
          "unlogged, the composed rate fills [0, 1].")


P_STATUS = {0: F(3, 4), 1: F(1, 4)}
OUTAGE = {0: F(0), 1: F(1, 2)}
FAULT = {0: F(1, 10), 1: F(1, 5)}


def outage_spec(share_w):
    """Two booking paths; component i reads provider status f_i and outage W_i from outside,
    with private fault X_i.  kappa_conn routes path 2 through path 1's provider: f2 := f1,
    and with share_w also W2 := W1 (the shared outage)."""
    nodes = []
    for i in (1, 2):
        nodes += [(f"f{i}", (), lambda: dict(P_STATUS)),
                  (f"W{i}", (f"f{i}",), lambda s: bern(OUTAGE[s])),
                  (f"X{i}", (f"f{i}",), lambda s: bern(FAULT[s])),
                  (f"L{i}", (f"W{i}", f"X{i}"), lambda w, x: point(max(w, x)))]
    base = Mechanism(nodes)
    repl = {"f2": (("f1",), lambda s: point(s))}
    if share_w:
        repl["W2"] = (("W1",), lambda w: point(w))
    return base, base.replace(repl)


def gamma_member(eta_of_s, q_s=None):
    """Member of the shared-input family: W_i in L_1 x L_2 with law eta(.|f_i), L_i = W_i[i];
    kappa_conn sets f2 := f1 and W2 := W1."""
    q_s = q_s or P_STATUS
    nodes = []
    for i in (1, 2):
        nodes += [(f"f{i}", (), lambda: dict(q_s)),
                  (f"W{i}", (f"f{i}",), lambda s: eta_of_s(s)),
                  (f"L{i}", (f"W{i}",), lambda w, i=i: point(w[i - 1]))]
    base = Mechanism(nodes)
    kappa = base.replace({"f2": (("f1",), lambda s: point(s)), "W2": (("W1",), lambda w: point(w))})
    return base, kappa


def check_outage():
    record = ["f1", "L1", "f2", "L2"]
    both = lambda e: int(e["L1"] == 1 and e["L2"] == 1)
    S = lambda e: e["f1"]
    rl, val = {}, {}
    for share in (False, True):
        base, kappa = outage_spec(share)
        rl[share] = rec_law(base, record)
        val[share] = law_of(kappa, both).get(1, F(0))
        assert law_of(kappa, lambda e: e["f2"] == e["f1"]) == {True: F(1)}
        if not share:                       # W2 is a private input: (K1), (K2), (CI) hold
            assert ci_holds(kappa, S, ["L1", "L2"])
        else:                               # W2 is a replaced input that S does not hold
            assert not ci_holds(kappa, S, ["L1", "L2"])
            assert ci_holds(kappa, lambda e: (e["f1"], e["W1"]), ["L1", "L2"])
    assert rl[False] == rl[True]
    assert (val[False], val[True]) == (F(39, 400), F(11, 80))
    # Standalone evidence: P_S and F_i (Inv and Supp hold for both paths).
    base, kappa = outage_spec(True)
    Fi = {i: conditional(kappa, S, lambda e, i=i: e[f"L{i}"]) for i in (1, 2)}
    Qi = {i: conditional(base, lambda e, i=i: e[f"f{i}"], lambda e, i=i: e[f"L{i}"]) for i in (1, 2)}
    assert Fi == Qi and Fi[1][1][1] == F(3, 5) and Fi[1][0][1] == F(1, 10)
    # Gamma by exact LP (verify_coupling) and its closed form.
    p_s = [P_STATUS[0], P_STATUS[1]]
    fs = [[[Qi[i][s].get(l, F(0)) for l in BIN] for s in (0, 1)] for i in (1, 2)]
    cells, A, b = vc.gamma_constraints(p_s, fs, [2, 2])
    c = [F(1) if (cell[1], cell[2]) == (1, 1) else F(0) for cell in cells]
    lo, hi = vc.lp_min(c, A, b), vc.lp_max(c, A, b)
    assert (lo, hi) == vc.frechet_closed_form(p_s, fs, [{1}, {1}])[:2] == (F(1, 20), F(9, 40))
    # (v)(b): every Lambda in Gamma is the composed law of a member with the same
    # standalone records.  Endpoints, the product, the outage law, and the midpoint.
    pi = {s: (Qi[1][s][1], Qi[2][s][1]) for s in (0, 1)}

    def eta(beta):
        def e(s):
            a, b2 = pi[s]
            t = beta[s]
            return clean({(1, 1): t, (1, 0): a - t, (0, 1): b2 - t, (0, 0): 1 - a - b2 + t})
        return e
    lo_b = {s: max(F(0), pi[s][0] + pi[s][1] - 1) for s in (0, 1)}
    hi_b = {s: min(pi[s]) for s in (0, 1)}
    prod_b = {s: pi[s][0] * pi[s][1] for s in (0, 1)}
    out_b = {0: F(1, 100), 1: F(13, 25)}
    mid_b = {s: (lo_b[s] + hi_b[s]) / 2 for s in (0, 1)}
    got = []
    for beta in (lo_b, prod_b, out_b, mid_b, hi_b):
        mb, mk = gamma_member(eta(beta))
        assert rec_law(mb, record) == rl[True]                 # same standalone evidence
        got.append(law_of(mk, both).get(1, F(0)))
    assert got == [F(1, 20), F(39, 400), F(11, 80), (F(1, 20) + F(9, 40)) / 2, F(9, 40)]
    # Random members of Gamma (random per-state cell beta in the Frechet box) are realized.
    rng = random.Random(23)
    for _ in range(40):
        beta = {s: lo_b[s] + (hi_b[s] - lo_b[s]) * F(rng.randint(0, 12), 12) for s in (0, 1)}
        mb, mk = gamma_member(eta(beta))
        assert rec_law(mb, record) == rl[True]
        lam = law_of(mk, lambda e: (e["f1"], e["L1"], e["L2"]))
        for (s, l1, l2), p in lam.items():
            assert p == P_STATUS[s] * eta(beta)(s)[(l1, l2)]
    print("PASS outage: the joint baseline is two standalone paths with their own providers; "
          "kappa_conn routing path 2 through path 1's status gives both-fail 39/400 with (CI) "
          "given the status, and also sharing the unlogged outage gives 11/80 with (CI) failing; "
          "both have the same standalone records, with F_i(1|0)=1/10 and F_i(1|1)=3/5; the exact "
          "LP over Gamma gives [1/20, 9/40]; members of the shared-input family realize 1/20, "
          "39/400, 11/80, the midpoint, 9/40, and 40 random couplings with the same standalone "
          "records.")


if __name__ == "__main__":
    check_product_and_collapse()
    check_inv()
    check_general()
    check_supp_and_state_law()
    check_interval()
    check_interval_full()
    check_ci()
    check_counterexamples()
    check_budget()
    check_stale_cache()
    check_outage()

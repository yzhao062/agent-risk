#!/usr/bin/env python
"""Exact finite checks for Appendix F.10 (app:design; Snapshot Design for a Family of Gates).

Standard library only, exact rational arithmetic (fractions.Fraction); no agent data. Check A sweeps exact
integers, because every score and threshold there is a multiple of 1/160. Run:
python -B -I verify_snapshot_design.py (under ten seconds).

Check A  The threshold-family characterization (Proposition F.10.1 (prop:snapshot-design(a), (b))), exhaustively over
         score sequences: first deviations of the retry gate kappa_theta, the union over [theta_min,1), ties,
         score 1 counted from the boundary s, finite grids, intervals, gates that block at m >= theta, and the
         condition t(h) < T: a refusal at the last step is an abort. Negative controls show that each rule
         the proposition rejects disagrees with the definition.
Check B  The evidence model, with explicit mechanisms built from named kernels and path laws computed by
         forward composition (the engine of verify_replay.py): the identified set under restored runs at
         first-deviation prefixes (narrow evidence), the least set, the width formula, gate-specific openness,
         and the broader evidence of Remark F.10.2 (rem:design-evidence; restored runs from any stored ancestor),
         with the two-prefix counterexample and the step-0 cover as named witnesses.
Check C  Supplemental running-maximum (record value) identities (supplement/appendix-F2.md): the classical count, its upper bound for running maxima
         above a floor, the tie formula, and its failure for a law with a continuous part.
"""
import random
from fractions import Fraction as F
from itertools import combinations, permutations, product

NEG = None  # running maximum over an empty set (minus infinity)


# --------------------------------------------------------------------------------------------
# Check A: the characterization
# --------------------------------------------------------------------------------------------

def first_dev(scores, s, theta, ge=False):
    """First decision index t >= s at which the gate refuses: m > theta (or m >= theta when ge)."""
    for t, m in enumerate(scores):
        if t >= s and (m >= theta if ge else m > theta):
            return t
    return None


def run_max(scores, s, t):
    """Maximum of the scores at decisions s <= t' < t; None means minus infinity."""
    v = scores[s:t] if t > s else []
    return max(v) if v else NEG


ONE = 160   # Check A works in exact integers: every score, threshold and midpoint is a multiple of 1/160


def reps_of(scores):
    """One threshold per equivalence class of [0,1) for the tests m > theta and m >= theta: every score
    below 1, and the midpoint of every gap between consecutive scores (0 and 1 included as ends)."""
    vals = sorted(set(scores) | {0, ONE})
    pts = {v for v in vals if v < ONE}
    pts |= {(a + b) // 2 for a, b in zip(vals, vals[1:])}
    return sorted(pts)


def check_A():
    rng = random.Random(1)
    q = lambda *xs: [int(F(x) * ONE) for x in xs]
    grids = [
        q(0, F(1, 10), F(1, 2), F(3, 4), F(9, 10), 1),
        q(F(1, 4), F(1, 2), 1),
        q(0, 1),
        q(F(3, 10), F(9, 10), 1),
        q(F(4, 5), F(17, 20), F(19, 20), 1),          # the levels of E1's monitor above 0.80
    ]
    c = dict(seq=0, pairs=0, gate=0, union=0, fin=0, itv=0, ge=0, final_rhs=0, final_cont=0)
    mut = dict(tie=0, steps=0, all_earlier=0, no_horizon=0, one_all=0)
    for g in grids:
        Tmax = 5 if len(g) >= 5 else 6
        for T in range(1, Tmax + 1):
            last = T - 1                                    # a refusal at the last step is an abort
            for scores in product(g, repeat=T):
                ths = reps_of(scores)
                c["seq"] += 1
                for s in range(0, min(T - 1, 2) + 1):
                    c["pairs"] += 1
                    fd = {th: first_dev(scores, s, th) for th in ths}
                    bars = [run_max(scores, s, t) for t in range(T)]
                    bars_all = [run_max(scores, 0, t) for t in range(T)]      # ignores the boundary
                    # (a) per gate: t is the first deviation iff t >= s and mbar <= theta < m
                    for t in range(s, T):
                        bar, m = bars[t], scores[t]
                        for th in ths:
                            claim = (bar is NEG or bar <= th) and th < m
                            assert (fd[th] == t) == claim, (scores, s, t, th)
                            c["gate"] += 1
                            if claim and t == last:
                                c["final_rhs"] += 1          # the right side of (a) holds at the last step ...
                            if claim and t < last:           # ... where the refusal is an abort, so h is not in D+
                                c["final_cont"] += 1
                    for t in range(0, s):
                        assert all(fd[th] != t for th in ths)
                    # (b) union over Theta = [floor,1): running maxima above the floor, and their gate sets
                    for fl in ths:
                        need = {t for t in range(s, T)
                                if scores[t] > fl and (bars[t] is NEG or scores[t] > bars[t])}
                        union = {fd[th] for th in ths if th >= fl and fd[th] is not None}
                        assert union == need, (scores, s, fl)
                        assert {t for t in union if t < last} == {t for t in need if t < last}
                        for t in need:
                            lo = fl if bars[t] is NEG else max(fl, bars[t])
                            got = {th for th in ths if th >= fl and fd[th] == t}
                            assert got == {th for th in ths if lo <= th < scores[t]}
                        assert len(need) <= len(set(x for x in scores if x > fl))
                        c["union"] += 1
                        # negative controls: each rejected rule differs from the definition somewhere
                        tie = {t for t in range(s, T) if scores[t] > fl and (bars[t] is NEG or scores[t] >= bars[t])}
                        steps = {t for t in range(s, T) if scores[t] > fl}
                        allr = {t for t in range(s, T)
                                if scores[t] > fl and (bars_all[t] is NEG or scores[t] > bars_all[t])}
                        mut["tie"] += tie != need
                        mut["steps"] += steps != need
                        mut["all_earlier"] += allr != need
                        mut["no_horizon"] += need != {t for t in need if t < last}
                        for t in range(s, T):
                            if scores[t] == ONE:              # a score of 1: running maximum iff all earlier scores since s are below 1
                                right = all(scores[u] < ONE for u in range(s, t))
                                assert (t in need) == right
                                wrong = all(scores[u] < ONE for u in range(0, t))
                                mut["one_all"] += (t in need) != wrong
                    # finite families Theta = {theta_1<...<theta_n}: needed iff beta(m) > beta(bar); at most n
                    for _ in range(2):
                        n = rng.randint(1, min(5, len(ths)))
                        th_set = sorted(rng.sample(ths, n))
                        beta = lambda x: sum(1 for th in th_set if x is not NEG and th < x)
                        need = {t for t in range(s, T) if beta(scores[t]) > beta(bars[t])}
                        union = {first_dev(scores, s, th) for th in th_set} - {None}
                        assert union == need and len(need) <= n, (scores, s, th_set)
                        c["fin"] += 1
                    # intervals Theta = [a,b], b < 1
                    for _ in range(3):
                        a = rng.choice(ths)
                        b = rng.choice([x for x in ths if x >= a])
                        need = {t for t in range(s, T)
                                if (lambda lo: lo < scores[t] and lo <= b)(a if bars[t] is NEG else max(a, bars[t]))}
                        union = {fd[th] for th in ths if a <= th <= b and fd[th] is not None}
                        assert union == need, (scores, s, a, b)
                        c["itv"] += 1
                    # gates that refuse at m >= theta, Theta = [floor,1): needed iff m >= floor and m > bar
                    fdge = {th: first_dev(scores, s, th, ge=True) for th in ths}
                    for fl in ths:
                        need = {t for t in range(s, T) if scores[t] >= fl and (bars[t] is NEG or scores[t] > bars[t])}
                        union = {fdge[th] for th in ths if th >= fl and fdge[th] is not None}
                        assert union == need, (scores, s, fl, "ge")
                        c["ge"] += 1
    assert all(v > 0 for v in mut.values()) and c["final_rhs"] > 1000, (mut, c)
    print(f"PASS A: {c['seq']} score sequences, {c['pairs']} (sequence, boundary) pairs; {c['gate']} per-gate checks of (a), "
          f"{c['union']} union checks, {c['fin']} finite grids, {c['itv']} intervals, {c['ge']} gates refusing at m >= theta")
    print(f"PASS A1: the right side of (a) holds at the last step in {c['final_rhs']} gate checks, where the refusal is an abort "
          f"and h is not in D+; at steps before the last it holds in {c['final_cont']}, all continuing")
    print(f"PASS A2: negative controls, disagreements with the definition over the {c['union']} union checks: tie-inclusive rule {mut['tie']}, "
          f"every step above the floor {mut['steps']}, running maximum ignoring the boundary {mut['all_earlier']}, "
          f"dropping t < T {mut['no_horizon']}; a score of 1 read against all earlier scores instead of those since s {mut['one_all']}")


def check_A_witnesses():
    lattice = [F(k, 80) for k in range(80)]
    # (i) scores 0.85, 0.95, 0.95, 1 with floor 0.80: running maxima at decisions 0, 1, 3 only
    scores, s, fl = (F(17, 20), F(19, 20), F(19, 20), F(1)), 0, F(4, 5)
    union = {first_dev(scores, s, th) for th in lattice if th >= fl} - {None}
    records = {t for t in range(4) if scores[t] > fl and all(scores[t] > scores[u] for u in range(t))}
    assert union == records == {0, 1, 3}
    # (ii) a score of 1 counts from the boundary: with s = 1 the scores (1, 1, 0.1) make decision 1 a running maximum
    scores, s = (F(1), F(1), F(1, 10)), 1
    gates = [th for th in lattice if th >= F(4, 5)]
    assert all(first_dev(scores, s, th) == 1 for th in gates)
    assert run_max(scores, s, 1) is NEG and scores[0] == 1
    # (iii) the last step: decisions 0..3, scores (0.1, 0.1, 0.1, 0.95), theta = 0.9 refuses at decision 3
    law = {(1, None): {F(1, 10): F(1)}, (2, F(1, 10)): {F(1, 10): F(1)}, (3, F(1, 10)): {F(1, 10): F(1)},
           (4, F(1, 10)): {F(19, 20): F(1)}}
    tree = Tree(4, 0, law, {}, True)
    fd, cd = tree.first_dev(F(9, 10)), tree.cont_dev(F(9, 10))
    h4 = tree.decision_hist(next(iter(tree.baseline())), 4)
    assert list(fd) == [h4] and cd == {}
    path_scores = (F(1, 10), F(1, 10), F(1, 10), F(19, 20))
    assert run_max(path_scores, 0, 3) <= F(9, 10) < path_scores[3]
    print("PASS A3: witnesses. (i) scores (0.85, 0.95, 0.95, 1) with floor 0.80 give first deviations at decisions {0, 1, 3}. "
          "(ii) with s = 1 the scores (1, 1, 0.1) make decision 1 a first deviation of every gate in [0.8, 1) although an "
          "earlier score is 1. (iii) On the path (0.1, 0.1, 0.1, 0.95) with theta = 0.9, mbar <= theta < m holds at the "
          "last decision, but the refusal there ends the episode, so h is a terminal deviation and not in D+")


# --------------------------------------------------------------------------------------------
# Check B: explicit mechanisms, restored runs, identification
# --------------------------------------------------------------------------------------------

class Lin:
    """Affine form c0 + sum_k c_k y_k in the unknown refusal-continuation laws y."""
    __slots__ = ("c",)

    def __init__(self, c=None):
        self.c = dict(c or {})

    def __mul__(self, k):
        return Lin({a: v * k for a, v in self.c.items()})

    __rmul__ = __mul__

    def __add__(self, o):
        r = dict(self.c)
        for a, v in o.c.items():
            r[a] = r.get(a, 0) + v
        return Lin(r)

    def val(self, y):
        return sum(v * (1 if a is None else y[a]) for a, v in self.c.items())


def const(x):
    return Lin({None: F(x)})


class Tree:
    """T decisions. Coordinates in generation order, for j = 1..T: score m_j, decision d_j, harm h_j;
    then the tail Z. The gate kappa_theta refuses at decision j >= s+1 when m_j > theta. After the first
    refusal the unreplaced kernels read the full history (condition (U)): the score at step j is the fixed
    value post[j], later harm coordinates are 0, and the tail Z ~ Bernoulli(y[cylinder]) is free. The gate
    keeps refusing later decisions when post[j] > theta. A refusal at the last decision j = T is an abort: the
    episode ends there and the declared valuation gives X = 1 iff harm accrued before. A refusal at j < T
    is followed by decision j+1. X = 1 iff some harm coordinate or Z equals 1."""

    def __init__(self, T, s, score_law, harm_p, shared_refusal=True, post=None):
        self.T, self.s, self.score_law, self.harm_p, self.shared = T, s, score_law, harm_p, shared_refusal
        self.post = post or {j: F(1, 10) for j in range(1, T + 1)}

    def refusal(self, theta):
        return "refuse" if self.shared else f"refuse@{theta}"

    def policy(self, theta, j, score):
        if theta is None or not (j - 1 >= self.s and score > theta):
            return "exec"
        return self.refusal(theta)

    def forward(self, theta):
        """{path: probability (Lin)} of the mechanism with decision rule f^theta (None = pass-all)."""
        out = {}

        def add(path, p):
            out[path] = out.get(path, Lin()) + p

        def rec(j, hist, prob, refused, prev):
            if j > self.T:
                if refused is None:
                    add(hist + (("Z", 0),), prob)
                else:
                    ck = hist[: hist.index(refused) + 1]
                    base = prob.c[None]                    # prob is a constant here, so the product is affine
                    for z in (0, 1):
                        w = Lin({ck: F(1)}) if z else Lin({None: F(1), ck: F(-1)})
                        add(hist + (("Z", z),), Lin({a: v * base for a, v in w.c.items()}))
                return
            law = {self.post[j]: F(1)} if refused is not None else self.score_law[(j, prev)]
            for m, pm in law.items():
                h2 = hist + ((f"m{j}", m),)
                d = self.policy(theta, j, m)
                h2 = h2 + ((f"d{j}", d),)
                r2 = refused if refused is not None else (h2[-1] if d != "exec" else None)
                if d != "exec" and j == self.T:            # refusal at the last decision: abort
                    add(h2, prob * pm)
                    continue
                if r2 is None:
                    hp = self.harm_p.get(j, F(0))
                    hl = {1: hp, 0: 1 - hp}
                else:
                    hl = {0: F(1)}
                for hv, ph in hl.items():
                    if ph == 0:
                        continue
                    rec(j + 1, h2 + ((f"h{j}", hv),), prob * (pm * ph), r2, m if refused is None else prev)

        rec(1, (), const(1), None, None)
        return out

    @staticmethod
    def X(path):
        return int(any(v for n, v in path if n[0] == "h" or n == "Z"))

    # -- derived objects -----------------------------------------------------------------------
    def decision_hist(self, path, j):
        return path[: 3 * (j - 1) + 1]          # through the score m_j, before decision d_j

    def step(self, h):
        return (len(h) - 1) // 3 + 1            # j of the decision history h

    def baseline(self):
        return {p: pr.c[None] for p, pr in self.forward(None).items()}

    def first_dev(self, theta):
        """{h: weight w_h} of every first-deviation history of the gate (positive baseline mass)."""
        out = {}
        for p, pr in self.baseline().items():
            for j in range(1, self.T + 1):
                if self.policy(theta, j, dict(p)[f"m{j}"]) != "exec":
                    h = self.decision_hist(p, j)
                    out[h] = out.get(h, F(0)) + pr
                    break
        return out

    def cont_dev(self, theta):
        """The continuing first deviations D+: those before the last step."""
        return {h: w for h, w in self.first_dev(theta).items() if self.step(h) < self.T}

    def all_decisions(self):
        """Every decision history of positive baseline mass: {h: P0([h])}."""
        out = {}
        for p, pr in self.baseline().items():
            for j in range(1, self.T + 1):
                h = self.decision_hist(p, j)
                out[h] = out.get(h, F(0)) + pr
        return out

    def key(self, theta, h):
        """The cylinder [h^+] of the replayed prefix: h followed by the gate's refusal."""
        return h + ((f"d{self.step(h)}", self.refusal(theta)),)

    def xset(self, fwd_th, theta, h):
        """Values of X on the paths of the gate that extend h^+ (the feasible set)."""
        k = self.key(theta, h)
        return {Tree.X(p) for p in fwd_th if p[: len(k)] == k}


def row_of(tree, theta, g, forward_cache):
    """N(y) = P^kappa(X=1, [g]) as an affine form, and P^kappa([g]) for the restored run of the gate
    theta from the stored history g (the law of X in that run is N / P^kappa([g]))."""
    num, tot = Lin(), Lin()
    for p, pr in forward_cache[theta].items():
        if p[: len(g)] == g:
            tot = tot + pr
            if Tree.X(p):
                num = num + pr
    assert all(a is None or v == 0 for a, v in tot.c.items()), "P([g]) must not depend on y"
    den = tot.c.get(None, F(0))
    assert den > 0
    return num, den


def rank(rows, keys):
    m = [[r.get(k, F(0)) for k in keys] for r in rows]
    rk = 0
    for col in range(len(keys)):
        piv = next((i for i in range(rk, len(m)) if m[i][col] != 0), None)
        if piv is None:
            continue
        m[rk], m[piv] = m[piv], m[rk]
        pv = m[rk][col]
        m[rk] = [x / pv for x in m[rk]]
        for i in range(len(m)):
            if i != rk and m[i][col] != 0:
                f = m[i][col]
                m[i] = [a - f * b for a, b in zip(m[i], m[rk])]
        rk += 1
    return rk


def in_rowspace(rows, vec, keys):
    return rank(rows, keys) == rank(rows + [vec], keys)


VALS = [F(3, 10), F(17, 20), F(9, 10), F(19, 20), F(1)]


def random_tree(rng, shared=True, T=None, s=None):
    T = T or rng.choice([2, 3])
    s = rng.choice([0, 0, 1]) if s is None else s
    law = {}

    def draw():
        vs = rng.sample(VALS, rng.choice([2, 3]))
        ws = [rng.randint(1, 4) for _ in vs]
        return {v: F(w, sum(ws)) for v, w in zip(vs, ws)}

    for j in range(1, T + 1):
        for prev in ([None] if j == 1 else VALS):
            law[(j, prev)] = draw()
    harm_p = {j: rng.choice([F(0), F(1, 4)]) for j in range(1, T + 1)}
    post = {j: rng.choice([F(1, 10), F(19, 20), F(19, 20)]) for j in range(1, T + 1)}
    return Tree(T, s, law, harm_p, shared, post)


THETAS = [F(41, 50), F(22, 25), F(23, 25), F(97, 100)]   # all above theta_min = 0.80


def setup(tree, K):
    fwd = {th: tree.forward(th) for th in K}
    fd = {th: tree.first_dev(th) for th in K}
    cd = {th: tree.cont_dev(th) for th in K}
    keys = sorted({a for th in K for pr in fwd[th].values() for a in pr.c if a is not None}, key=repr)
    op = {th: {h for h in cd[th] if len(tree.xset(fwd[th], th, h)) == 2} for th in K}   # open for th
    return fwd, fd, cd, keys, op


def target_row(tree, th, fwd):
    num = Lin()
    for p, pr in fwd[th].items():
        if Tree.X(p):
            num = num + pr
    return num


def ident_check(tree, K, S, evidence, fwd, fd, cd, keys):
    """For each gate: is its law identified from the evidence? `evidence` is 'narrow' or 'broad'."""
    rows = []
    for th in K:
        for g in S:
            if evidence == "narrow":
                use = g in cd[th]
            else:  # restore at g and run kappa from g when kappa agrees with f^0 at every earlier decision
                use = not any(len(h) < len(g) and g[: len(h)] == h for h in fd[th])
            if use:
                num, den = row_of(tree, th, g, fwd)
                rows.append({k: v for k, v in num.c.items() if k is not None and v != 0})
    out = {}
    for th in K:
        tg = {k: v for k, v in target_row(tree, th, fwd).c.items() if k is not None and v != 0}
        out[th] = in_rowspace(rows, tg, keys)
    return out


def check_B(n_trees=600, seed=7):
    rng = random.Random(seed)
    n_narrow = n_broad = n_least = n_width = n_strict = n_cases = 0
    n_id = n_not = n_closed_free = n_term = n_spec = n_every_fail = n_open_gates = 0
    for it in range(n_trees):
        tree = random_tree(rng, shared=rng.random() < 0.6)
        K = sorted(rng.sample(THETAS, rng.randint(1, 4)))
        fwd, fd, cd, keys, op = setup(tree, K)
        base = tree.baseline()
        for th in K:                                          # w_h from the baseline law of the cylinder
            for h, w in fd[th].items():
                assert w == sum(pr for p, pr in base.items() if p[: len(h)] == h)
            n_term += sum(1 for h in fd[th] if h not in cd[th])
        union = sorted({h for th in K for h in fd[th]}, key=repr)
        O = {h for th in K for h in op[th]}                   # open for some gate that first deviates there
        # gate-specific openness: a history open for one gate and closed for another that deviates there
        spec = {h for h in O if any(h in cd[th] and h not in op[th] for th in K)}
        n_spec += len(spec)
        every = {h for h in O if all(h in op[th] for th in K if h in cd[th])}   # open for EVERY such gate
        decs = sorted(tree.all_decisions(), key=repr)
        pool = union + rng.sample(decs, min(3, len(decs)))
        for trial in range(12):
            S = set(rng.sample(pool, rng.randint(0, len(pool))))
            if trial == 0:
                S = set(O)                                    # exactly the open first deviations
            if trial == 1 and O:
                S = set(O) - {rng.choice(sorted(O, key=repr))}  # one open prefix missing
            # narrow evidence: identified iff every open first deviation of the gate is stored
            nar = ident_check(tree, K, S, "narrow", fwd, fd, cd, keys)
            for th in K:
                assert nar[th] == (op[th] <= S), (it, th, nar[th], op[th], S)
            n_narrow += len(K)
            n_id += sum(nar.values())
            n_not += len(K) - sum(nar.values())
            n_closed_free += sum(1 for th in K if nar[th] and not set(cd[th]) <= S)
            # all gates identified iff the union O of the open sets is a subset of S
            assert all(nar.values()) == (O <= S)
            n_least += 1
            # the target coefficient of a first deviation is w_h when it is open for the gate, else 0
            for th in K:
                tg = target_row(tree, th, fwd)
                for h, w in cd[th].items():
                    assert tg.c.get(tree.key(th, h), F(0)) == (w if h in op[th] else F(0))
                n_width += 1
            for th2 in K:
                for g in S:
                    if g in cd[th2]:
                        num, den = row_of(tree, th2, g, fwd)
                        assert [k for k, v in num.c.items() if k is not None and v != 0] in (
                            [], [tree.key(th2, g)])
            # broad evidence: identified iff every open first deviation has a stored ancestor-or-self
            brd = ident_check(tree, K, S, "broad", fwd, fd, cd, keys)
            for th in K:
                cover = all(any(h[: len(g)] == g for g in S) for h in op[th])
                assert brd[th] == cover, (it, th, brd[th], cover)
                n_broad += 1
                if brd[th] and not (op[th] <= S):
                    n_strict += 1
            n_cases += 1
        # the least set O identifies every gate; the reading "open for every such gate" can fail
        assert all(ident_check(tree, K, set(O), "narrow", fwd, fd, cd, keys).values())
        if spec and not all(ident_check(tree, K, set(every), "narrow", fwd, fd, cd, keys).values()):
            n_every_fail += 1
        n_open_gates += len(O)
    print(f"PASS B1: {n_trees} random trees, {n_cases} stored sets; narrow evidence: {n_narrow} gate checks "
          f"(identified iff open first deviations stored), {n_least} least-set checks (iff the union O of the open sets is in S), "
          f"{n_width} coefficient checks; broad evidence: {n_broad} gate checks (identified iff covered by a stored "
          f"ancestor-or-self), of which {n_strict} identified with an open first deviation unstored")
    print(f"      non-vacuity: narrow gate checks identified {n_id}, not identified {n_not}; "
          f"{n_closed_free} identified although a closed first deviation is unstored; {n_term} terminal first deviations "
          f"(last-step refusals) need no snapshot; {n_spec} histories open for one gate and closed for another; "
          f"in {n_every_fail} trees the set open for every such gate fails to identify")
    assert n_id > 1000 and n_not > 1000 and n_closed_free > 100 and n_strict > 100 and n_term > 100
    assert n_spec > 50 and n_every_fail > 10


def grid_identified_set(tree, th, K, S, evidence, fwd, fd, cd, keys, y_star, grid):
    """Brute force: all y on the grid consistent with the evidence; return the set of target values."""
    rows = []
    for th2 in K:
        for g in S:
            if evidence == "narrow":
                use = g in cd[th2]
            else:
                use = not any(len(h) < len(g) and g[: len(h)] == h for h in fd[th2])
            if use:
                rows.append(row_of(tree, th2, g, fwd))
    obs = [num.val(y_star) / den for num, den in rows]
    tg = target_row(tree, th, fwd)
    vals = set()
    for combo in product(grid, repeat=len(keys)):
        y = dict(zip(keys, combo))
        if all(num.val(y) / den == o for (num, den), o in zip(rows, obs)):
            vals.add(tg.val(y))
    return vals


def formula_set(tree, th, S, fwd, fd, cd, y_star, grid):
    """nu + sum_{D+ cap S} w_h Q*_h + sum_{D+ minus S} w_h Q_h, with the singleton X-sets of closed and
    terminal first deviations fixed by the declared valuation."""
    base = tree.baseline()
    nu = sum(pr for p, pr in base.items() if Tree.X(p) and not any(p[: len(h)] == h for h in fd[th]))
    want = {nu}
    for h, w in fd[th].items():
        xs = tree.xset(fwd[th], th, h)
        if len(xs) == 1:
            part = {w * next(iter(xs))}
        elif h in S:
            part = {w * y_star[tree.key(th, h)]}
        else:
            part = {w * x for x in grid}
        want = {a + b for a in want for b in part}
    return want


def check_B2(n_trees=100, seed=11):
    """Brute-force identified sets on a grid (boundary truths included) against the formula of
    Proposition prop:snapshot-design(c) and against the rank test."""
    rng = random.Random(seed)
    grid = [F(0), F(1, 2), F(1)]
    n = n_single = n_multi = 0
    while n < n_trees:
        tree = random_tree(rng, shared=rng.random() < 0.6, T=rng.choice([2, 2, 3]))
        K = sorted(rng.sample(THETAS, rng.randint(1, 3)))
        fwd, fd, cd, keys, op = setup(tree, K)
        if not any(op[th] for th in K) or len(keys) > 6:
            continue
        y_star = {k: rng.choice(grid) for k in keys}
        union = sorted({h for th in K for h in cd[th]}, key=repr)
        S = set(rng.sample(union, rng.randint(0, len(union))))
        for th in K:
            got = grid_identified_set(tree, th, K, S, "narrow", fwd, fd, cd, keys, y_star, grid)
            assert got == formula_set(tree, th, S, fwd, fd, cd, y_star, grid), (th, got)
            tgt = {k: v for k, v in target_row(tree, th, fwd).c.items() if k is not None and v != 0}
            rows = []
            for th2 in K:
                for g in S:
                    if g in cd[th2]:
                        num, _ = row_of(tree, th2, g, fwd)
                        rows.append({k: v for k, v in num.c.items() if k is not None and v != 0})
            assert (len(got) == 1) == in_rowspace(rows, tgt, keys)
            assert (len(got) == 1) == (op[th] <= S)
            if len(got) > 1 and all(isinstance(x, F) for x in got):
                assert max(got) - min(got) == sum(cd[th][h] for h in op[th] - S)      # width formula
            n_single += len(got) == 1
            n_multi += len(got) > 1
        n += 1
    print(f"PASS B2: {n_trees} trees; grid enumeration of the identified set equals the formula "
          f"nu + sum_(D+ cap S) w_h Q_h + sum_(D+ minus S) w_h Q_h, is a singleton exactly when O is in S, and has width "
          f"sum_(O minus S) w_h otherwise ({n_single} singleton sets, {n_multi} wider sets)")
    assert n_single > 20 and n_multi > 20


def check_gate_specific():
    """Two nested gates 0.92 < 0.97 and one history h (score 1 at the first step) that both deviate at.
    After the refusal the score is 0.95: gate 0.92 refuses again at the last step (an abort, X fixed, so h is
    closed for it), gate 0.97 executes it (X free, so h is open for it)."""
    law = {(1, None): {F(1): F(1)}, (2, F(1)): {F(1, 10): F(1)}}
    tree = Tree(2, 0, law, {1: F(0), 2: F(0)}, True, {1: F(1, 10), 2: F(19, 20)})
    K = [F(23, 25), F(97, 100)]
    fwd, fd, cd, keys, op = setup(tree, K)
    (h,) = cd[K[0]]
    assert list(cd[K[1]]) == [h] and op[K[0]] == set() and op[K[1]] == {h}
    grid = [F(k, 4) for k in range(5)]
    y_star = {k: F(1, 4) for k in keys}
    res = {}
    for name, S in (("none", set()), ("h", {h})):
        res[name] = [sorted(grid_identified_set(tree, th, K, S, "narrow", fwd, fd, cd, keys, y_star, grid)) for th in K]
    assert res["none"][0] == [F(0)] and res["none"][1] == grid and res["h"][0] == [F(0)] and res["h"][1] == [F(1, 4)]
    every = {x for x in op[K[0]] | op[K[1]] if x in op[K[0]] and x in op[K[1]]}      # open for every gate deviating there
    assert every == set()
    print("PASS B3: gate-specific openness: h is a first deviation of gates 0.92 and 0.97; it is closed for 0.92 (a later refusal at "
          "the last step fixes X) and open for 0.97. With nothing stored, 0.92 is identified as {0} and 0.97 has the full range "
          "{0, 1/4, 1/2, 3/4, 1}; storing h identifies 0.97 as {1/4}. The set open for every such gate is empty and identifies "
          "only 0.92")


def check_counterexample():
    """The two-prefix tree: m_1 = 0.90, m_2 = 0.95, K = {0.85, 0.92}, S = {h0}."""
    law = {(1, None): {F(9, 10): F(1)}, (2, F(9, 10)): {F(19, 20): F(1)}, (3, F(19, 20)): {F(1, 10): F(1)}}
    tree = Tree(3, 0, law, {1: F(0), 2: F(0), 3: F(0)}, True)
    K = [F(17, 20), F(23, 25)]
    fwd, fd, cd, keys, op = setup(tree, K)
    assert [len(cd[th]) for th in K] == [1, 1]
    h0, h1 = sorted({h for th in K for h in cd[th]}, key=len)
    assert cd[K[0]] == {h0: 1} and cd[K[1]] == {h1: 1}
    S = {h0}
    grid = [F(k, 4) for k in range(5)]
    y_star = {k: F(1, 4) if len(k) == len(h0) + 1 else F(3, 4) for k in keys}
    res = {}
    for ev in ("narrow", "broad"):
        res[ev] = [sorted(grid_identified_set(tree, th, K, S, ev, fwd, fd, cd, keys, y_star, grid)) for th in K]
    assert res["narrow"][0] == [F(1, 4)] and res["narrow"][1] == [F(k, 4) for k in range(5)]
    assert res["broad"][0] == [F(1, 4)] and res["broad"][1] == [F(3, 4)]
    fmt = lambda xs: "{" + ", ".join(str(x) for x in xs) + "}"
    print(f"PASS B4: two-prefix counterexample: narrow evidence identifies P(X=1) under kappa_0.85 as "
          f"{fmt(res['narrow'][0])} and leaves kappa_0.92 at the full range {fmt(res['narrow'][1])}; "
          f"restoring at h0 and running kappa_0.92 from there identifies {fmt(res['broad'][1])}, although h1 is not stored")


def check_step0_cover(n_trees=80, seed=5):
    """Storing every first-decision history (the live-run cover) identifies every gate with no snapshot
    at a running maximum, under broad evidence, for the trivial record and a decision at step 0 on every path."""
    rng = random.Random(seed)
    for _ in range(n_trees):
        tree = random_tree(rng)
        K = sorted(rng.sample(THETAS, rng.randint(1, 4)))
        fwd, fd, cd, keys, op = setup(tree, K)
        S = {h for h in tree.all_decisions() if len(h) == 1}
        assert all(ident_check(tree, K, S, "broad", fwd, fd, cd, keys).values())
    print(f"PASS B5: {n_trees} trees; every step-0 history stored identifies every gate under broad evidence")


# --------------------------------------------------------------------------------------------
# Check C: running-maximum identities
# --------------------------------------------------------------------------------------------

def check_C():
    H = lambda T: sum(F(1, j) for j in range(1, T + 1))
    for T in range(1, 9):
        n = tot = 0
        cnt = [0] * T
        above = {fl: 0 for fl in range(T)}
        for perm in permutations(range(T)):
            mx, c = -1, 0
            for j, x in enumerate(perm):
                if x > mx:
                    c += 1
                    cnt[j] += 1
                    mx = x
                    for fl in above:                                  # running maxima with value above the floor fl
                        above[fl] += x > fl
            tot += c
            n += 1
        assert F(tot, n) == H(T), T
        assert all(F(cnt[j], n) == F(1, j + 1) for j in range(T))
        assert all(F(a, n) <= H(T) for a in above.values())
    # i.i.d. discrete law with ties: E[N] = sum_j sum_{v > floor} pi_v G(v-)^(j-1)
    pi = {F(3, 10): F(1, 2), F(17, 20): F(1, 5), F(19, 20): F(1, 10), F(1): F(1, 5)}
    floor = F(4, 5)
    G = lambda v: sum(p for u, p in pi.items() if u < v)
    for T in range(1, 6):
        e = F(0)
        for seq in product(pi, repeat=T):
            pr = F(1)
            for x in seq:
                pr *= pi[x]
            m, c = floor, 0
            for x in seq:
                if x > m:
                    c += 1
                    m = x
            e += pr * c
        f = sum(pi[v] * sum(G(v) ** (j - 1) for j in range(1, T + 1)) for v in pi if v > floor)
        assert e == f, (T, e, f)
    # a law with a continuous part: atom at 1 of mass 1/2 and a uniform part on [0,1) of mass 1/2, floor 0, j = 2.
    # The atomic formula keeps only the atom at 1 and gives 1/4; the exact value is 3/8, the limit of the
    # formula over uniform parts discretized to N points.
    atom_only = F(1, 2) * F(1, 2)
    lims = []
    for N in (10, 100, 1000):
        atoms = {F(k, N): F(1, 2 * N) for k in range(N)}
        atoms[F(1)] = F(1, 2)
        order = sorted(atoms)
        below, cum = {}, F(0)
        for v in order:
            below[v] = cum
            cum += atoms[v]
        exact_j2 = sum(atoms[v] * below[v] for v in order if v > 0)
        assert exact_j2 == F(3, 8) - F(1, 8 * N)
        lims.append(exact_j2)
    assert atom_only < lims[0] < lims[1] < lims[2] < F(3, 8)
    print("PASS C: P(running maximum at j) = 1/j and E[running maxima] = H_T over all orderings for T <= 8, with at most H_T above any floor; "
          "the tie formula sum_v pi_v G(v-)^(j-1) matches exhaustive enumeration for T <= 5; for an atom at 1 plus a uniform part "
          "the atomic formula gives 1/4 at j = 2, below the exact 3/8, which it approaches as the uniform part is refined")


if __name__ == "__main__":
    check_A()
    check_A_witnesses()
    check_B()
    check_B2()
    check_gate_specific()
    check_counterexample()
    check_step0_cover()
    check_C()

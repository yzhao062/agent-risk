"""Exact checks for the constructed framework example; no agent benchmark data.

Paper map: these checks support Appendix A.3 (app:diagnostic), "Records That Leave a Gate Change
Open". verify() covers the recovery example (S, B, V, H stand for stale
state, a proposal, a gate block, and recovery; for q = 0.2, r = 0.8, g = 0.5, raising h from 0.25
to 0.75 lowers damage from 0.06 to 0.02 and raises P(cost > 3) from 0.02 to 0.06, with goal
failure 0.68) and the looser-gate instance (record (1, 0, 3) for q in {0.1, 0.9} under a blocking
gate, damage 0.1 and 0.9 after the gate executes everything). verify_stricter_gate() covers the
stricter-gate instance: every p in {0, 1/4, 1/2, 1} gives one coarse record law, the replayed
and the independent forward harm both equal p, and the fibre condition fails. The recovery
example and these details are supplementary checks documented here and in checks/README.md."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product

def law(q, r, g, h):
    rows = []
    for s, b, v, recovery in product((0, 1), repeat=4):
        mass = F(1)
        for bit, parameter in zip((s, b, v, recovery), (q, r, g, h)):
            mass *= parameter if bit else 1 - parameter
        damage = s * b * (1 - v) * (1 - recovery)
        unmet = 1 - (1 - s) * b * (1 - v)
        cost = 1 + 2 * b + 3 * s * b * (1 - v) * recovery
        rows.append((mass, b, damage, unmet, cost))
    return rows

def verify():
    grid = (F(0), F(1, 5), F(1, 2), F(1))
    for q, r, g, h in product(grid, repeat=4):
        rows = law(q, r, g, h)
        assert sum(p for p, *_ in rows) == 1
        assert sum(p*d for p, _, d, _, _ in rows) == q*r*(1-g)*(1-h)
        assert sum(p*u for p, _, _, u, _ in rows) == 1-(1-q)*r*(1-g)
        assert sum(p*c for p, _, _, _, c in rows) == 1+2*r+3*q*r*(1-g)*h
        assert sum(p for p, _, _, _, c in rows if c > 3) == q*r*(1-g)*h

    def record_law(q, g):
        records = defaultdict(F)
        for p, b, d, _, c in law(q, F(1), g, F(0)):
            if p:
                records[(b, d, c)] += p
        return dict(records)

    assert record_law(F(1, 10), F(1)) == record_law(F(9, 10), F(1))
    assert record_law(F(1, 10), F(0)) != record_law(F(9, 10), F(0))
    for h, expected_damage, expected_tail in (
        (F(1, 4), F(3, 50), F(1, 50)),
        (F(3, 4), F(1, 50), F(3, 50)),
    ):
        rows = law(F(1, 5), F(4, 5), F(1, 2), h)
        assert sum(p*d for p, _, d, _, _ in rows) == expected_damage
        assert sum(p*u for p, _, _, u, _ in rows) == F(17, 25)
        assert sum(p for p, _, _, _, c in rows if c > 3) == expected_tail
    print("PASS: 256 parameter settings, 16 atoms each; all manuscript diagnostic identities.")


def stricter_gate_paths(p):
    """Appendix A.3, omitted harm timing under a stricter gate: one episode, steps 0..3,
    fixed call u, scores m_t = 1{t = 2}, pass-all baseline gate, unrecorded Z with
    P(Z = early) = p, step harm h_t = 1 iff (Z, t) is (early, 1) or (late, 3)."""
    scores = (0, 0, 1, 0)
    rows = []
    for z, mass in (("early", p), ("late", 1 - p)):
        harms = tuple(int((z, t) in (("early", 1), ("late", 3))) for t in range(4))
        rows.append((mass, z, scores, harms))
    return rows


def stricter_gate_forward(p, theta):
    """Forward run of the replaced mechanism, independent of the baseline rows and of sigma:
    draw Z, step t = 0..3, the gate aborts if m_t > theta (terminal; the valuation adds no harm),
    and the sandbox kernel is called only on executed steps."""
    total = F(0)
    for z, mass in (("early", p), ("late", 1 - p)):
        loss = 0
        for t in range(4):
            m_t = int(t == 2)                      # monitor score for the fixed call u
            if m_t > theta:                        # stopping gate kappa_theta aborts
                break
            h_t = int((z, t) in (("early", 1), ("late", 3)))   # sandbox kernel on the executed call
            loss = max(loss, h_t)
        total += mass * loss
    return total


def verify_stricter_gate():
    grid = (F(0), F(1, 4), F(1, 2), F(1))
    thresholds = (F(0), F(1, 2), F(9, 10))
    coarse_laws, replayed, forward = {}, {}, {}
    for p in grid:
        rows = stricter_gate_paths(p)
        assert sum(m for m, *_ in rows) == 1
        # Baseline: every call executes, so L = max_t h_t = 1 on both paths.
        assert all(max(h) == 1 for _, _, _, h in rows)
        # Coarse record: proposals, scores, gate decisions, and L; no per-step harm.
        coarse = defaultdict(F)
        full = defaultdict(F)
        for m, _, s, h in rows:
            if m:
                coarse[(("u",) * 4, s, ("exec",) * 4, max(h))] += m
                full[(("u",) * 4, s, ("exec",) * 4, h)] += m
        coarse_laws[p] = dict(coarse)
        for theta in thresholds:
            # Replay: first deviation is the first step with m_t > theta; it is terminal,
            # and the declared valuation keeps the harm accrued before the abort.
            rep = F(0)
            for m, _, s, h in rows:
                sigma = min(t for t in range(4) if s[t] > theta)
                assert sigma == 2
                rep += m * max(h[:sigma])
            replayed[(p, theta)] = rep
            forward[(p, theta)] = stricter_gate_forward(p, theta)
            assert rep == forward[(p, theta)] == p
            # The full record identifies the replayed harm: P(h_0 = 1 or h_1 = 1).
            assert sum(mass for rec, mass in full.items() if max(rec[3][:2]) == 1) == p
    # Every p gives the same coarse record law (a point mass), yet replayed harm runs over the grid.
    first = coarse_laws[grid[0]]
    assert all(law == first for law in coarse_laws.values()) and len(first) == 1
    assert {replayed[(p, F(1, 2))] for p in grid} == set(grid)
    # Fibre condition of the logging proposition fails: the two paths share one coarse log value
    # but carry contributions delta_1 (early) and delta_0 (late).
    (_, _, s_e, h_e), (_, _, s_l, h_l) = stricter_gate_paths(F(1, 2))
    assert (s_e, max(h_e)) == (s_l, max(h_l))
    assert (max(h_e[:2]), max(h_l[:2])) == (1, 0)
    print("PASS: stricter-gate omitted-timing instance; equal coarse record laws for p in {0, 1/4, 1/2, 1},"
          " replayed harm p (1 against 0 at the extremes), forward run matches replay, fibre condition fails.")


if __name__ == "__main__":
    verify()
    verify_stricter_gate()

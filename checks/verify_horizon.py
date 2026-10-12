"""Exact checks for the long-horizon corollary (Corollary E.7.1 (cor:horizon)).

A gate acts once per step t = 0..T. At each step a recorded monitor bit U_t ~ Bern(p_t) is drawn,
and the executed call causes harm with probability q0 (U_t = 0) or q1 (U_t = 1). The loss is the
indicator that harm occurred in the episode. The baseline executes every call.

Stopping replacement: abort at the first step with U_t = 1, valued at the harm accrued before that
step (A3). Continuing replacement: at the first step with U_t = 1, the call is replaced by a retry
that reaches an input no baseline path reaches; an unknown kernel then decides whether later harm
occurs, one free value per replayed prefix (saturation with distinct inputs). Harm already accrued
before the retry cannot be undone, so the feasible set after a prefix with accrued harm is {1}.

The script computes forward laws under each replacement by enumeration, independently of the
replay formulas, and checks (i) the product formula for the agreement mass, (ii) identification of
the stopping law at every horizon, and (iii) the width of the identified mean interval for the
continuing replacement. The width equals the spread-weighted continuing mass, and equals the
continuing mass when no harm can accrue before a deviation.


Paper map: Corollary E.7.1 (cor:horizon) in Appendix E.7 (app:horizon).
"""
from fractions import Fraction as F
from itertools import product


def bern(p, x):
    return p if x == 1 else 1 - p


def baseline_paths(ps, q0, q1):
    """Yield (U tuple, Y tuple, probability) for the baseline; Y_t is harm at step t."""
    T1 = len(ps)
    for us in product((0, 1), repeat=T1):
        pu = F(1)
        for p, u in zip(ps, us):
            pu *= bern(p, u)
        for ys in product((0, 1), repeat=T1):
            py = F(1)
            for u, y in zip(us, ys):
                py *= bern(q1 if u else q0, y)
            yield us, ys, pu * py


def first_one(us):
    for t, u in enumerate(us):
        if u == 1:
            return t
    return None


def stopping_forward(ps, q0, q1):
    """Forward law of L under the stopping gate: simulate step by step."""
    law = {0: F(0), 1: F(0)}
    T1 = len(ps)

    def rec(t, harm, prob):
        if prob == 0:
            return
        if t == T1:
            law[harm] += prob
            return
        for u in (0, 1):
            pu = prob * bern(ps[t], u)
            if u == 1:
                law[harm] += pu  # abort: accrued harm, no later harm (A3)
                continue
            for y in (0, 1):
                rec(t + 1, max(harm, y), pu * bern(q0, y))

    rec(0, 0, F(1))
    return law


def stopping_replay(ps, q0, q1):
    """Replay from the baseline law: agreement part plus terminal contributions valued by (A3)."""
    law = {0: F(0), 1: F(0)}
    for us, ys, pr in baseline_paths(ps, q0, q1):
        s = first_one(us)
        if s is None:
            law[max(ys)] += pr
        else:
            accrued = max(ys[:s]) if s > 0 else 0
            law[accrued] += pr
    return law


def continuing_forward_means(ps, q0, q1):
    """Forward mean of L under the continuing replacement for every vertex choice of the free kernel.

    The free kernel assigns L in {0, 1} after each replayed prefix (U_<s = 0, Y_<s, U_s = 1), one
    independent value per prefix, which is the saturated class with distinct inputs.
    """
    T1 = len(ps)
    prefixes = []
    for s in range(T1):
        for ys in product((0, 1), repeat=s):
            prefixes.append((s, ys))
    if len(prefixes) <= 7:
        choices = list(product((0, 1), repeat=len(prefixes)))
    else:
        # The class is too large to enumerate; use both constant choices plus a fixed set of mixed
        # vertices, so that the extremes are tested against interior vertices as well.
        choices = [tuple([0] * len(prefixes)), tuple([1] * len(prefixes))]
        choices += [tuple((i * 7 + j * 3) % 2 for j in range(len(prefixes))) for i in range(8)]
        choices += [tuple(1 if (j % (i + 2)) == 0 else 0 for j in range(len(prefixes))) for i in range(8)]
    means = []
    for choice in choices:
        g = dict(zip(prefixes, choice))
        mean = F(0)

        def rec(t, ys, prob):
            nonlocal mean
            if prob == 0:
                return
            if t == T1:
                mean += prob * max(ys, default=0)
                return
            for u in (0, 1):
                pu = prob * bern(ps[t], u)
                if u == 1:
                    mean += pu * max(max(ys, default=0), g[(t, ys)])
                    continue
                for y in (0, 1):
                    rec(t + 1, ys + (y,), pu * bern(q0, y))

        rec(0, (), F(1))
        means.append(mean)
    return min(means), max(means)


def agreement_forward(ps):
    m = F(1)
    for p in ps:
        m *= 1 - p
    return m


def check(ps, q0, q1):
    T1 = len(ps)
    # (i) agreement mass: forward probability of never deviating equals the product of (1 - hazard).
    agree_baseline = sum(pr for us, ys, pr in baseline_paths(ps, q0, q1) if first_one(us) is None)
    hazards = []
    surv = F(1)
    for t in range(T1):
        at_risk = sum(pr for us, ys, pr in baseline_paths(ps, q0, q1)
                      if (first_one(us) is None or first_one(us) >= t))
        dev = sum(pr for us, ys, pr in baseline_paths(ps, q0, q1) if first_one(us) == t)
        eta = dev / at_risk if at_risk else F(0)
        hazards.append(eta)
        surv *= 1 - eta
    assert surv == agree_baseline == agreement_forward(ps)
    # (ii) stopping: replay equals the forward law at this horizon.
    assert stopping_forward(ps, q0, q1) == stopping_replay(ps, q0, q1)
    # (iii) continuing: identified mean interval from the forward class equals the replay interval,
    # and its width equals the spread-weighted continuing mass (the continuing mass
    # 1 - prod(1 - eta_t) when no harm can accrue before a deviation).
    lo, hi = continuing_forward_means(ps, q0, q1)
    # Replay interval from the baseline law: the agreement part, plus, on each continuing prefix,
    # min and max of the feasible set (accrued harm or {0, 1}).
    rlo = rhi = F(0)
    spread = F(0)
    for us, ys, pr in baseline_paths(ps, q0, q1):
        s = first_one(us)
        if s is None:
            rlo += pr * max(ys)
            rhi += pr * max(ys)
        else:
            accrued = max(ys[:s], default=0)
            rlo += pr * accrued
            rhi += pr
            spread += pr * (1 - accrued)
    eps = 1 - surv
    # The extremes are attained by the constant choices; full enumeration (up to 3 steps) and the
    # sampled vertices (longer horizons) never leave the replay interval. The width is the
    # spread-weighted continuing mass, at most the continuing mass, with equality when no harm can
    # accrue before a deviation.
    assert (lo, hi) == (rlo, rhi)
    assert hi - lo == spread <= eps
    if q0 == 0:
        assert spread == eps
    return hazards, surv, hi - lo


def main():
    q0, q1 = F(1, 20), F(1, 2)
    p = F(1, 10)
    rows = []
    for T1 in range(1, 6):
        # q0 = 0: no harm can accrue before the first deviation, so every continuing prefix keeps
        # the feasible set {0, 1} and the width equals the continuing mass.
        hazards, m, width = check([p] * T1, F(0), q1)
        assert all(h == p for h in hazards) and m == (1 - p) ** T1
        rows.append((T1, m, width))
    # q0 = 1/20: accrued harm shrinks some feasible sets to {1}; two steps give width 371/2000,
    # below the continuing mass 19/100.
    _, m2, w2 = check([p] * 2, q0, q1)
    assert m2 == F(81, 100) and w2 == F(371, 2000) < 1 - m2
    assert all(rows[i][2] < rows[i + 1][2] for i in range(len(rows) - 1))
    # Varying hazards: the product formula and the bound by the smallest hazard.
    ps = [F(1, 10), F(1, 5), F(1, 20), F(1, 4)]
    hazards, m, width = check(ps, q0, q1)
    assert m <= (1 - min(hazards)) ** len(ps)
    # Fifteen decisions, each with hazard 1/10, leave agreement mass (9/10)^15.
    m15 = agreement_forward([F(1, 10)] * 15)
    assert abs(float(m15) - 0.2059) < 1e-4
    print("PASS H1 agreement mass equals the product of (1 - hazard) for horizons 1 to 5 and for "
          "varying hazards (1/10, 1/5, 1/20, 1/4), and is bounded by (1 - min hazard)^steps.")
    print("PASS H2 stopping replay equals the forward law at every horizon 1 to 5.")
    print("PASS H3 continuing replacement with no harm before a deviation: identified mean width equals the "
          "continuing mass " + ", ".join(f"T+1={t}: {w}" for t, _, w in rows)
          + "; strictly increasing toward 1.")
    print("PASS H5 accrued harm cannot be undone: with harm before a deviation possible, the width is the "
          "spread-weighted continuing mass, 371/2000 for two steps against continuing mass 19/100.")
    print(f"PASS H4 hazard 1/10 at each of fifteen decisions leaves agreement mass (9/10)^15 = {float(m15):.4f}.")


if __name__ == "__main__":
    main()

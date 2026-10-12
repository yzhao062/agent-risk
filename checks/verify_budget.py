#!/usr/bin/env python
"""Exact checks for budgeted targeted execution and finite sequential bounds.

Paper map: Proposition F.7.1 (prop:budget; Appendix F.7 (app:budget)), parts (i) and (iii);
the design D1 numbers of the budget design table (tab:budget; not printed; archived in
supplement/appendix-D.md#budget-design-table; Appendix D.3 (app:demo-gates)), Section 4.1, and Table 2 (tab:statements).

Run with: python verify_budget.py

Checks:
- (B1) Exhaustive coverage of hypergeometric one-sided upper count D^U:
  For every unresolved pool size N up to 30, every true harm count D in {0,...,N},
  every sample size k in {0,...,N}, and error budget e in {1/40, 1/120},
  the coverage failure probability P(D > D^U(X, k; e)) <= e, exactly.
- (B2) Nested looks coverage under sequential sampling without replacement:
  For N up to 20 and looks (k1 < k2 < k3) with look error budgets e_j summing to e,
  the probability that any look's upper count falls below D is at most e
  (enumerating the exact nested sampling law).
- (B3) The E1 design D1:
  N = 479 unresolved units with 8 harmful, b = 27 baseline harms, n = 1200 cohort size,
  sequential looks at k = (34, 50, 80), e_j = 1/120 (sum = 1/40), a_L = 1/40, target 1/10:
  the largest certifying cohort count is 99;
  the exact certification probability equals 3810952577729839/3812364243804765
  (0.999629714 to 9 decimal places);
  the expected charged count equals 1077647242874066/23105237841241
  (46.6408202 to 7 decimal places);
  design D1 never certifies target 1/20.
- (B4) The live comparator:
  Under Binomial(n_L, 36/1200), the smallest sample size n_L where the one-sided 95%
  Clopper-Pearson upper bound is at most 1/10 with power >= 4/5 is n_L = 76,
  with exact probability 0.805651725.

All calculations use exact rational arithmetic (fractions.Fraction) and Python standard library.
"""
from fractions import Fraction as F
import math
import sys
import time

comb = math.comb


# ---------------------------------------------------------------------------
# Core Distributional and Bounding Primitives
# ---------------------------------------------------------------------------

def p_hypergeom(N, D, k, x):
    """P(X = x) for X ~ Hypergeometric(N, D, k)."""
    if x < max(0, k - (N - D)) or x > min(k, D):
        return F(0)
    return F(comb(D, x) * comb(N - D, k - x), comb(N, k))


def p_hypergeom_le(N, D, k, x):
    """P(X <= x) for X ~ Hypergeometric(N, D, k)."""
    if x < 0:
        return F(0)
    if x >= min(k, D):
        return F(1)
    low = max(0, k - (N - D))
    s = sum(comb(D, j) * comb(N - D, k - j) for j in range(low, x + 1))
    return F(s, comb(N, k))


def solve_du(N, k, x, e):
    """Exact one-sided upper count D^U(x, k; e).

    Definition:
      D^U(x, k; e) = max{D in {x, ..., N - k + x} : P_Hypergeom(N, D, k)(X <= x) > e},
      with D^U = N at k = 0.
    """
    if k == 0:
        return N
    low = x
    high = N - k + x
    best = x
    denom = comb(N, k)
    while low <= high:
        mid = (low + high) // 2
        # P_mid(X <= x)
        j_low = max(0, k - (N - mid))
        s = sum(comb(mid, j) * comb(N - mid, k - j) for j in range(j_low, x + 1))
        if F(s, denom) > e:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def p_binom_le(n, p, z):
    """P(Binomial(n, p) <= z) computed exactly in Fractions."""
    if z < 0:
        return F(0)
    if z >= n:
        return F(1)
    a_val, b_val = p.numerator, p.denominator
    rem = b_val - a_val
    s = sum(comb(n, j) * (a_val**j) * (rem**(n - j)) for j in range(z + 1))
    return F(s, b_val**n)


def cp_upper_le_target(z, n, a, target):
    """Exact test of u_CP(z, n; a) <= target.

    Equivalence:
      u_CP(z, n; a) <= t  <=>  P(Binomial(n, t) <= z) <= a.
    """
    return p_binom_le(n, target, z) <= a


def largest_certifying_z(n, target, a):
    """Largest count z in {0, ..., n} such that u_CP(z, n; a) <= target."""
    z_star = -1
    for z in range(n + 1):
        if cp_upper_le_target(z, n, a, target):
            z_star = z
        else:
            break
    return z_star


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_b1_exhaustive_coverage():
    """(B1) Exhaustive coverage of D^U for N <= 30, all D, k, and e in {1/40, 1/120}."""
    t0 = time.time()
    n_instances = 0
    for e in (F(1, 40), F(1, 120)):
        for N in range(1, 31):
            for D in range(N + 1):
                denom = comb(N, D)
                for k in range(N + 1):
                    n_instances += 1
                    fail_prob = F(0)
                    for x in range(max(0, k - (N - D)), min(k, D) + 1):
                        du = solve_du(N, k, x, e)
                        if D > du:
                            fail_prob += F(comb(k, x) * comb(N - k, D - x), denom)
                    assert fail_prob <= e, (
                        f"B1 coverage failure at N={N}, D={D}, k={k}, e={e}: {fail_prob} > {e}"
                    )
    elapsed = time.time() - t0
    print(
        f"PASS B1: exhaustive coverage of D^U: for every N up to 30, every true D, every k, "
        f"and e in {{1/40, 1/120}}, P(D > D^U(X,k;e)) <= e holds exactly across all {n_instances} cases "
        f"({elapsed:.2f}s)."
    )


def check_b2_nested_looks():
    """(B2) Nested looks coverage for N <= 20 and looks (k1 < k2 < k3) with sum(e_j) = e."""
    t0 = time.time()
    e1 = F(1, 120)
    e2 = F(1, 120)
    e3 = F(1, 120)
    e_sum = e1 + e2 + e3  # 1/40

    # Precompute solve_du for speed across the grid
    du_cache1 = {}
    du_cache2 = {}
    du_cache3 = {}
    for N in range(3, 21):
        for k in range(1, N + 1):
            for x in range(k + 1):
                du_cache1[(N, k, x)] = solve_du(N, k, x, e1)
                du_cache2[(N, k, x)] = solve_du(N, k, x, e2)
                du_cache3[(N, k, x)] = solve_du(N, k, x, e3)

    n_checks = 0
    for N in range(3, 21):
        for D in range(N + 1):
            denom = comb(N, D)
            for k1 in range(1, N - 1):
                for k2 in range(k1 + 1, N):
                    for k3 in range(k2 + 1, N + 1):
                        n_checks += 1
                        fail_num = 0
                        for x1 in range(min(k1, D) + 1):
                            c1 = (du_cache1[(N, k1, x1)] < D)
                            for y2 in range(min(k2 - k1, D - x1) + 1):
                                x2 = x1 + y2
                                c2 = (du_cache2[(N, k2, x2)] < D)
                                for y3 in range(min(k3 - k2, D - x2) + 1):
                                    x3 = x2 + y3
                                    if D - x3 > N - k3:
                                        continue
                                    c3 = (du_cache3[(N, k3, x3)] < D)
                                    if c1 or c2 or c3:
                                        fail_num += (
                                            comb(k1, x1)
                                            * comb(k2 - k1, y2)
                                            * comb(k3 - k2, y3)
                                            * comb(N - k3, D - x3)
                                        )
                        assert F(fail_num, denom) <= e_sum, (
                            f"B2 failure at N={N}, D={D}, looks=({k1},{k2},{k3}): "
                            f"{F(fail_num, denom)} > {e_sum}"
                        )
    elapsed = time.time() - t0
    print(
        f"PASS B2: nested looks: for N up to 20 and looks (k1<k2<k3) with e_j summing to 1/40, "
        f"the probability that any look's upper count falls below D is at most 1/40 across all "
        f"{n_checks} configurations ({elapsed:.2f}s)."
    )


def check_b3_e1_design():
    """(B3) The E1 design D1: N = 479 unresolved, 8 harmful, b = 27, n = 1200, looks 34, 50, 80."""
    t0 = time.time()
    N = 479
    D = 8
    b = 27
    n = 1200
    looks = (34, 50, 80)
    e_j = F(1, 120)
    a_L = F(1, 40)
    target = F(1, 10)
    target_stricter = F(1, 20)

    # 1. Largest certifying cohort count for target 1/10
    z_max_10 = largest_certifying_z(n, target, a_L)
    assert z_max_10 == 99, f"Expected largest certifying count 99, got {z_max_10}"
    assert cp_upper_le_target(99, n, a_L, target) is True
    assert cp_upper_le_target(100, n, a_L, target) is False

    # 2. Sequential certification and charged counts under exact sampling law
    k1, k2, k3 = looks
    p_look1 = F(0)
    p_look2 = F(0)
    p_look3 = F(0)
    p_never = F(0)

    # Pre-evaluate certification thresholds for each look
    # Look 1: certifies if b + D^U(x1, k1; e_j) <= 99
    # Look 2: certifies if b + D^U(x2, k2; e_j) <= 99
    # Look 3: certifies if b + D^U(x3, k3; e_j) <= 99
    cert1 = {x: (b + solve_du(N, k1, x, e_j) <= z_max_10) for x in range(D + 1)}
    cert2 = {x: (b + solve_du(N, k2, x, e_j) <= z_max_10) for x in range(D + 1)}
    cert3 = {x: (b + solve_du(N, k3, x, e_j) <= z_max_10) for x in range(D + 1)}

    denom = comb(N, D)
    for x1 in range(min(k1, D) + 1):
        for y2 in range(min(k2 - k1, D - x1) + 1):
            x2 = x1 + y2
            for y3 in range(min(k3 - k2, D - x2) + 1):
                x3 = x2 + y3
                if D - x3 > N - k3:
                    continue
                prob = F(
                    comb(k1, x1) * comb(k2 - k1, y2) * comb(k3 - k2, y3) * comb(N - k3, D - x3),
                    denom,
                )
                if cert1[x1]:
                    p_look1 += prob
                elif cert2[x2]:
                    p_look2 += prob
                elif cert3[x3]:
                    p_look3 += prob
                else:
                    p_never += prob

    total_cert_prob = p_look1 + p_look2 + p_look3
    expected_cert_fraction = F(3810952577729839, 3812364243804765)
    assert total_cert_prob == expected_cert_fraction, (
        f"Certification probability {total_cert_prob} != {expected_cert_fraction}"
    )
    assert f"{float(total_cert_prob):.9f}" == "0.999629714"

    # Expected charged count
    ev_charged = p_look1 * k1 + p_look2 * k2 + (p_look3 + p_never) * k3
    expected_ev_fraction = F(1077647242874066, 23105237841241)
    assert ev_charged == expected_ev_fraction, (
        f"Expected charged count {ev_charged} != {expected_ev_fraction}"
    )
    assert f"{float(ev_charged):.7f}" == "46.6408202"

    # 3. Target 1/20 is never certified
    z_max_20 = largest_certifying_z(n, target_stricter, a_L)
    assert z_max_20 == 45, f"Expected largest certifying count 45 for 1/20, got {z_max_20}"
    # Even with 0 harms observed at all looks, b + D^U exceeds 45
    min_su_k1 = b + solve_du(N, k1, 0, e_j)
    min_su_k2 = b + solve_du(N, k2, 0, e_j)
    min_su_k3 = b + solve_du(N, k3, 0, e_j)
    assert min_su_k1 > z_max_20 and min_su_k2 > z_max_20 and min_su_k3 > z_max_20, (
        f"Minimum cohort bounds ({min_su_k1}, {min_su_k2}, {min_su_k3}) must exceed {z_max_20}"
    )
    elapsed = time.time() - t0
    print(
        f"PASS B3: E1 design D1: largest certifying cohort count is 99; exact certification probability "
        f"is {total_cert_prob} ({float(total_cert_prob):.9f}); expected charged count is "
        f"{ev_charged} ({float(ev_charged):.7f}); D1 never certifies target 1/20 ({elapsed:.2f}s)."
    )


def check_b4_live_comparator():
    """(B4) The live comparator: smallest n_L with CP upper bound <= 1/10 achieving power >= 4/5."""
    t0 = time.time()
    p_true = F(36, 1200)  # 3/100
    target = F(1, 10)
    a_conf = F(1, 20)     # one-sided 95% Clopper-Pearson

    smallest_nl = None
    power_at_smallest = None

    for n_L in range(1, 150):
        z_star = largest_certifying_z(n_L, target, a_conf)
        if z_star >= 0:
            power = p_binom_le(n_L, p_true, z_star)
        else:
            power = F(0)

        if power >= F(4, 5):
            smallest_nl = n_L
            power_at_smallest = power
            break

    assert smallest_nl == 76, f"Expected smallest n_L 76, got {smallest_nl}"
    assert f"{float(power_at_smallest):.9f}" == "0.805651725"
    elapsed = time.time() - t0
    print(
        f"PASS B4: live comparator: smallest n_L with P(CP upper bound from Binomial(n_L, 36/1200) <= 1/10) >= 4/5 "
        f"is {smallest_nl}, with probability {float(power_at_smallest):.9f} ({elapsed:.2f}s)."
    )


def main():
    t_start = time.time()
    check_b1_exhaustive_coverage()
    check_b2_nested_looks()
    check_b3_e1_design()
    check_b4_live_comparator()
    total_elapsed = time.time() - t_start
    print(f"All 4 budgeted targeted-execution checks PASSED in {total_elapsed:.2f}s.")


if __name__ == "__main__":
    main()

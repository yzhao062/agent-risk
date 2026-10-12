"""Exact checks for evaluation awareness as test-to-deployment transfer.

An agent reads a cue Z (system prompt, task style, sandbox artifacts) and acts through a kernel
G(harm | Z) with private randomness. Testing and deployment draw Z from different laws. Test records
give the law of (Z, harm) in testing. The question is the deployment harm probability.

Recorded cue: by the transfer result for components (Proposition G.1.1(iv) (prop:component-joint)), with the
agent's kernel unchanged and reading only Z, the identified deployment harm probability is
[c, c + P_dep(U)], where U holds the deployment cue values never seen in testing and c sums the
supported part. The script enumerates the kernel values at unseen cues on a grid and at vertices,
and checks the interval and its width.

Unrecorded cue: test records hold only the harm indicator. Two agents with the same test harm law
can have deployment harm probabilities at the two ends of [0, 1] when deployment cues lie outside
the test support. With overlapping cue laws the range is narrower (check E3).


Paper map: Remark G.4.3 (rem:p3-eval; "Oversight cues") and Proposition G.1.1(iv) (prop:component-joint).
The numerical instances [3/40, 23/40], 3/20, and [3/40, 9/40] are supplementary checks documented
here (checks E1 to E3) and in checks/README.md.
"""
from fractions import Fraction as F
from itertools import product

TEST = {"t1": F(1, 2), "t2": F(1, 2)}
G_TEST = {"t1": F(1, 10), "t2": F(1, 5)}  # harm probability given the cue, revealed by test records


def deployment_harm(dep, g):
    return sum(p * g[z] for z, p in dep.items())


def identified_interval_recorded(dep, grid):
    unseen = [z for z in dep if z not in TEST]
    values = []
    for choice in product(grid, repeat=len(unseen)):
        g = dict(G_TEST)
        g.update(dict(zip(unseen, choice)))
        values.append(deployment_harm(dep, g))
    return min(values), max(values), unseen


def check_recorded():
    grid = [F(k, 10) for k in range(11)]
    cases = [
        ({"t1": F(1, 2), "t2": F(1, 2)}, F(0)),                    # deployment looks like testing
        ({"t1": F(1, 4), "t2": F(1, 4), "d": F(1, 2)}, F(1, 2)),    # half of deployment is recognizable
        ({"d": F(1)}, F(1)),                                        # the agent always tells them apart
        ({"t1": F(1, 10), "d1": F(3, 10), "d2": F(3, 5)}, F(9, 10)),
    ]
    for dep, mass_unseen in cases:
        lo, hi, unseen = identified_interval_recorded(dep, grid)
        c = sum(p * G_TEST[z] for z, p in dep.items() if z in TEST)
        assert (lo, hi) == (c, c + mass_unseen)
        assert sum(dep[z] for z in unseen) == mass_unseen
        assert hi - lo == mass_unseen
    lo, hi, _ = identified_interval_recorded({"t1": F(1, 4), "t2": F(1, 4), "d": F(1, 2)}, grid)
    assert (lo, hi) == (F(3, 40), F(23, 40))
    print("PASS E1 recorded cue: identified deployment harm is [c, c + P_dep(unseen cues)]; widths 0, 1/2, 1, 9/10; "
          "with half of deployment recognizable the interval is [3/40, 23/40].")


def check_unrecorded():
    # Test records hold only harm. Agent A ignores the cue: G = 3/20 everywhere.
    # Agent B is cue-aware: harm 3/20 on test cues (same test harm law), harm 1 on the deployment cue.
    test_harm_A = sum(p * F(3, 20) for p in TEST.values())
    test_harm_B = sum(p * F(3, 20) for p in TEST.values())
    assert test_harm_A == test_harm_B == F(3, 20)
    dep = {"d": F(1)}
    harm_A = deployment_harm(dep, {"d": F(3, 20)})
    harm_B = deployment_harm(dep, {"d": F(1)})
    harm_C = deployment_harm(dep, {"d": F(0)})
    assert (harm_A, harm_B, harm_C) == (F(3, 20), F(1), F(0))
    print("PASS E2 unrecorded cue: agents with the same test harm law 3/20 have deployment harm 3/20, 1, or 0.")


def check_unrecorded_overlap():
    # Overlapping cue laws with an unchanged kernel: test (1/2, 1/2), deployment (3/4, 1/4) on cues
    # z0, z1. Test records hold only the harm rate 3/20, so g0 + g1 = 3/10 and deployment harm is
    # 3/4 g0 + 1/4 g1 = 3/40 + g0/2 with g0 in [0, 3/10]: the range is [3/40, 9/40], not [0, 1].
    grid = [F(k, 100) for k in range(0, 31)]
    values = []
    for g0 in grid:
        g1 = F(3, 10) - g0
        assert (g0 + g1) / 2 == F(3, 20)
        values.append(F(3, 4) * g0 + F(1, 4) * g1)
    assert (min(values), max(values)) == (F(3, 40), F(9, 40))
    print("PASS E3 unrecorded cue with overlapping cue laws (test 1/2, 1/2; deployment 3/4, 1/4): "
          "deployment harm ranges over [3/40, 9/40], so hiding the cue need not leave the set vacuous.")


if __name__ == "__main__":
    check_recorded()
    check_unrecorded()
    check_unrecorded_overlap()

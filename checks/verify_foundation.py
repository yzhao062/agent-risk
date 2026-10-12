"""Finite counterexamples for the foundation audit, using exact probabilities.

Paper map: Section 2.3, Appendix A.1 (app:formal-spec), and Appendix A.2 (app:formal-question),
including Definition A.1.1, Eq. (3) (eq:specification)."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product


def conditional_mean(rows, event):
    selected = [(p, value) for p, value, record in rows if event(record)]
    return sum(p * value for p, value in selected) / sum(p for p, _ in selected)


def cvar(rows, alpha):
    remaining = 1 - alpha
    total = F(0)
    for value, p in sorted(rows, reverse=True):
        take = min(p, remaining)
        total += take * value
        remaining -= take
    assert remaining == 0
    return total / (1 - alpha)


# Both histories have the same loss law; their next evidence updates differ.
rows = [(F(1, 4), loss, (h, loss ^ h)) for h, loss in product((0, 1), repeat=2)]
initial = [conditional_mean(rows, lambda r, h=h: r[0] == h) for h in (0, 1)]
updated = [conditional_mean(rows, lambda r, h=h: r == (h, 0)) for h in (0, 1)]
assert initial == [F(1, 2), F(1, 2)]
assert updated == [0, 1]
print("Loss-law insufficiency: same Bernoulli(1/2) and next observation 0; updated loss probabilities 0 and 1.")

# A passive but anticipative record can change the hidden-prefix posterior.
old = [(F(1, 4), x, z) for x, z in product((0, 1), repeat=2)]
replaced = [(F(1, 2), x, x) for x in (0, 1)]
old_posterior = conditional_mean(old, lambda o: o == 1)
new_posterior = conditional_mean(replaced, lambda o: o == 1)
assert old_posterior == F(1, 2) and new_posterior == 1
assert sum(p for p, _, o in old if o == 1) == sum(p for p, _, o in replaced if o == 1)
print("Anticipative observation: identical O_0 marginal/record function, but P(X=1|O_0=1) changes from 1/2 to 1 after future replacement.")

# Distributional conditioning does not impose recursive scalar-risk consistency.
direct = cvar([(F(0), F(3, 4)), (F(100), F(1, 4))], F(1, 2))
branch_zero = cvar([(F(0), F(1))], F(1, 2))
branch_mixed = cvar([(F(0), F(1, 2)), (F(100), F(1, 2))], F(1, 2))
nested = cvar([(branch_zero, F(1, 2)), (branch_mixed, F(1, 2))], F(1, 2))
assert direct == 50 and nested == 100
print("CVaR recursion: direct CVaR_0.5(L)=50, nested CVaR_0.5(CVaR_0.5(L|H))=100.")


# Prefix-preserving replacement and the full-loss continuation integral.
def full_law(future_probabilities):
    result = []
    for x, observation, future in product((0, 1), repeat=3):
        prior = F(1, 5) if x else F(4, 5)
        evidence = F(3, 4) if observation == x else F(1, 4)
        probability = future_probabilities[x]
        continuation = probability if future else 1 - probability
        result.append((prior * evidence * continuation, x, observation, x + future))
    return result


old_probabilities = (F(3, 10), F(2, 5))
old_joint = full_law(old_probabilities)
prefix = defaultdict(F)
for p, x, observation, _ in old_joint:
    prefix[x, observation] += p

for probabilities in (old_probabilities, (F(1, 10), F(9, 10))):
    replaced_joint = full_law(probabilities)
    replaced_prefix = defaultdict(F)
    for p, x, observation, _ in replaced_joint:
        replaced_prefix[x, observation] += p
    assert prefix == replaced_prefix
    for record in (0, 1):
        record_probability = sum(p for (x, o), p in prefix.items() if o == record)
        direct_law = defaultdict(F)
        mixed_law = defaultdict(F)
        for p, _, o, loss in replaced_joint:
            if o == record:
                direct_law[loss] += p / record_probability
        for x, future in product((0, 1), repeat=2):
            posterior = prefix[x, record] / record_probability
            continuation = probabilities[x] if future else 1 - probabilities[x]
            mixed_law[x + future] += posterior * continuation
        assert direct_law == mixed_law
        assert sum(mixed_law.values()) == 1
print("Prefix mixing: identity and changed continuations match the conditional joint law at both positive-probability records, retaining past loss.")

# R_0=O_0 precedes H_2=(X,O_0,Y,O_1); O_1=Y xor O_0 controls the replacement.
joint_prefix = {(x, o, y, y ^ o): F(1, 8) for x, o, y in product((0, 1), repeat=3)}
for changed in (False, True):
    continuation = {}
    full_joint = []
    for h, p in joint_prefix.items():
        x, o, y, o1 = h
        abort = o1 if changed else 0
        future_p = F(1 - o1) if changed else F(1, 2)
        continuation[h] = defaultdict(F)
        for z in (0, 1):
            mass = future_p if z else 1 - future_p
            if mass:
                loss = x + y + z + F(abort, 2)
                continuation[h][loss] += mass
                full_joint.append((p * mass, h, loss))
    recovered_prefix = defaultdict(F)
    for p, h, loss in full_joint:
        recovered_prefix[h] += p
    assert recovered_prefix == joint_prefix
    for record in (0, 1):
        direct_law, mixed_law, coarse_law = (defaultdict(F) for _ in range(3))
        for p, h, loss in full_joint:
            if h[1] == record:
                direct_law[loss] += 2 * p
        for h, p in joint_prefix.items():
            for loss, mass in continuation[h].items():
                if h[1] == record:
                    mixed_law[loss] += 2 * p * mass
                # Physical (X,Y) has the same posterior at either R_0 here.
                coarse_law[loss] += p * mass
        assert direct_law == mixed_law
        assert (coarse_law == direct_law) == (not changed)
print("Observation-dependent continuation: full joint prefix mixing holds; physical-prefix averaging fails for the record-dependent replacement.")

# A policy's observed selection can reveal information absent from evaluator records.
hypothetical = [(F(1, 2), h, h) for h in (0, 1)]
assert conditional_mean(hypothetical, lambda selected: True) == F(1, 2)
assert conditional_mean(hypothetical, lambda selected: selected == 1) == 1
print("Selection: hypothetical intervention loss probability 1/2 differs from 1 conditional on its hidden-information-based selection.")

# Replacing a random terminal outcome with its mean changes tail queries.
random_terminal = [(F(0), F(1, 2)), (F(10), F(1, 2))]
mean_terminal = [(F(5), F(1))]
assert sum(x * p for x, p in random_terminal) == sum(x * p for x, p in mean_terminal)
assert cvar(random_terminal, F(1, 2)) == 10
assert cvar(mean_terminal, F(1, 2)) == 5
print("Terminal valuation: equal mean 5; CVaR_0.5 changes from 10 to 5.")

# Same observational paths do not determine response to named-kernel replacement.
observed_one = [(u, u, u) for u in (0, 1)]  # U, X=U, Y=U
observed_two = [(u, u, x) for u, x in ((0, 0), (1, 1))]  # U, X=U, Y=X
assert observed_one == observed_two
replaced_one = [(u, 0, u) for u in (0, 1)]
replaced_two = [(u, 0, 0) for u in (0, 1)]
assert sum(y for _, _, y in replaced_one) * F(1, 2) == F(1, 2)
assert sum(y for _, _, y in replaced_two) == 0
print("Mechanism structure: identical original path laws; after X:=0, Y probabilities are 1/2 and 0.")

# Changelog

## 0.2.1 (2026-11-15)

Documentation only; the code is unchanged from 0.2.0.

- The README is organized as function blocks: certify a deployment, re-check a stricter stopping gate, bound a retry gate, compare two policies on the same tasks, and load logs. Each block opens with the question it answers and when to use it, then a runnable example with its output; the mechanism and conditions fold into details, and a reference table lists every public call.
- `tests/test_readme.py` checks that each README example prints what its comments state and that the quoted certificate is the real `to_markdown()` output.

## 0.2.0 (2026-11-14)

Adds statistical certificates: from per-episode violation outcomes, report whether the violation rate is at most a limit at a stated one-sided confidence (exact Clopper-Pearson). `replay_stop`, `retry_harm_bounds`, and the paired comparison are unchanged.

### Added

- `certify_violations(outcomes, *, limit, confidence=0.95, scope=None)` certifies the violation rate from a list, tuple, or NumPy array of per-episode outcomes. A missing label (`None` or NaN) counts as a violation and is reported in `n_unknown`; strings, bytes, mappings, multidimensional input, and any other entry raise.
- `sample_size(limit, *, confidence=0.95, max_violations=0)` returns the smallest number of episodes at which `max_violations` observed violations still pass (299 for zero and 473 for one, at limit 0.01 and confidence 0.95). It assumes the observed count equals `max_violations` and is not a power calculation.
- `certify_retry(bounds, *, limit, confidence=0.95, scope=None)` certifies the upper count of a retry gate's violation rate from a `RetryBounds`; continuing episodes with no harm before their first block count as violations and are reported in `n_unknown`.
- `Certificate` gains the fields `limit` and `scope`, the read-only properties `confidence`, `n_violations`, `passed`, and `status` (`passed`, `not_certified`, or `None` without a limit), and the methods `matches_scope`, `to_dict`, `to_json`, `to_markdown`, and `from_json`. The exported forms state the method, the counts, the three conditions of the guarantee, and the warning that separate certificates do not control the chance that at least one incorrect certificate passes. `from_json(c.to_json()) == c`.
- `scope` records the deployment a certificate covers as a non-empty mapping of JSON values, normalized to a canonical JSON string, so certificates stay hashable and comparable.
- Keyword-only `confidence` on `clopper_pearson_upper`, `certify`, and `certify_replay`, and keyword-only `limit` and `scope` on `certify` and `certify_replay`. Docstrings note that statsmodels `proportion_confint(..., method="beta")` is two-sided: `alpha=0.05` there gives the one-sided 97.5% bound, and the one-sided 95% bound needs `alpha=0.10`.
- `statsmodels>=0.13` in the `test` extra, so the one-sided comparison test runs in CI.

### Changed

- The default of `alpha` is now a sentinel. With neither `alpha` nor `confidence`, the one-sided error level is 0.05 as before; passing both raises `TypeError`. `confidence=0.95` stores `alpha=0.05`, so `certify(k, n, confidence=0.95) == certify(k, n, alpha=0.05)`. Existing positional calls keep working.
- `certify_replay` with a `limit` raises `ValueError` when the replay declares a positive `post_stop_harm`, because the binary count omits that harm. The message names two substitutes: replay with `post_stop_harm = 0`, or certify the worst-case count with `certify`.
- `compare_logging_schemes` is unchanged and returns certificates without a limit.
- The package description and the README present one workflow: certify a deployment, then re-check a stricter gate against the logs you already have.

## 0.1.0 (2026-10-11)

First public release accompanying the preprint "agent-risk: Risk Assessment for Changes to AI Agents".
Provides episode data representations, counterfactual replay of stopping and retry gates,
exact Clopper-Pearson harm certification, and paired equivalence testing for AI agents.

The tag `v0.1.0` marks `a1ab24a`, the revision that reproduces the paper. Its package code and tests match the 0.1.0 sdist on PyPI, while the sdist's `README.md` and `pyproject.toml` predate the arXiv links.

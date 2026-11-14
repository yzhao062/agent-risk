"""Risk modeling for AI agents: certify a deployment, then re-check a gate from logs.

Certify that an agent's violation rate is at most a limit, at a stated
one-sided confidence, from per-episode violation outcomes. Replay a proposed
change over logged episodes and separate what the log settles, what a declared
stop valuation fixes, and what needs bounds.

Main entry points:

* :func:`certify_violations`, :func:`sample_size`, :class:`Certificate`:
  certify a violation rate from fresh episodes, and size the sample.
* :func:`clopper_pearson_upper`, :func:`certify`: the exact one-sided bound
  on counts.
* :func:`certify_replay`, :func:`certify_retry`,
  :func:`compare_logging_schemes`: certified violation-rate bounds from replay.
* :class:`Episode`, :func:`load_episodes`, :func:`load_jsonl`: logged data.
* :func:`replay_stop`, :class:`StopValuation`: replay of a stopping gate.
* :func:`retry_harm_bounds`: outer interval for a retry gate.
* :func:`newcombe_paired_interval`, :func:`is_equivalent`: paired comparison.
"""

from .certify import (
    Certificate,
    LoggingComparison,
    certify,
    certify_replay,
    certify_retry,
    certify_violations,
    clopper_pearson_upper,
    compare_logging_schemes,
    sample_size,
)
from .episode import Episode, any_harm, load_episodes, load_jsonl
from .paired import PairedInterval, is_equivalent, newcombe_paired_interval, wilson_interval
from .retry import RetryBounds, retry_harm_bounds
from .stopping import EpisodeReplay, StopReplay, StopValuation, replay_stop

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Certificate",
    "Episode",
    "EpisodeReplay",
    "LoggingComparison",
    "PairedInterval",
    "RetryBounds",
    "StopReplay",
    "StopValuation",
    "any_harm",
    "certify",
    "certify_replay",
    "certify_retry",
    "certify_violations",
    "clopper_pearson_upper",
    "compare_logging_schemes",
    "is_equivalent",
    "load_episodes",
    "load_jsonl",
    "newcombe_paired_interval",
    "replay_stop",
    "retry_harm_bounds",
    "sample_size",
    "wilson_interval",
]

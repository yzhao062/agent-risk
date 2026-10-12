"""Minimal ControlArena additions for the "Risk Modeling for AI Agents" paper.

This package layers five small, auditable pieces on top of ControlArena's bash
(BashBench) setting, without modifying the vendored ControlArena source:

- a baseline gate: a non-intervening trusted monitor that scores every proposed
  tool call and passes all of them (``build_baseline_task``);
- a stopping gate: an Inspect approver that returns ``terminate`` once the logged
  monitor suspicion score exceeds a threshold ``theta`` (``stopping_approver``);
- a cost scorer: input tokens, output tokens, and tool-call count as separate
  per-episode coordinates (``cost_scorer``);
- a per-step harm probe: the bash side-task check run at every step boundary so
  the step of first harm is recorded (``harm_probe``, ``harm_time_scorer``);
- explicit recording of the gate rule, threshold, models, temperature, seed, and
  caps in the Inspect task metadata (``build_baseline_task`` / ``build_stopping_task``).

The arms A and B pilot (honest mode) uses the declared-policy modules instead of
the side-task probe: ``policy`` (declared policy and command checks),
``harm_ledger`` (per-call policy check on sandbox state), ``agent_policy``
(seeded honest policy), ``loss`` (declared loss with the abort valuation), and
``pilot`` (task builder). The smoke-test modules are kept for reference.

Everything here reads the suspicion score that ControlArena's trusted monitor
already writes into assistant-message metadata; nothing here changes how that
score is produced.
"""

try:
    from .gates import GateLedger, pass_all_approver, stopping_approver
    from .harm_ledger import HarmLedger, wrap_tool
    from .harm_probe import HarmState, harm_probe
    from .loss import declared_loss_scorer
    from .scorers import cost_scorer, harm_time_scorer
    from .tasks import build_baseline_task, build_stopping_task
except ImportError:
    pass

__all__ = [
    "GateLedger",
    "pass_all_approver",
    "HarmLedger",
    "wrap_tool",
    "declared_loss_scorer",
    "stopping_approver",
    "HarmState",
    "harm_probe",
    "cost_scorer",
    "harm_time_scorer",
    "build_baseline_task",
    "build_stopping_task",
]

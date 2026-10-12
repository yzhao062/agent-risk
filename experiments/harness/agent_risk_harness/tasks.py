"""Task builders for the risk-modeling harness.

``build_baseline_task`` assembles the non-intervening baseline gate; it logs a
suspicion score for every step and passes all tool calls. ``build_stopping_task``
adds the stopping gate (an Inspect approver that terminates on score > theta) on top
of the same monitor, so the score driving the decision is the score in the log.

Both bake the full configuration (models, temperature, seed, caps, gate rule,
abort valuation) into the Inspect task metadata, so an episode's log is
self-describing for later replay and auditing.
"""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai.approval import ApprovalPolicy
from inspect_ai.model import GenerateConfig, Model, get_model

from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import basic_monitor_builder
from control_arena.policy import untrusted_policy_builder
from control_arena.settings.bash.bash_setting import BashSetting
from control_arena.tools._tool_output import ToolOutputLimit

from .gates import stopping_approver
from .harm_probe import harm_probe

# --- Frozen harness settings (recorded into task metadata) --------------------

AGENT_MODEL = "vllm/Qwen/Qwen3-14B-FP8"
AGENT_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
MONITOR_MODEL = "vllm/Qwen/Qwen3-4B-Instruct-2507"
MONITOR_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
AGENT_BASE_URL = "http://127.0.0.1:8000/v1"
MONITOR_BASE_URL = "http://127.0.0.1:8001/v1"

MAX_STEPS = 15  # matches CTRL-Z and the experiment spec
# Token budget per episode. 15 steps * ~32k-token context leaves generous head
# room; 400k stops a runaway episode well before the agent server's own limits.
TOKEN_LIMIT = 400_000
# Show the model up to 30k bytes of any one tool result (recoverable via the
# read/grep recovery tools); retain up to 1 MiB in the log so nothing is silently
# truncated below that ceiling.
TOOL_OUTPUT_MAX_BYTES = 30_000
MAX_TOOL_OUTPUT = 1024 * 1024

AGENT_TEMPERATURE = 0.0
AGENT_SEED = 0
MONITOR_TEMPERATURE = 0.0
MONITOR_SEED = 0

# Abort valuation (recorded, not computed here): on abort, failure = 1, harm = harm
# accrued up to the abort (from the per-step probe), cost = cost up to the abort.
# Assuming no harm after the abort is a substantive assumption and is labelled here.
ABORT_VALUATION = {
    "failure": 1,
    "harm": "accrued_before_abort",
    "cost": "accrued_before_abort",
    "assumption": "no harm accrues after the abort (substantive)",
}


def _agent_model() -> Model:
    return get_model(
        AGENT_MODEL,
        base_url=AGENT_BASE_URL,
        api_key="EMPTY",
        config=GenerateConfig(temperature=AGENT_TEMPERATURE, seed=AGENT_SEED),
    )


def _monitor_model() -> Model:
    return get_model(
        MONITOR_MODEL,
        base_url=MONITOR_BASE_URL,
        api_key="EMPTY",
        config=GenerateConfig(temperature=MONITOR_TEMPERATURE, seed=MONITOR_SEED),
    )


def _eval_config() -> ControlEvalConfig:
    return ControlEvalConfig(
        max_steps=MAX_STEPS,
        token_limit=TOKEN_LIMIT,
        tool_output_limit=ToolOutputLimit(max_bytes=TOOL_OUTPUT_MAX_BYTES),
        max_tool_output=MAX_TOOL_OUTPUT,
        untrusted_model=_agent_model(),
        trusted_model=_monitor_model(),
        # One sample at a time on a single shared A100; raise for the real pilot.
        max_tasks=1,
        max_samples=1,
        max_sandboxes=1,
    )


def _micro_protocol():
    """Non-intervening trusted monitor, wrapped with the per-step harm probe."""
    monitor = basic_monitor_builder(model_role="trusted")
    policy = untrusted_policy_builder()
    tm = trusted_monitoring_builder(policy=policy, monitor=monitor)
    return harm_probe(inner=tm)


def _base_metadata(gate: str, theta: float | None) -> dict:
    return {
        "harness": "agent-risk-harness",
        "gate": gate,
        "gate_rule": "terminate iff suspicion_score > theta" if gate == "stopping" else "pass all",
        "theta": theta,
        "monitor_score_range": "normalised [0,1]",
        "abort_valuation": ABORT_VALUATION,
        "max_steps": MAX_STEPS,
        "token_limit": TOKEN_LIMIT,
        "tool_output_max_bytes": TOOL_OUTPUT_MAX_BYTES,
        "max_tool_output": MAX_TOOL_OUTPUT,
        "agent_model": AGENT_MODEL,
        "agent_revision": AGENT_REVISION,
        "agent_temperature": AGENT_TEMPERATURE,
        "agent_seed": AGENT_SEED,
        "monitor_model": MONITOR_MODEL,
        "monitor_revision": MONITOR_REVISION,
        "monitor_temperature": MONITOR_TEMPERATURE,
        "monitor_seed": MONITOR_SEED,
    }


def _scorers():
    # Imported here so the registry decorators run once the package is importable.
    from .scorers import cost_scorer, harm_time_scorer

    return [cost_scorer(), harm_time_scorer()]


def build_baseline_task(mode: EvalMode) -> Task:
    """Baseline gate: non-intervening trusted monitor, all tool calls pass."""
    return get_control_task(
        setting=BashSetting(sandbox_type="docker"),
        micro_protocol=_micro_protocol(),
        mode=mode,
        eval_config=_eval_config(),
        monitoring_scorers=_scorers(),
        metadata=_base_metadata("baseline", theta=None),
        task_name_prefix=f"arh_baseline_{mode.value}",
    )


def build_stopping_task(mode: EvalMode, theta: float) -> Task:
    """Stopping gate: terminate the episode once suspicion_score > theta."""
    return get_control_task(
        setting=BashSetting(sandbox_type="docker"),
        micro_protocol=_micro_protocol(),
        mode=mode,
        eval_config=_eval_config(),
        monitoring_scorers=_scorers(),
        metadata=_base_metadata("stopping", theta=theta),
        approval=[ApprovalPolicy(stopping_approver(theta), tools="*")],
        task_name_prefix=f"arh_stopping_{mode.value}_theta{theta}",
    )

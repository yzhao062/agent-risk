"""Honest agent policy with a recorded per-episode random tape.

Each episode carries an integer seed in its sample metadata. The agent samples at
temperature 0.7, and the seed sent to the server for step ``t`` is
``episode_seed * 1000 + t`` (see ``step_seed``), so the whole episode's sampling
randomness is addressed by the episode seed. Every generation setting, the step
seed, and the step's token usage are written into the assistant message
metadata, so the log is self-describing for replay.

Apart from the explicit seed and settings, this matches ControlArena's
``honest_policy``: it forwards the conversation to the untrusted model unchanged.
"""

from __future__ import annotations

from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import Tool
from inspect_ai.util import store_as
from pydantic import Field

from control_arena._control_agent import control_agent
from control_arena._control_solver import StepCounter
from control_arena._control_task_metadata import control_task_metadata

import os

AGENT_TEMPERATURE = 0.7
AGENT_TOP_P = 1.0
AGENT_MAX_TOKENS = 4096


def get_agent_max_tokens() -> int:
    val = os.environ.get("ARH_AGENT_MAX_TOKENS")
    if val:
        return int(val)
    return AGENT_MAX_TOKENS


def is_thinking_enabled() -> bool:
    return os.environ.get("ARH_THINKING", "").strip() in ("1", "true", "True")


def extract_reasoning(output, message) -> str | None:
    """Extract reasoning text normalized by Inspect AI or returned by vLLM reasoning parser."""
    for obj in (message, output):
        if obj is not None:
            for attr in ("reasoning", "reasoning_content", "thought"):
                val = getattr(obj, attr, None)
                if val:
                    return str(val)

    if message is not None and getattr(message, "metadata", None):
        for key in ("arh_reasoning", "reasoning", "reasoning_content", "thought"):
            val = message.metadata.get(key)
            if val:
                return str(val)

    if message is not None and isinstance(message.content, list):
        parts = []
        for part in message.content:
            if getattr(part, "type", None) == "reasoning" or hasattr(part, "reasoning"):
                r = getattr(part, "reasoning", None) or getattr(part, "text", None)
                if r:
                    parts.append(str(r))
            elif type(part).__name__ == "ContentReasoning":
                r = getattr(part, "reasoning", None) or getattr(part, "text", None)
                if r:
                    parts.append(str(r))
        if parts:
            return "\n".join(parts)

    if message is not None:
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            val = model_extra.get("reasoning_content") or model_extra.get("reasoning")
            if val:
                return str(val)
        source = getattr(message, "source", None)
        if isinstance(source, dict):
            val = source.get("reasoning_content") or source.get("reasoning")
            if val:
                return str(val)

    return None


def extract_reasoning_tokens(usage) -> int | None:
    """Extract reasoning token count from usage if present."""
    if not usage:
        return None
    if getattr(usage, "reasoning_tokens", None) is not None:
        return int(usage.reasoning_tokens)
    for attr in ("completion_tokens_details", "output_tokens_details"):
        details = getattr(usage, attr, None)
        if details is not None:
            if isinstance(details, dict):
                val = details.get("reasoning_tokens")
                if val is not None:
                    return int(val)
            elif hasattr(details, "reasoning_tokens"):
                val = details.reasoning_tokens
                if val is not None:
                    return int(val)
    return None


def step_seed(episode_seed: int, step: int) -> int:
    return int(episode_seed) * 1000 + int(step)


def episode_seed_from_metadata() -> int:
    md = control_task_metadata().metadata or {}
    return int(md["arh_episode_seed"])


def _resolve_int(val, default: int) -> int:
    if hasattr(val, "default"):
        val = val.default
    if val is None:
        val = default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


@control_agent
def seeded_honest_policy_builder(
    temperature: float = Field(default=AGENT_TEMPERATURE),
    top_p: float = Field(default=AGENT_TOP_P),
    max_tokens: int = Field(default=AGENT_MAX_TOKENS),
) -> Agent:
    """Honest policy on the untrusted role with a per-step seed from the episode seed."""

    async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
        step = store_as(StepCounter).count
        ep_seed = episode_seed_from_metadata()
        seed = step_seed(ep_seed, step)
        eff_max_tokens = int(os.environ.get("ARH_AGENT_MAX_TOKENS", _resolve_int(max_tokens, AGENT_MAX_TOKENS)))
        thinking = is_thinking_enabled()

        config_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": eff_max_tokens,
            "seed": seed,
        }
        if thinking:
            config_kwargs["reasoning_history"] = "none"
        config = GenerateConfig(**config_kwargs)

        output = await get_model(role="untrusted").generate(
            state._messages, tools=tools, config=config, cache=False
        )
        message = output.message
        usage = output.usage
        meta = dict(message.metadata or {})

        gen_meta = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": eff_max_tokens,
            "seed": seed,
            "cache": False,
        }
        if thinking:
            gen_meta["reasoning_history"] = "none"

        usage_meta = {
            "input_tokens": (usage.input_tokens or 0) if usage else 0,
            "output_tokens": (usage.output_tokens or 0) if usage else 0,
        }
        r_tokens = extract_reasoning_tokens(usage)
        if r_tokens is not None:
            usage_meta["reasoning_tokens"] = r_tokens
        elif thinking:
            usage_meta["reasoning_tokens"] = None

        meta.update(
            {
                "arh_step": step,
                "arh_episode_seed": ep_seed,
                "arh_step_seed": seed,
                "arh_agent_generation": gen_meta,
                "arh_agent_usage": usage_meta,
                "arh_agent_stop_reason": output.stop_reason,
            }
        )
        reasoning_text = extract_reasoning(output, message)
        if reasoning_text is not None or thinking:
            meta["arh_reasoning"] = reasoning_text

        message.metadata = meta
        state.output = output
        state._messages.append(message)
        return state

    return impl

"""E3 model calls: a replacement for tau2's ``generate`` that talks to the local
vLLM servers directly (no litellm request path), sets the per-call seed, and
records every call compactly.

Kernels (E3-PREREG):
  agent  Qwen/Qwen3-14B-FP8 on 127.0.0.1:8000, default kernels, temperature 0.7,
         top_p 1.0, max_tokens 4096, seed = episode_seed * 1000 + agent turn
  user   the same server and weights, tau2's user-simulator prompt, temperature
         0.0 (tau2's default), max_tokens 4096, seed = episode_seed * 1000 + 500
         + user turn (recorded; greedy decoding does not read it)
The agent turn index t counts the agent's generated messages from 0 (tau2's fixed
greeting is not generated and has no index). Tool-call ids returned by vLLM are
random; they are replaced by deterministic ids "e3-<role>-<turn>-<k>" before
anything else sees them (the Qwen3 chat template does not render ids, so this
changes no prompt).

Each call appends a record to the episode context: role, turn, seed, the sha256
of the request body, the number of input messages, usage, finish reason, and the
returned text and tool calls. Full request bodies are not stored; every one is a
deterministic function of the logged trajectory and can be rebuilt offline.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

import requests

AGENT_URL = "http://127.0.0.1:8000/v1"
AGENT_SERVED = "Qwen/Qwen3-14B-FP8"
AGENT_TEMPERATURE = 0.7
AGENT_TOP_P = 1.0
AGENT_MAX_TOKENS = 4096
USER_TEMPERATURE = 0.0
USER_MAX_TOKENS = 4096
USER_SEED_OFFSET = 500
RETRIES = 5
TIMEOUT_S = 600.0


class ContextLimit(Exception):
    """The server refused a prompt over the context limit (ends the episode)."""


class ServerError(Exception):
    """A call still failed after RETRIES attempts (the episode errors; outcome unknown)."""


_local = threading.local()


class EpisodeCtx:
    def __init__(self, episode_seed: int, url: str = AGENT_URL):
        self.seed = int(episode_seed)
        self.url = url
        self.agent_turn = 0
        self.user_turn = 0
        self.calls: list[dict] = []
        self.session = requests.Session()


def set_ctx(ctx: EpisodeCtx | None) -> None:
    _local.ctx = ctx


def get_ctx() -> EpisodeCtx:
    ctx = getattr(_local, "ctx", None)
    if ctx is None:
        raise RuntimeError("E3 generate called outside an episode context")
    return ctx


def body_sha(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def to_api_messages(messages) -> list[dict]:
    """tau2 messages -> OpenAI chat messages (content, tool calls, tool results)."""
    from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            d = {"role": "assistant", "content": m.content}
            if m.is_tool_call():
                d["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                   for tc in m.tool_calls]
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content, "tool_call_id": m.id})
        else:
            raise TypeError(f"unexpected message type {type(m)}")
    return out


def post_json(session: requests.Session, url: str, body: dict) -> dict:
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.post(url, json=body, timeout=TIMEOUT_S)
            if r.status_code == 400:
                txt = r.text
                if "context length" in txt or "maximum context" in txt or "too long" in txt:
                    raise ContextLimit(txt[:500])
                raise ServerError(f"HTTP 400: {txt[:500]}")
            r.raise_for_status()
            return r.json()
        except (ContextLimit, ServerError):
            raise
        except Exception as exc:  # noqa: BLE001  (connection errors, 5xx)
            last = exc
            time.sleep(min(2 ** attempt, 20))
    raise ServerError(f"{url}: failed after {RETRIES} attempts: {last!r}")


def generate(model=None, messages=None, tools=None, tool_choice=None, call_name=None, **kwargs):
    """Drop-in for tau2.utils.llm_utils.generate (agent and user simulator)."""
    from tau2.data_model.message import AssistantMessage, ToolCall
    ctx = get_ctx()
    if call_name == "agent_response":
        role, turn = "agent", ctx.agent_turn
        seed = ctx.seed * 1000 + turn
        temperature, max_tokens = AGENT_TEMPERATURE, AGENT_MAX_TOKENS
    elif call_name == "user_simulator_response":
        role, turn = "user", ctx.user_turn
        seed = ctx.seed * 1000 + USER_SEED_OFFSET + turn
        temperature, max_tokens = USER_TEMPERATURE, USER_MAX_TOKENS
    else:
        raise RuntimeError(f"E3: unexpected generate call {call_name!r}")
    body = {"model": AGENT_SERVED, "messages": to_api_messages(messages), "temperature": temperature,
            "top_p": AGENT_TOP_P, "max_tokens": max_tokens, "seed": seed}
    if tools:
        body["tools"] = [t.openai_schema for t in tools]
        body["tool_choice"] = "auto"
    t0 = time.time()
    j = post_json(ctx.session, ctx.url + "/chat/completions", body)
    choice = j["choices"][0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    raw_calls = msg.get("tool_calls") or []
    calls = []
    for k, tc in enumerate(raw_calls):
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {"__unparsed__": fn.get("arguments")}
        except json.JSONDecodeError:
            args = {"__unparsed__": fn.get("arguments")}
        calls.append(ToolCall(id=f"e3-{role}-{turn}-{k}", name=fn.get("name") or "", arguments=args,
                              requestor="assistant"))
    usage = j.get("usage") or {}
    ctx.calls.append({
        "role": role, "turn": turn, "seed": seed, "temperature": temperature, "max_tokens": max_tokens,
        "body_sha256": body_sha(body), "n_messages": len(body["messages"]),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": choice.get("finish_reason"), "content": content,
        "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
        "seconds": round(time.time() - t0, 3), "model": j.get("model"),
    })
    if role == "agent":
        ctx.agent_turn += 1
    else:
        ctx.user_turn += 1
    return AssistantMessage(role="assistant", content=content, tool_calls=calls or None, cost=0.0,
                            usage={"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                                   "completion_tokens": int(usage.get("completion_tokens") or 0)})


def install() -> None:
    """Point tau2's agent and user simulator at ``generate`` above."""
    import tau2.agent.llm_agent as la
    import tau2.user.user_simulator as us
    la.generate = generate
    us.generate = generate

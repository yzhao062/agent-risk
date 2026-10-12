"""Arm E: decisions and interface comparisons shared by the replay client and the analysis.

A decision is what the agent kernel returned at one step: either an assistant
message (text content, optional reasoning text, ordered tool calls with their raw
JSON argument strings) or a server error (the context-length rejection that ends
an episode with ``model_length``).

Interfaces (strictest first):

  raw      content, reasoning, tool names, and the raw argument strings, all exact.
  full     what later kernels read. Content and reasoning exact (None counts as "",
           as Inspect stores it), tool names, and arguments parsed from JSON with
           key order kept. Whitespace inside argument strings is excluded because
           vLLM parses the string into an object before the chat template
           re-serializes it (vllm/entrypoints/chat_utils.py), Inspect hands parsed
           arguments to the monitor and the tools, and the templates never render
           tool-call ids (vLLM draws those at random).
  full_ws  diagnostic: as ``full`` with leading and trailing whitespace of the
           content and reasoning stripped (the Qwen templates trim some of it).
  action   action-only: tool names and parsed arguments compared as objects (key
           order ignored); text is ignored.

Two error decisions agree when they have the same error class; an error never
agrees with a message.
"""

from __future__ import annotations

import json
from typing import Any

LEVELS = ("raw", "full", "full_ws", "action")


def decision_from_message(msg: dict, finish_reason: str | None = None) -> dict:
    tcs = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tcs.append([fn.get("name"), fn.get("arguments")])
    reasoning = msg.get("reasoning")
    if reasoning is None:
        reasoning = msg.get("reasoning_content")
    return {"kind": "msg", "content": msg.get("content") or "", "reasoning": reasoning or "",
            "tool_calls": tcs, "finish_reason": finish_reason}


def error_class(message: str, code: int | None) -> str:
    if "maximum context length" in (message or ""):
        return "context_length"
    return f"http_{code}"


def decision_from_error(message: str, code: int | None) -> dict:
    return {"kind": "error", "error_class": error_class(message, code), "message": (message or "")[:400]}


def _parse(raw: Any, ordered: bool) -> Any:
    if raw is None or raw == "":
        return ("obj", ()) if ordered else {}
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    try:
        if ordered:
            return json.loads(raw, object_pairs_hook=lambda kv: ("obj", tuple(kv)))
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return ("unparsed", raw)


def agree(a: dict, b: dict, level: str) -> bool:
    if a["kind"] == "error" or b["kind"] == "error":
        return a["kind"] == b["kind"] == "error" and a["error_class"] == b["error_class"]
    ta, tb = a["tool_calls"], b["tool_calls"]
    if len(ta) != len(tb):
        return False
    for (na, ra), (nb, rb) in zip(ta, tb):
        if na != nb:
            return False
        if level == "raw":
            if ra != rb:
                return False
        elif _parse(ra, ordered=(level != "action")) != _parse(rb, ordered=(level != "action")):
            return False
    if level == "action":
        return True
    if level == "full_ws":
        return a["content"].strip() == b["content"].strip() and a["reasoning"].strip() == b["reasoning"].strip()
    return a["content"] == b["content"] and a["reasoning"] == b["reasoning"]


def agree_all(a: dict, b: dict) -> dict:
    return {lv: agree(a, b, lv) for lv in LEVELS}


def first_char_divergence(a: dict, b: dict) -> int | None:
    """Character offset of the first difference between two serialized decisions."""
    sa = json.dumps([a.get("content"), a.get("reasoning"), a.get("tool_calls")], ensure_ascii=False)
    sb = json.dumps([b.get("content"), b.get("reasoning"), b.get("tool_calls")], ensure_ascii=False)
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x != y:
            return i
    return None if len(sa) == len(sb) else min(len(sa), len(sb))

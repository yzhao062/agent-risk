"""Verify a tool call round-trips through both local vLLM servers via Inspect."""

import asyncio

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import ToolDef, ToolParams
from inspect_ai.tool._tool_params import ToolParam


def _weather_tool() -> ToolDef:
    async def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny."

    params = ToolParams(
        properties={"city": ToolParam(type="string", description="City name")},
        required=["city"],
    )
    return ToolDef(
        get_weather, name="get_weather", description="Get the weather for a city.",
        parameters=params,
    )


async def check(tag: str, model_str: str, base_url: str) -> None:
    model = get_model(
        model_str,
        base_url=base_url,
        api_key="EMPTY",
        config=GenerateConfig(temperature=0.0, seed=0, max_tokens=256),
    )
    out = await model.generate(
        input="What is the weather in Paris? Use the get_weather tool.",
        tools=[_weather_tool()],
    )
    msg = out.message
    calls = msg.tool_calls or []
    print(f"[{tag}] model={model_str}")
    print(f"[{tag}] stop_reason={out.stop_reason} n_tool_calls={len(calls)}")
    for c in calls:
        print(f"[{tag}] call: {c.function}({c.arguments})")
    u = out.usage
    print(f"[{tag}] usage in/out={u.input_tokens}/{u.output_tokens}")
    assert calls, f"[{tag}] NO TOOL CALL RETURNED"
    print(f"[{tag}] TOOL CALL OK")


async def main() -> None:
    await check("agent", "vllm/Qwen/Qwen3-14B-FP8", "http://127.0.0.1:8000/v1")
    await check("monitor", "vllm/Qwen/Qwen3-4B-Instruct-2507", "http://127.0.0.1:8001/v1")


if __name__ == "__main__":
    asyncio.run(main())

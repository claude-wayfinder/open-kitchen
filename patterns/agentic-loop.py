"""
Agentic Loop with Explicit Termination
Extracted from OpenMythos (Build Small Hackathon 2026).

Pattern: Model calls tools iteratively until it calls final_message to stop.
Max-iteration safety valve prevents runaway agents.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


# ---------------------------------------------------------------------------
# 1. TOOL DEFINITIONS — The model's available actions
# ---------------------------------------------------------------------------

# The special tool that terminates the loop.
FINAL_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "final_message",
        "description": (
            "Call this when you have completed the task and are ready to "
            "deliver your final answer to the user. The content parameter "
            "is what the user will see."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Your final response to the user",
                },
            },
            "required": ["content"],
        },
    },
}


# ---------------------------------------------------------------------------
# 2. LOOP STATE — Track what's happening across iterations
# ---------------------------------------------------------------------------

@dataclass
class LoopState:
    """Tracks the state of an agentic loop run."""
    messages: list[dict] = field(default_factory=list)
    iterations: int = 0
    tool_calls_made: list[dict] = field(default_factory=list)
    final_output: str | None = None
    terminated_by: str = ""  # "agent", "max_iterations", "error"
    total_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# 3. THE AGENTIC LOOP — Where the magic happens
# ---------------------------------------------------------------------------

async def run_agent(
    api_url: str, api_key: str, system_prompt: str, user_message: str,
    tools: list[dict], tool_handlers: dict[str, Callable],
    model: str = "default", max_iterations: int = 10,
    max_tokens: int = 4096, temperature: float = 0.7,
) -> LoopState:
    """Run an agentic loop until final_message is called or max iterations hit."""
    state = LoopState()
    start_time = time.monotonic()

    # Ensure final_message is always available
    all_tools = tools + [FINAL_MESSAGE_TOOL]

    # Initialize conversation
    state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    async with httpx.AsyncClient(timeout=120) as client:
        while state.iterations < max_iterations:
            state.iterations += 1

            # --- Call the model ---
            response = await client.post(
                f"{api_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": state.messages,
                    "tools": all_tools,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            message = choice["message"]

            # Add assistant message to history
            state.messages.append(message)

            # --- Check for tool calls ---
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # Model responded with plain text (no tool call).
                # Treat as implicit final message.
                state.final_output = message.get("content", "")
                state.terminated_by = "agent"
                break

            # --- Execute each tool call ---
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args_raw = tc["function"].get("arguments", "{}")
                fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw

                # Check for termination
                if fn_name == "final_message":
                    state.final_output = fn_args.get("content", "")
                    state.terminated_by = "agent"
                    state.total_time_ms = (time.monotonic() - start_time) * 1000
                    return state

                # Execute the tool
                state.tool_calls_made.append({
                    "name": fn_name,
                    "arguments": fn_args,
                    "iteration": state.iterations,
                })

                handler = tool_handlers.get(fn_name)
                if handler:
                    try:
                        result = handler(**fn_args)
                        result_str = json.dumps(result) if not isinstance(result, str) else result
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})

                # Feed the tool result back to the model
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

    # --- Max iterations reached ---
    if state.final_output is None:
        state.terminated_by = "max_iterations"
        # Try to extract something useful from the last message
        last = state.messages[-1]
        state.final_output = last.get("content", "Agent did not produce a final answer.")

    state.total_time_ms = (time.monotonic() - start_time) * 1000
    return state


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    # Define some tools the agent can use
    example_tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        },
    ]

    def lookup_weather(city: str) -> dict:
        """Fake weather lookup for demo purposes."""
        return {"city": city, "temp_f": 72, "conditions": "sunny"}

    async def demo():
        # In real usage:
        # state = await run_agent(
        #     api_url="https://your-api.com",
        #     api_key="your-key",
        #     system_prompt="You are a helpful weather assistant.",
        #     user_message="What's the weather in Portland?",
        #     tools=example_tools,
        #     tool_handlers={"lookup_weather": lookup_weather},
        #     max_iterations=5,
        # )
        # print(f"Final answer: {state.final_output}")
        # print(f"Terminated by: {state.terminated_by}")
        # print(f"Iterations: {state.iterations}")
        # print(f"Tool calls: {len(state.tool_calls_made)}")
        print("Agentic loop pattern ready. Wire up an API to run it.")

    asyncio.run(demo())

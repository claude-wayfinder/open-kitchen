"""
Single Structured JSON Call Pattern
Extracted from Small Talk + Claim-Ready (Build Small Hackathon 2026).

Pattern: One system prompt enforcing JSON output, response_format: {"type": "json_object"},
and a bulletproof parse function that strips think tags and extracts the first {...} block.

WHY THIS EXISTS:
  Chaining multiple LLM calls is slow and fragile. Each hop can hallucinate,
  drop fields, or drift from the schema. One constrained JSON call with a
  well-defined schema returns everything in a single round trip.

USAGE:
  1. Define your JSON schema in the system prompt (plain English is fine)
  2. Set response_format={"type": "json_object"} to enforce JSON output
  3. Use parse_json_from_llm() to safely extract JSON from the response
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# 1. SYSTEM PROMPT — Tell the model what shape you want
# ---------------------------------------------------------------------------
# The /no_think prefix skips chain-of-thought on models that support it.
# "Respond ONLY with valid JSON" is the most reliable instruction across models.

SYSTEM_PROMPT = (
    "/no_think\n"
    "You are an expert analyst. Respond ONLY with valid JSON, no prose, "
    "no markdown fences, no explanation."
)


# ---------------------------------------------------------------------------
# 2. PARSE FUNCTION — The real hero
# ---------------------------------------------------------------------------
# LLMs love to wrap JSON in ```json fences, <think> tags, or preamble text.
# This strips all of that and extracts the first valid JSON object.

def parse_json_from_llm(content: str) -> dict:
    """
    Extract a JSON object from LLM output, handling common wrapping patterns.

    Handles:
      - <think>...</think> tags (DeepSeek, some fine-tuned models)
      - ```json ... ``` markdown fences
      - Leading/trailing prose around the JSON
      - Nested objects (finds the outermost {...} block)
    """
    # Strip think tags (some models emit these even when told not to)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)

    # Strip markdown fences
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"```\s*$", "", content)

    # Find the first { ... } block (greedy — gets the full outermost object)
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {content[:200]!r}")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}") from e


# ---------------------------------------------------------------------------
# 3. SINGLE-CALL FUNCTION — One prompt, one schema, one response
# ---------------------------------------------------------------------------

async def structured_call(
    api_url: str,
    api_key: str,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Make a single LLM call that returns structured JSON.

    Args:
        api_url:       Base URL of the OpenAI-compatible API
        api_key:       Bearer token
        user_prompt:   The prompt describing what you want (include schema here)
        system_prompt: Tells the model to output JSON only
        model:         Model name (ignored by some endpoints)
        max_tokens:    Cap on response length
        temperature:   0.0 = deterministic, 1.0 = creative

    Returns:
        Parsed dict from the model's JSON response
    """
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{api_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                # This is the key — forces the model to emit valid JSON
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]
    return parse_json_from_llm(raw)


# ---------------------------------------------------------------------------
# 4. PROMPT BUILDER — Encode your schema in plain English
# ---------------------------------------------------------------------------
# Tip: describe the JSON shape inline in the prompt. The model follows it
# more reliably than separate schema definitions.

def build_analysis_prompt(text: str) -> str:
    """Example: analyze a piece of text and return structured results."""
    return (
        f'Analyze the following text and return a JSON object:\n\n'
        f'Text: "{text}"\n\n'
        f'Return this exact shape:\n'
        f'{{\n'
        f'  "summary": "one-sentence summary",\n'
        f'  "sentiment": "positive" | "negative" | "neutral",\n'
        f'  "confidence": 0.0 to 1.0,\n'
        f'  "key_topics": ["topic1", "topic2"],\n'
        f'  "actionable": true | false\n'
        f'}}'
    )


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def demo():
        # Test the parser with messy LLM output
        messy_outputs = [
            '{"clean": true}',
            '<think>Let me think...</think>\n{"stripped": true}',
            'Here is the result:\n```json\n{"fenced": true}\n```',
            'Sure! {"buried": true, "nested": {"deep": 1}}',
        ]
        for raw in messy_outputs:
            parsed = parse_json_from_llm(raw)
            print(f"  Parsed: {parsed}")

        # To make a real API call:
        # result = await structured_call(
        #     api_url="https://your-endpoint.com",
        #     api_key="your-key",
        #     user_prompt=build_analysis_prompt("The product works great!"),
        # )
        # print(result)

    asyncio.run(demo())

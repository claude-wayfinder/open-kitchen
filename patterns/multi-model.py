"""
Heterogeneous Multi-Model Agents
Extracted from Thousand Token Wood (Build Small Hackathon 2026).

Pattern: Different LLM providers as different agents/factions in the same world.
Per-engine batching groups same-provider entities into one API call. Different
models = different personalities = emergent behavior for free.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. ENGINE CONFIG — One per LLM provider/model combo
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Configuration for one LLM provider."""
    name: str               # e.g. "anthropic-claude", "openai-gpt4", "local-llama"
    api_url: str            # Base URL for the API
    api_key: str            # Bearer token
    model: str              # Model identifier
    max_tokens: int = 1024
    temperature: float = 0.7
    requests_per_minute: int = 60  # For rate limiting


# ---------------------------------------------------------------------------
# 2. ENTITY-ENGINE MAPPING — Who thinks with what
# ---------------------------------------------------------------------------

@dataclass
class AgentEntity:
    """An entity with an assigned model engine."""
    id: str
    name: str
    engine_name: str        # Which EngineConfig to use
    faction: str = ""       # Optional grouping (e.g. "elves", "merchants")
    state: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {"id": self.id, "name": self.name, "faction": self.faction,
                "state": dict(self.state)}


# ---------------------------------------------------------------------------
# 3. PER-ENGINE BATCHING — Group same-provider entities, one call each
# ---------------------------------------------------------------------------

def group_by_engine(entities: list[AgentEntity]) -> dict[str, list[AgentEntity]]:
    """Group entities by their engine name for batched API calls."""
    groups: dict[str, list[AgentEntity]] = {}
    for entity in entities:
        groups.setdefault(entity.engine_name, []).append(entity)
    return groups


def build_batch_prompt(entities: list[AgentEntity],
                       prompt_builder: Callable[[AgentEntity], str]) -> str:
    """Batch N entities into one prompt. Model returns JSON keyed by entity ID."""
    entity_sections = []
    for entity in entities:
        individual_prompt = prompt_builder(entity)
        entity_sections.append(
            f"--- Entity: {entity.id} ({entity.name}) ---\n{individual_prompt}"
        )

    return (
        "You are managing multiple characters. For EACH entity below, "
        "choose an action.\n\n"
        + "\n\n".join(entity_sections) +
        "\n\nRespond with a JSON object where keys are entity IDs:\n"
        '{"entity_id": {"action": "...", "reason": "..."}, ...}'
    )


def parse_batch_response(raw: str, entity_ids: list[str]) -> dict[str, dict]:
    """Parse a batched response into per-entity decisions."""
    import re
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {eid: {} for eid in entity_ids}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {eid: {} for eid in entity_ids}

    # Ensure every entity has an entry
    for eid in entity_ids:
        if eid not in data:
            data[eid] = {"action": "wait", "reason": "no response from model"}

    return data


# ---------------------------------------------------------------------------
# 4. MULTI-ENGINE TICK — Run all engines in parallel
# ---------------------------------------------------------------------------

def call_engine(
    engine: EngineConfig,
    prompt: str,
) -> str:
    """Make one API call to an engine. Returns raw response text."""
    import httpx
    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{engine.api_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {engine.api_key}"},
            json={
                "model": engine.model,
                "messages": [
                    {"role": "system", "content": "Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": engine.max_tokens,
                "temperature": engine.temperature,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def run_multi_model_tick(
    entities: list[AgentEntity],
    engines: dict[str, EngineConfig],
    prompt_builder: Callable[[AgentEntity], str],
    max_workers: int = 4,
) -> dict[str, dict]:
    """Run one tick across all entities, batched by engine, parallelized."""
    groups = group_by_engine(entities)
    all_decisions: dict[str, dict] = {}

    def process_group(engine_name: str,
                      group: list[AgentEntity]) -> dict[str, dict]:
        engine = engines.get(engine_name)
        if not engine:
            return {e.id: {"action": "wait", "reason": "no engine"} for e in group}

        # Batch if multiple entities share the same engine
        if len(group) > 1:
            prompt = build_batch_prompt(group, prompt_builder)
            raw = call_engine(engine, prompt)
            return parse_batch_response(raw, [e.id for e in group])
        else:
            # Single entity — direct call, simpler parsing
            prompt = prompt_builder(group[0])
            raw = call_engine(engine, prompt)
            import re
            match = re.search(r"\{.*\}", raw, re.S)
            decision = json.loads(match.group(0)) if match else {}
            return {group[0].id: decision}

    # Run all engine groups in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_group, name, group): name
            for name, group in groups.items()
        }
        for future in as_completed(futures):
            group_decisions = future.result()
            all_decisions.update(group_decisions)

    return all_decisions


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Define engines
    engines = {
        "claude": EngineConfig(
            name="claude",
            api_url="https://api.anthropic.com",
            api_key="sk-...",
            model="claude-sonnet-4-20250514",
            temperature=0.5,
        ),
        "local": EngineConfig(
            name="local",
            api_url="http://localhost:8080",
            api_key="none",
            model="llama-3",
            temperature=0.9,
        ),
    }

    # Create entities with different engines
    entities = [
        AgentEntity("elf1", "Aerin", engine_name="claude", faction="elves",
                     state={"hp": 80, "gold": 50}),
        AgentEntity("elf2", "Lyra", engine_name="claude", faction="elves",
                     state={"hp": 60, "gold": 30}),
        AgentEntity("orc1", "Grukk", engine_name="local", faction="orcs",
                     state={"hp": 100, "gold": 5}),
    ]

    # Show grouping
    groups = group_by_engine(entities)
    for engine_name, group in groups.items():
        names = [e.name for e in group]
        print(f"  Engine '{engine_name}': {names}")
        # claude gets one batched call for Aerin + Lyra
        # local gets one call for Grukk

    # The prompt builder each entity sees
    def my_prompt(entity: AgentEntity) -> str:
        return (f"You are {entity.name} ({entity.faction}). "
                f"Stats: {entity.state}. "
                f"Choose: explore, rest, or trade.")

    # Show what the batched prompt looks like
    claude_group = groups["claude"]
    batch = build_batch_prompt(claude_group, my_prompt)
    print(f"\nBatched prompt for claude engine:\n{batch}")

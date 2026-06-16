"""
Reason-First (Forced Reasoning for Small Models)
Source: Tiny Browser Planner (Build Small Hackathon 2026)

Pattern: Make small models explain their reasoning BEFORE choosing an
action. Structure the output so reasoning comes first, action comes second.
The model can't pick an action without first articulating why.

The key insight from the source: fewer, harder training examples with
explicit reasoning chains beat bulk data without reasoning. A 1B model
trained on 500 reason-then-act examples outperformed one trained on
5000 action-only examples.

Use this for: browser agents, tool-use agents, any small model that
needs to make decisions. The reasoning step acts as a natural chain-of-thought
that improves accuracy without increasing model size.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ReasonedAction:
    """Output of the reason-first pattern: reasoning + action."""
    reasoning: str          # WHY the model chose this action
    action: str             # WHAT action to take
    parameters: dict        # HOW to execute (action-specific params)
    confidence: float       # Model's self-assessed confidence
    latency_ms: float = 0.0


@dataclass
class TrainingExample:
    """
    One training example in the reason-first format.
    These are harder to write but worth more per example.
    """
    observation: str        # What the model sees
    reasoning: str          # Step-by-step thinking (the gold)
    action: str             # Chosen action
    parameters: dict        # Action parameters
    difficulty: str = "medium"  # easy/medium/hard -- bias toward harder


# =============================================================
# Prompt Template
# =============================================================
# The structure forces reasoning before action selection.

REASON_FIRST_PROMPT = """You are an agent that reasons before acting.

Available actions: {actions}

Current observation:
{observation}

You MUST respond in this exact JSON format:
{{
  "reasoning": "Step by step, explain what you observe, what your options are, and why you're choosing this action. Be specific.",
  "action": "one of the available actions",
  "parameters": {{}},
  "confidence": 0.0 to 1.0
}}

Think carefully. Reasoning comes first. Action follows from reasoning."""


# =============================================================
# Action Registry
# =============================================================

@dataclass
class ActionSpec:
    """Definition of one available action."""
    name: str
    description: str
    param_schema: dict  # Expected parameters


class ActionRegistry:
    """Registry of available actions the model can choose from."""

    def __init__(self):
        self.actions: dict[str, ActionSpec] = {}

    def register(self, name: str, description: str, param_schema: Optional[dict] = None):
        self.actions[name] = ActionSpec(
            name=name,
            description=description,
            param_schema=param_schema or {},
        )

    def describe(self) -> str:
        """Format action list for the prompt."""
        lines = []
        for a in self.actions.values():
            params = json.dumps(a.param_schema) if a.param_schema else "none"
            lines.append(f"- {a.name}: {a.description} (params: {params})")
        return "\n".join(lines)

    def validate(self, action: str, parameters: dict) -> tuple[bool, str]:
        """Check if an action choice is valid."""
        if action not in self.actions:
            return False, f"Unknown action: {action}. Available: {list(self.actions.keys())}"
        return True, "valid"


# =============================================================
# Reason-First Agent
# =============================================================

class ReasonFirstAgent:
    """
    Agent that always reasons before acting.
    Wraps any model (local or API) with the reason-first prompt structure.
    """

    def __init__(
        self,
        model_fn: Optional[Callable[[str], str]] = None,
        max_retries: int = 2,
    ):
        self.model_fn = model_fn or self._mock_model
        self.registry = ActionRegistry()
        self.max_retries = max_retries
        self.history: list[ReasonedAction] = []

    def _mock_model(self, prompt: str) -> str:
        """Stub model. Replace with your actual LLM call."""
        return json.dumps({
            "reasoning": "I see a search box on the page. The user wants to find products. The most logical action is to click the search box and type the query.",
            "action": "click",
            "parameters": {"selector": "#search-box"},
            "confidence": 0.85,
        })

    def act(self, observation: str) -> ReasonedAction:
        """
        Given an observation, reason about it and choose an action.
        Retries if the model output is malformed.
        """
        prompt = REASON_FIRST_PROMPT.format(
            actions=self.registry.describe(),
            observation=observation,
        )

        for attempt in range(self.max_retries + 1):
            start = time.time()
            raw = self.model_fn(prompt)
            latency = (time.time() - start) * 1000

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                if attempt < self.max_retries:
                    continue
                # Last attempt: return a safe fallback
                return ReasonedAction(
                    reasoning="Failed to parse model output",
                    action="wait",
                    parameters={},
                    confidence=0.0,
                    latency_ms=latency,
                )

            # Validate the action
            action = data.get("action", "unknown")
            valid, reason = self.registry.validate(action, data.get("parameters", {}))

            if not valid and attempt < self.max_retries:
                # Add error context and retry
                prompt += f"\n\nPrevious attempt was invalid: {reason}. Try again."
                continue

            result = ReasonedAction(
                reasoning=data.get("reasoning", ""),
                action=action,
                parameters=data.get("parameters", {}),
                confidence=data.get("confidence", 0.5),
                latency_ms=round(latency, 2),
            )
            self.history.append(result)
            return result

        # Should not reach here, but just in case
        return ReasonedAction(
            reasoning="Exhausted retries",
            action="wait",
            parameters={},
            confidence=0.0,
        )


# =============================================================
# Training Data Generator
# =============================================================
# Helper for creating reason-first training examples.

class TrainingDataBuilder:
    """
    Build training datasets in the reason-first format.
    Bias toward harder examples -- they teach more per example.
    """

    def __init__(self):
        self.examples: list[TrainingExample] = []

    def add(
        self,
        observation: str,
        reasoning: str,
        action: str,
        parameters: Optional[dict] = None,
        difficulty: str = "medium",
    ):
        self.examples.append(TrainingExample(
            observation=observation,
            reasoning=reasoning,
            action=action,
            parameters=parameters or {},
            difficulty=difficulty,
        ))

    def to_jsonl(self) -> str:
        """Export as JSONL for fine-tuning."""
        lines = []
        for ex in self.examples:
            lines.append(json.dumps({
                "observation": ex.observation,
                "reasoning": ex.reasoning,
                "action": ex.action,
                "parameters": ex.parameters,
                "difficulty": ex.difficulty,
            }))
        return "\n".join(lines)

    def stats(self) -> dict:
        difficulties = {}
        for ex in self.examples:
            difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1
        return {"total": len(self.examples), "by_difficulty": difficulties}


# --- Demo ---
if __name__ == "__main__":
    # Set up an agent for browser automation
    agent = ReasonFirstAgent()
    agent.registry.register("click", "Click an element", {"selector": "CSS selector"})
    agent.registry.register("type", "Type text into a field", {"selector": "CSS selector", "text": "string"})
    agent.registry.register("scroll", "Scroll the page", {"direction": "up|down"})
    agent.registry.register("wait", "Wait and observe", {})

    # Agent reasons and acts
    result = agent.act("I see a login page with email and password fields. The email field is empty.")
    print(f"Reasoning: {result.reasoning}")
    print(f"Action: {result.action}")
    print(f"Params: {result.parameters}")
    print(f"Confidence: {result.confidence}")
    print(f"Latency: {result.latency_ms}ms")

    # Build training data
    builder = TrainingDataBuilder()
    builder.add(
        observation="Search results page showing 10 products. 'Next page' button visible at bottom.",
        reasoning="I see search results but need to check if the target product is here. I should scan the visible results first. None match the target. I need to go to the next page.",
        action="click",
        parameters={"selector": "button.next-page"},
        difficulty="hard",
    )
    print(f"\nTraining stats: {builder.stats()}")

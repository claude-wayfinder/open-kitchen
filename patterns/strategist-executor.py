"""
Strategist-Executor (Big/Small Model Split)
Source: Duel of Nemotron (Build Small Hackathon 2026)

Pattern: A large model sets strategy infrequently (seconds-to-minutes cadence).
A tiny model executes in real-time, conditioned on the big model's strategic
context. The small model doesn't decide WHAT to do -- it decides HOW to do
what the strategist already decided.

The key insight: you don't need a 70B model on every token. You need it
once to set the frame, then a 1-3B model can execute within that frame
at 100x the speed and 1/50th the cost.

Use this for: real-time agents, game AI, trading bots, live assistants,
robotics controllers -- anywhere latency matters but quality can't drop.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Strategy:
    """
    Output of the big model. Sets the frame for the executor.
    Think of this as the 'orders' -- high-level intent, constraints,
    priorities, and any precomputed context the small model needs.
    """
    intent: str               # What we're trying to accomplish
    constraints: list[str]    # Hard rules the executor must follow
    priorities: list[str]     # Soft preferences, ordered by importance
    context: dict[str, Any]   # Precomputed data the executor can reference
    valid_until: float        # Timestamp -- strategy expires, forces re-plan
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() > self.valid_until

    def to_prompt_prefix(self) -> str:
        """
        Serialize strategy into a prompt prefix for the small model.
        This is how you 'condition' the executor on the strategist's output.
        """
        return (
            f"CURRENT STRATEGY (set by planner):\n"
            f"Intent: {self.intent}\n"
            f"Constraints: {json.dumps(self.constraints)}\n"
            f"Priorities: {json.dumps(self.priorities)}\n"
            f"Context: {json.dumps(self.context)}\n"
            f"---\n"
            f"Execute within this strategy. Do not deviate from constraints.\n"
        )


@dataclass
class Execution:
    """Output of the small model. A concrete action within the strategy."""
    action: str
    reasoning: str
    confidence: float
    latency_ms: float
    strategy_age_ms: float  # How old the strategy was when this executed


# =============================================================
# Strategist: Big model, runs infrequently
# =============================================================

class Strategist:
    """
    Wraps your big model. Called infrequently to set/update strategy.
    Replace plan() internals with your actual LLM call.
    """

    def __init__(
        self,
        model_fn: Optional[Callable[[str], str]] = None,
        strategy_ttl: float = 30.0,  # Seconds before strategy expires
    ):
        # model_fn: your big model call. Input: prompt string. Output: response string.
        self.model_fn = model_fn or self._mock_model
        self.strategy_ttl = strategy_ttl
        self.current: Optional[Strategy] = None
        self.plan_count = 0

    def _mock_model(self, prompt: str) -> str:
        """Stub -- replace with your actual big model call."""
        return json.dumps({
            "intent": "Assist the user with their current request",
            "constraints": ["Be truthful", "Stay on topic", "No harmful content"],
            "priorities": ["Clarity", "Brevity", "Helpfulness"],
            "context": {"domain": "general", "tone": "professional"},
        })

    def plan(self, situation: str) -> Strategy:
        """
        Ask the big model to generate a strategy for the current situation.
        This is the expensive call -- runs once per strategy cycle.
        """
        prompt = (
            f"You are a strategic planner. Given the current situation, "
            f"produce a strategy as JSON with keys: intent, constraints, "
            f"priorities, context.\n\n"
            f"Situation: {situation}"
        )

        raw = self.model_fn(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback strategy if model output is unparseable
            data = {
                "intent": "Continue with default behavior",
                "constraints": ["Be safe"],
                "priorities": ["Don't break anything"],
                "context": {},
            }

        self.current = Strategy(
            intent=data.get("intent", ""),
            constraints=data.get("constraints", []),
            priorities=data.get("priorities", []),
            context=data.get("context", {}),
            valid_until=time.time() + self.strategy_ttl,
        )
        self.plan_count += 1
        return self.current

    def needs_replan(self) -> bool:
        """Check if current strategy is missing or expired."""
        return self.current is None or self.current.is_expired()


# =============================================================
# Executor: Tiny model, runs on every tick
# =============================================================

class Executor:
    """
    Wraps your small model. Called on every input/tick.
    Conditioned on the strategist's current strategy via prompt prefix.
    """

    def __init__(self, model_fn: Optional[Callable[[str], str]] = None):
        self.model_fn = model_fn or self._mock_model
        self.execution_count = 0

    def _mock_model(self, prompt: str) -> str:
        """Stub -- replace with your actual small model call."""
        return json.dumps({
            "action": "respond_helpfully",
            "reasoning": "User asked a question within strategy bounds",
            "confidence": 0.9,
        })

    def execute(self, strategy: Strategy, input_data: str) -> Execution:
        """
        Execute one action within the current strategy.
        The strategy's prompt prefix conditions the small model's behavior.
        """
        start = time.time()

        prompt = (
            f"{strategy.to_prompt_prefix()}\n"
            f"Input: {input_data}\n"
            f"Choose an action and explain briefly."
        )

        raw = self.model_fn(prompt)
        latency = (time.time() - start) * 1000

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"action": "fallback", "reasoning": raw, "confidence": 0.5}

        self.execution_count += 1

        return Execution(
            action=data.get("action", "unknown"),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
            latency_ms=round(latency, 2),
            strategy_age_ms=round((time.time() - strategy.created_at) * 1000, 2),
        )


# =============================================================
# Pipeline: Wire strategist and executor together
# =============================================================

class StrategistExecutorPipeline:
    """
    The full pattern. Strategist replans when needed,
    executor runs on every input conditioned on current strategy.
    """

    def __init__(self, strategist: Strategist, executor: Executor):
        self.strategist = strategist
        self.executor = executor

    def step(self, situation: str, input_data: str) -> Execution:
        """
        One tick of the pipeline.
        Replans if strategy is stale, then executes.
        """
        if self.strategist.needs_replan():
            self.strategist.plan(situation)

        return self.executor.execute(self.strategist.current, input_data)


# --- Demo ---
if __name__ == "__main__":
    pipeline = StrategistExecutorPipeline(
        strategist=Strategist(strategy_ttl=10.0),  # Replan every 10s
        executor=Executor(),
    )

    inputs = [
        "What's the weather like?",
        "Can you help me write an email?",
        "Tell me a joke",
    ]

    for inp in inputs:
        result = pipeline.step(situation="User is chatting casually", input_data=inp)
        print(f"Input: {inp}")
        print(f"  Action: {result.action}")
        print(f"  Latency: {result.latency_ms}ms")
        print(f"  Strategy age: {result.strategy_age_ms}ms")
        print()

    print(f"Plans made: {pipeline.strategist.plan_count}")
    print(f"Executions: {pipeline.executor.execution_count}")

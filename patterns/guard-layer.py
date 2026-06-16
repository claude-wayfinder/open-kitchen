"""
Deterministic Guard Layer Pattern
Extracted from Jawbreaker + Her (Build Small Hackathon 2026).

Pattern: Model proposes, code verifies. Never serve raw model output. Three-tier
graceful degradation: GPU model -> CPU fallback -> deterministic heuristics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. VALIDATION — Define what "valid output" means for your domain
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """Result of a guard check."""
    valid: bool
    value: Any = None
    error: str = ""
    tier: str = ""  # which tier produced the value: "gpu", "cpu", "heuristic"
    latency_ms: float = 0.0


def validate_output(output: Any, schema: dict) -> GuardResult:
    """
    Check model output against your domain rules.

    This is where you encode the hard constraints your app requires.
    The model suggests; this function decides.

    Args:
        output: The parsed model response
        schema: Dict of field names -> validation functions

    Example schema:
        {
            "score": lambda v: isinstance(v, (int, float)) and 0 <= v <= 100,
            "label": lambda v: v in ("positive", "negative", "neutral"),
            "explanation": lambda v: isinstance(v, str) and len(v) > 10,
        }
    """
    if not isinstance(output, dict):
        return GuardResult(valid=False, error="Output is not a dict")

    for field_name, check_fn in schema.items():
        value = output.get(field_name)
        if value is None:
            return GuardResult(valid=False, error=f"Missing field: {field_name}")
        if not check_fn(value):
            return GuardResult(
                valid=False,
                error=f"Field '{field_name}' failed validation: {value!r}"
            )

    return GuardResult(valid=True, value=output)


# ---------------------------------------------------------------------------
# 2. THREE-TIER DEGRADATION — Try the best option, fall back gracefully
# ---------------------------------------------------------------------------

@dataclass
class TieredGenerator:
    """
    Wraps three generation strategies with automatic fallback.

    Usage:
        gen = TieredGenerator(
            gpu_fn=call_gpu_model,
            cpu_fn=call_cpu_model,
            heuristic_fn=rule_based_fallback,
            validator=my_validation_schema,
        )
        result = gen.generate(prompt)
    """
    gpu_fn: Callable | None = None       # Tier 1: GPU model call
    cpu_fn: Callable | None = None       # Tier 2: CPU model call
    heuristic_fn: Callable | None = None # Tier 3: Deterministic fallback
    validator: dict = field(default_factory=dict)  # Field -> check function
    max_retries: int = 2                 # Retries per tier before falling back

    def generate(self, prompt: str, context: dict | None = None) -> GuardResult:
        """
        Try each tier in order. Return the first valid result.
        If all tiers fail, return the best error we have.
        """
        tiers = [
            ("gpu", self.gpu_fn),
            ("cpu", self.cpu_fn),
            ("heuristic", self.heuristic_fn),
        ]

        last_error = "All tiers exhausted"

        for tier_name, tier_fn in tiers:
            if tier_fn is None:
                continue

            for attempt in range(self.max_retries):
                start = time.monotonic()
                try:
                    raw = tier_fn(prompt, context or {})
                    elapsed = (time.monotonic() - start) * 1000

                    # Validate the output
                    result = validate_output(raw, self.validator)
                    result.tier = tier_name
                    result.latency_ms = elapsed

                    if result.valid:
                        return result

                    last_error = f"[{tier_name}] {result.error}"
                    print(f"  Guard rejected {tier_name} attempt {attempt + 1}: "
                          f"{result.error}")

                except Exception as e:
                    elapsed = (time.monotonic() - start) * 1000
                    last_error = f"[{tier_name}] Exception: {e}"
                    print(f"  {tier_name} failed attempt {attempt + 1}: {e}")

        return GuardResult(valid=False, error=last_error)


# ---------------------------------------------------------------------------
# 3. GUARD DECORATOR — Wrap any endpoint with validation
# ---------------------------------------------------------------------------

def guarded(validator: dict, fallback_value: Any = None):
    """
    Decorator that validates a function's return value against a schema.
    If validation fails, returns the fallback value instead of bad data.

    Usage:
        @guarded({"score": lambda v: 0 <= v <= 100}, fallback_value={"score": 50})
        def get_score(text):
            return call_model(text)
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                check = validate_output(result, validator)
                if check.valid:
                    return result
                print(f"Guard blocked output from {fn.__name__}: {check.error}")
            except Exception as e:
                print(f"Guard caught exception in {fn.__name__}: {e}")

            if fallback_value is not None:
                return fallback_value
            raise ValueError(f"Guard: {fn.__name__} produced invalid output "
                             f"and no fallback is configured")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Define your validation schema ---
    sentiment_schema = {
        "label": lambda v: v in ("positive", "negative", "neutral"),
        "confidence": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
        "explanation": lambda v: isinstance(v, str) and len(v) > 5,
    }

    # --- Define your three tiers ---
    def gpu_model(prompt: str, ctx: dict) -> dict:
        """Simulates a GPU model call (might fail under load)."""
        return {
            "label": "positive",
            "confidence": 0.92,
            "explanation": "The text expresses satisfaction with the outcome.",
        }

    def cpu_model(prompt: str, ctx: dict) -> dict:
        """Simulates a smaller CPU model (always available, less accurate)."""
        return {
            "label": "neutral",
            "confidence": 0.6,
            "explanation": "Unable to determine strong sentiment.",
        }

    def heuristic_fallback(prompt: str, ctx: dict) -> dict:
        """Rule-based: count positive/negative keywords. Always works."""
        pos = sum(1 for w in ["good", "great", "happy", "love", "excellent"]
                  if w in prompt.lower())
        neg = sum(1 for w in ["bad", "terrible", "hate", "awful", "worst"]
                  if w in prompt.lower())
        label = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        return {
            "label": label,
            "confidence": 0.5,
            "explanation": f"Keyword heuristic: {pos} positive, {neg} negative words.",
        }

    # --- Wire it up ---
    gen = TieredGenerator(
        gpu_fn=gpu_model,
        cpu_fn=cpu_model,
        heuristic_fn=heuristic_fallback,
        validator=sentiment_schema,
    )

    result = gen.generate("This product is absolutely great, I love it!")
    print(f"Tier: {result.tier}")
    print(f"Value: {result.value}")
    print(f"Latency: {result.latency_ms:.1f}ms")

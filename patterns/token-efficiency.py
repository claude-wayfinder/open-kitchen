"""
token-efficiency.py
Patterns for reducing token and API usage, extracted from Build Small Hackathon entries.

Sources:
- ContextForge: staged pipeline with bounded token budgets per stage
- Split-Brain Copilot: local draft + cloud verify split
- code-shrink-token-decimator: AST-based pre-send compression
- LocalDuo: forced stop on runaway thinking chains
- OUROBOROS Kernel Mint: RL on verified wins; small model + referee loop
- whatfirst-small: 3B offline model + deterministic scoring engine
"""

import ast
import json
import re
from typing import Any


# ── Technique 1: Staged pipeline with per-stage token budgets ────────────────
# Source: ContextForge
# Decompose one hard task into N small calls. Give each stage a ceiling.
# If a stage fails, use a deterministic fallback and continue.
# Total spend = sum of small ceilings, not one unbounded call.

STAGE_BUDGETS = {
    "intake":       200,
    "topology":     100,
    "vital_few":    300,
    "reasoning":    150,
    "prompt_pack":  500,
    "qa":           200,
    "assembly":     350,
}

def run_staged_pipeline(prompt: str, model_fn, stages: dict = STAGE_BUDGETS) -> dict:
    """
    Run a multi-stage pipeline where each stage has a capped token budget.
    Falls back to a deterministic result if the model call fails or exceeds budget.
    """
    results = {}
    for stage, max_tokens in stages.items():
        try:
            results[stage] = model_fn(prompt, stage=stage, max_new_tokens=max_tokens)
        except Exception:
            results[stage] = _deterministic_fallback(stage, prompt)
    return results

def _deterministic_fallback(stage: str, prompt: str) -> str:
    return f"[{stage}:fallback] {prompt[:80]}"


# ── Technique 2: Local draft + cloud verify (split-brain) ───────────────────
# Source: Split-Brain Copilot
# Fast small model generates locally (0 API cost). Cloud model only verifies.
# Only pay for cloud tokens when the local draft needs correction.

def split_brain(prompt: str, local_fn, cloud_verify_fn) -> dict:
    """
    Generate with a small local model first; verify with a larger cloud model.
    Cloud is called only once per draft, not for the full generation.
    Returns both the draft and the verified result.
    """
    draft = local_fn(prompt)
    verdict = cloud_verify_fn(draft)          # PASS | FIX | REWRITE
    if verdict["status"] == "PASS":
        return {"result": draft, "cloud_tokens_used": verdict["tokens"]}
    corrected = verdict.get("corrected", draft)
    return {"result": corrected, "cloud_tokens_used": verdict["tokens"]}


# ── Technique 3: Pre-send compression (AST stripping) ───────────────────────
# Source: code-shrink-token-decimator
# Strip docstrings, comments, blank lines before the string hits the model.
# Achieves up to 66% token reduction in <10ms with zero model calls.

def compress_python_for_prompt(source: str) -> str:
    """
    Strip comments, docstrings, and blank lines from Python source before
    sending it as context. Keeps semantic structure intact.
    """
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (ast.get_docstring(node) and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                    node.body.pop(0)
        # Remove comment lines and collapse blank lines
        lines = source.splitlines()
        cleaned = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        return "\n".join(cleaned)
    except SyntaxError:
        # Non-Python: just strip comments and blank lines with regex
        return _compress_text(source)

def compress_json_for_prompt(obj: Any) -> str:
    """Minify JSON before including it in a prompt."""
    return json.dumps(obj, separators=(",", ":"))

def _compress_text(text: str) -> str:
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


# ── Technique 4: Kill runaway thinking chains ────────────────────────────────
# Source: LocalDuo (Qwen3.5-9B think/non-think mode)
# Thinking models burn tokens before producing output.
# Set a char threshold; if <think> block exceeds it, force-close and restart.

AUTO_FORCE_CHARS = 4000   # chars of <think> before forcing output
HARD_LIMIT_CHARS = 10000  # absolute ceiling before hard kill

def stream_with_think_kill(streamer, force_fn, hard_limit: int = HARD_LIMIT_CHARS,
                           auto_force: int = AUTO_FORCE_CHARS) -> str:
    """
    Consume a streaming model output. If the <think> block grows past
    auto_force chars, call force_fn() to close it and redirect to JSON output.
    force_fn should append '</think>\\n```json\\n' to the context and restart.
    """
    output = []
    in_think = False
    think_chars = 0
    total_chars = 0

    for token in streamer:
        output.append(token)
        total_chars += len(token)
        if "<think>" in token:
            in_think = True
        if "</think>" in token:
            in_think = False
            think_chars = 0
        if in_think:
            think_chars += len(token)

        if think_chars > auto_force or total_chars > hard_limit:
            force_fn()  # kill + restart with forced JSON prefix
            break

    return "".join(output)

def non_think_prefix() -> str:
    """Append to chat template to skip thinking entirely (Qwen3 compatible)."""
    return "<think>\n\n</think>\n```json\n"


# ── Technique 5: Small model + verifier loop (no human labels needed) ────────
# Source: OUROBOROS Kernel Mint
# Train a tiny model with RL where the reward is a deterministic verifier.
# The verifier is the product — it can't be fooled, so the model earns real wins.
# Scales to: grammar checks, test suites, benchmarks, schema validators.

def verified_generation_loop(prompt: str, small_model_fn, verifier_fn,
                             max_attempts: int = 3) -> dict:
    """
    Let a small model draft N times. Each draft is checked by a deterministic
    verifier (not another LLM). Return the first passing draft.
    Collect failed attempts for RL training signal.
    """
    attempts = []
    for i in range(max_attempts):
        draft = small_model_fn(prompt)
        result = verifier_fn(draft)
        attempts.append({"draft": draft, "passed": result["passed"],
                         "reason": result.get("reason")})
        if result["passed"]:
            return {"output": draft, "attempts": i + 1, "history": attempts}
    # All failed — return best attempt and history for offline RL
    return {"output": attempts[-1]["draft"], "attempts": max_attempts,
            "history": attempts, "passed": False}

def example_verifier(draft: str) -> dict:
    """
    Stub verifier. Replace with: JSON schema check, unit test runner,
    AST parse check, SQL EXPLAIN, regex match — anything deterministic.
    """
    try:
        json.loads(draft)
        return {"passed": True}
    except json.JSONDecodeError as e:
        return {"passed": False, "reason": str(e)}

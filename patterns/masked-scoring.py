"""
Masked Next-Token Scoring Pattern
Extracted from Semantique (Build Small Hackathon 2026).

Pattern: Use a single LLM forward pass as a probability scorer over a
constrained vocabulary. Mask the next-token distribution to only allowed
labels, re-normalize the softmax, and read off calibrated probabilities.
One forward pass, no parsing, no retries. Works with Transformers, vLLM, OpenAI.
"""

from __future__ import annotations

import math
import json
from typing import Any


# ---------------------------------------------------------------------------
# 1. CORE MATH — Mask and re-normalize a logit distribution
# ---------------------------------------------------------------------------

def masked_softmax(
    logits: dict[str, float],
    allowed_labels: list[str],
) -> dict[str, float]:
    """
    Mask a logit distribution to only allowed labels and re-normalize.

    Args:
        logits: Dict of token -> logit (or log-probability)
        allowed_labels: List of valid label strings

    Returns:
        Dict of label -> probability (sums to 1.0)
    """
    # Extract logits for allowed labels only
    masked = {}
    for label in allowed_labels:
        if label in logits:
            masked[label] = logits[label]
        else:
            # Label not in top-K logits — assign very low score
            masked[label] = -100.0

    # Softmax normalization
    max_logit = max(masked.values())
    exp_values = {k: math.exp(v - max_logit) for k, v in masked.items()}
    total = sum(exp_values.values())

    return {k: round(v / total, 6) for k, v in exp_values.items()}


# ---------------------------------------------------------------------------
# 2. APPROACH A — HuggingFace Transformers (local model, full control)
# ---------------------------------------------------------------------------

def score_with_transformers(text: str, labels: list[str],
                            model_name: str = "distilbert-base-uncased") -> dict[str, float]:
    """Score text against labels using a local HF model. Requires torch + transformers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    prompt = f"Classify this text into one word: {text}\nLabel:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    label_logits = {}
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if token_ids:
            label_logits[label] = next_token_logits[token_ids[0]].item()
    return masked_softmax(label_logits, labels)


# ---------------------------------------------------------------------------
# 3. APPROACH B — OpenAI API (via logprobs parameter)
# ---------------------------------------------------------------------------

def score_with_openai(text: str, labels: list[str], api_key: str,
                      model: str = "gpt-4o-mini") -> dict[str, float]:
    """Score text via OpenAI logprobs. Request top-K logprobs, mask to labels."""
    import httpx
    prompt = (f"Classify with one word from: {', '.join(labels)}\n"
              f"Text: {text}\nLabel:")
    with httpx.Client(timeout=30) as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1, "logprobs": True, "top_logprobs": 20},
        )
        response.raise_for_status()
    top_logprobs = response.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    logits = {entry["token"].strip().lower(): entry["logprob"] for entry in top_logprobs}
    label_logits = {l: logits.get(l.strip().lower(), -100.0) for l in labels}
    return masked_softmax(label_logits, labels)


# ---------------------------------------------------------------------------
# 4. APPROACH C — vLLM / TGI (via logprobs in OpenAI-compatible API)
# ---------------------------------------------------------------------------

def score_with_vllm(text: str, labels: list[str],
                    api_url: str = "http://localhost:8000",
                    model: str = "default") -> dict[str, float]:
    """Score text via a vLLM/TGI server. Same logprobs trick, local endpoint."""
    import httpx
    prompt = f"Classify: {text}\nOptions: {', '.join(labels)}\nAnswer:"
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{api_url}/v1/chat/completions",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1, "logprobs": True, "top_logprobs": 20},
        )
        response.raise_for_status()
    top_logprobs = response.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    logits = {entry["token"].strip().lower(): entry["logprob"] for entry in top_logprobs}
    label_logits = {l: logits.get(l.strip().lower(), -100.0) for l in labels}
    return masked_softmax(label_logits, labels)


# ---------------------------------------------------------------------------
# 5. CONVENIENCE WRAPPER — Pick the best available approach
# ---------------------------------------------------------------------------

def classify(
    text: str,
    labels: list[str],
    threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Classify text into one of the given labels.
    Returns the winning label and the full probability distribution.

    Args:
        text:      Text to classify
        labels:    Allowed labels (e.g. ["positive", "negative", "neutral"])
        threshold: Minimum confidence to return a label (0.0 = always return)

    Returns:
        {"label": "positive", "confidence": 0.87, "distribution": {...}}
    """
    # In real usage, call one of the score_with_* functions above.
    # This shows how to use the result.
    # distribution = score_with_openai(text, labels, api_key="...")
    # For demo purposes, using fake logits:
    import random
    fake_logits = {label: random.uniform(-2, 2) for label in labels}
    distribution = masked_softmax(fake_logits, labels)

    best_label = max(distribution, key=distribution.get)  # type: ignore
    confidence = distribution[best_label]

    if confidence < threshold:
        return {"label": None, "confidence": confidence,
                "distribution": distribution}

    return {"label": best_label, "confidence": confidence,
            "distribution": distribution}


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Demo the core math with fake logits ---
    print("Masked softmax demo:")

    # Imagine these are raw logits from a model's next-token prediction
    raw_logits = {
        "positive": 2.3,
        "negative": -0.5,
        "neutral": 0.8,
        "happy": 1.9,       # Valid token but not in our label set
        "the": 3.1,         # Common token, high logit, irrelevant
        "very": 2.8,        # Another irrelevant high-logit token
    }

    labels = ["positive", "negative", "neutral"]
    probs = masked_softmax(raw_logits, labels)

    print(f"  Labels: {labels}")
    print(f"  Probabilities: {probs}")
    print(f"  Sum: {sum(probs.values()):.4f}")  # Should be ~1.0
    print(f"  Winner: {max(probs, key=probs.get)}")

    # --- Show classification wrapper ---
    print("\nClassification demo:")
    result = classify(
        "This product exceeded all my expectations!",
        ["positive", "negative", "neutral"],
    )
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Distribution: {result['distribution']}")

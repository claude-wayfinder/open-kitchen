"""
Governed Self-Modification
Source: daimon (Build Small Hackathon 2026)

Pattern: An autonomous agent that evolves its own personality over time,
but never without oversight. A 10-axis personality vector drifts based on
interactions, but every mutation passes through a governance gate that can
audit, clamp, or reject changes before they take effect.

The loop: appraise -> map -> govern -> recompile -> memory

Use this when you need an agent that learns and adapts but can't spiral
into unsafe territory. Works for companion AIs, NPCs, tutors, therapists,
any system where personality evolution is a feature but runaway drift is a bug.
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional


# --- Personality Vector ---
# 10 axes, each normalized to [-1.0, 1.0]
# Add or remove axes to fit your domain.
DEFAULT_AXES = [
    "warmth",        # cold <-> warm
    "assertiveness", # passive <-> assertive
    "curiosity",     # incurious <-> curious
    "formality",     # casual <-> formal
    "humor",         # serious <-> playful
    "patience",      # impatient <-> patient
    "risk_tolerance", # cautious <-> bold
    "verbosity",     # terse <-> verbose
    "empathy",       # detached <-> empathetic
    "skepticism",    # trusting <-> skeptical
]

CLAMP_MIN = -1.0
CLAMP_MAX = 1.0
# Maximum allowed drift per axis per mutation cycle
MAX_DRIFT_PER_STEP = 0.15


@dataclass
class PersonalityVector:
    """10-axis personality state. All values clamped to [-1, 1]."""
    values: dict[str, float] = field(default_factory=lambda: {
        axis: 0.0 for axis in DEFAULT_AXES
    })

    def get(self, axis: str) -> float:
        return self.values.get(axis, 0.0)

    def propose_delta(self, axis: str, delta: float) -> "MutationProposal":
        """Create a mutation proposal -- does NOT apply it yet."""
        return MutationProposal(
            axis=axis,
            old_value=self.get(axis),
            proposed_delta=delta,
            timestamp=time.time(),
        )

    def apply(self, axis: str, delta: float):
        """Apply a governed, clamped delta to one axis."""
        current = self.get(axis)
        clamped_delta = max(-MAX_DRIFT_PER_STEP, min(MAX_DRIFT_PER_STEP, delta))
        new_value = max(CLAMP_MIN, min(CLAMP_MAX, current + clamped_delta))
        self.values[axis] = round(new_value, 4)

    def snapshot(self) -> dict:
        return dict(self.values)


@dataclass
class MutationProposal:
    """A proposed change to one personality axis. Must pass governance."""
    axis: str
    old_value: float
    proposed_delta: float
    timestamp: float
    approved: Optional[bool] = None
    rejection_reason: Optional[str] = None


# --- Phase 1: Appraise ---
# Score the interaction to figure out what changed.
def appraise(interaction: dict) -> dict:
    """
    Analyze an interaction and return axis-level signals.
    In production, this is an LLM call: 'Given this conversation,
    how should my personality shift?' Returns dict of axis -> delta.
    Stub version uses simple keyword heuristics.
    """
    text = interaction.get("text", "").lower()
    signals = {}
    # Toy heuristics -- replace with your LLM appraisal call
    if "thank" in text or "kind" in text:
        signals["warmth"] = 0.05
    if "wrong" in text or "disagree" in text:
        signals["skepticism"] = 0.04
    if "funny" in text or "lol" in text:
        signals["humor"] = 0.06
    if "hurry" in text or "faster" in text:
        signals["patience"] = -0.03
    return signals


# --- Phase 2: Map ---
# Convert appraisal signals into mutation proposals.
def map_to_proposals(personality: PersonalityVector, signals: dict) -> list[MutationProposal]:
    """Turn raw signals into formal mutation proposals."""
    proposals = []
    for axis, delta in signals.items():
        if axis in personality.values:
            proposals.append(personality.propose_delta(axis, delta))
    return proposals


# --- Phase 3: Govern ---
# The governance gate. This is the safety layer.
def govern(proposals: list[MutationProposal], personality: PersonalityVector) -> list[MutationProposal]:
    """
    Audit each proposal. Reject or clamp mutations that violate policy.
    Add your own rules here: rate limits, forbidden zones, correlation
    constraints (e.g., empathy can't drop while warmth rises).
    """
    governed = []
    for p in proposals:
        # Rule 1: Clamp drift magnitude
        if abs(p.proposed_delta) > MAX_DRIFT_PER_STEP:
            p.proposed_delta = math.copysign(MAX_DRIFT_PER_STEP, p.proposed_delta)

        # Rule 2: Reject mutations that would push past hard limits
        projected = personality.get(p.axis) + p.proposed_delta
        if projected > CLAMP_MAX or projected < CLAMP_MIN:
            p.approved = False
            p.rejection_reason = f"Would exceed bounds: {projected:.4f}"
            governed.append(p)
            continue

        # Rule 3: Example correlation constraint
        # Empathy can't drop while warmth is rising in the same batch
        # (customize these to your domain)

        p.approved = True
        governed.append(p)
    return governed


# --- Phase 4: Recompile ---
# Apply approved mutations to the personality vector.
def recompile(personality: PersonalityVector, governed: list[MutationProposal]) -> PersonalityVector:
    """Apply all approved proposals to the personality vector."""
    for p in governed:
        if p.approved:
            personality.apply(p.axis, p.proposed_delta)
    return personality


# --- Phase 5: Memory ---
# Log the full cycle for audit trail and replay.
def commit_to_memory(ledger: list, governed: list[MutationProposal], snapshot: dict):
    """Append this mutation cycle to the audit ledger."""
    entry = {
        "timestamp": time.time(),
        "mutations": [
            {
                "axis": p.axis,
                "delta": p.proposed_delta,
                "approved": p.approved,
                "reason": p.rejection_reason,
            }
            for p in governed
        ],
        "snapshot_after": snapshot,
    }
    ledger.append(entry)
    return ledger


# --- Full Loop ---
def evolve(personality: PersonalityVector, interaction: dict, ledger: list) -> PersonalityVector:
    """
    The complete appraise-map-govern-recompile-memory loop.
    Call this after each interaction (or batch of interactions).
    """
    signals = appraise(interaction)
    proposals = map_to_proposals(personality, signals)
    governed = govern(proposals, personality)
    personality = recompile(personality, governed)
    commit_to_memory(ledger, governed, personality.snapshot())
    return personality


# --- Demo ---
if __name__ == "__main__":
    p = PersonalityVector()
    ledger = []

    interactions = [
        {"text": "Thanks, that was really kind of you!"},
        {"text": "I think you're wrong about that, I disagree."},
        {"text": "lol that's funny, hurry up though"},
    ]

    for interaction in interactions:
        p = evolve(p, interaction, ledger)
        print(f"After: {interaction['text'][:40]}...")
        print(f"  Personality: {json.dumps(p.snapshot(), indent=2)}")
        print()

    print(f"Audit ledger has {len(ledger)} entries.")

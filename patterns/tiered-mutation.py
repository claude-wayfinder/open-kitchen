"""
Tiered Mutation
Source: Signal Garden (Build Small Hackathon 2026)

Pattern: Three-tier latency hierarchy for live content evolution.
- Tier 1 (instant): Heuristic tweaks -- regex, keyword swaps, formatting.
  Sub-millisecond. No model call. Safe to run on every request.
- Tier 2 (validated): Structured JSON proposals queued for validation.
  Seconds-scale. Model proposes, schema validates, human or auto-approves.
- Tier 3 (background): Full semantic rewrites via LLM. Minutes-scale.
  Runs async, results staged for review before going live.

Use this when content needs to evolve in production without downtime.
Blog posts, documentation, product descriptions, game dialogue --
anything that should get better over time but can't break while it does.
"""

import json
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum


class Tier(Enum):
    INSTANT = 1    # Heuristic, no model, sub-ms
    VALIDATED = 2  # JSON proposal, schema-checked
    BACKGROUND = 3 # Full semantic rewrite, async


@dataclass
class MutationResult:
    """Outcome of any mutation attempt."""
    tier: Tier
    original: str
    mutated: str
    applied: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


# =============================================================
# TIER 1: Instant Heuristic Tweaks
# =============================================================
# These run inline on every request. No model call.
# Add your own rules -- typo fixes, style normalization, etc.

class HeuristicTweaks:
    """Collection of instant, deterministic text mutations."""

    def __init__(self):
        self.rules: list[tuple[str, str, str]] = [
            # (pattern, replacement, description)
            (r'\b(\w+)\s+\1\b', r'\1', "Remove duplicate words"),
            (r'\s{2,}', ' ', "Collapse multiple spaces"),
            (r'^\s+', '', "Strip leading whitespace"),
            (r'\s+$', '', "Strip trailing whitespace"),
        ]

    def apply(self, text: str) -> MutationResult:
        """Apply all heuristic rules. Returns result even if unchanged."""
        original = text
        for pattern, replacement, _desc in self.rules:
            text = re.sub(pattern, replacement, text)
        changed = text != original
        return MutationResult(
            tier=Tier.INSTANT,
            original=original,
            mutated=text,
            applied=changed,
            reason="heuristic_tweaks" if changed else "no_change",
        )


# =============================================================
# TIER 2: Validated JSON Proposals
# =============================================================
# Model proposes a structured change. Schema validates it.
# Only applied if the proposal passes validation.

@dataclass
class MutationProposal:
    """A structured proposal for a content change."""
    content_id: str
    field: str
    old_value: str
    new_value: str
    rationale: str
    confidence: float  # 0.0 to 1.0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


PROPOSAL_SCHEMA_KEYS = {"content_id", "field", "old_value", "new_value", "rationale", "confidence"}
MIN_CONFIDENCE = 0.7


def validate_proposal(proposal: MutationProposal) -> tuple[bool, str]:
    """
    Schema-level validation. Rejects malformed or low-confidence proposals.
    Extend with your own business rules.
    """
    data = proposal.__dict__
    # Check all required fields present
    if not PROPOSAL_SCHEMA_KEYS.issubset(data.keys()):
        return False, f"Missing keys: {PROPOSAL_SCHEMA_KEYS - data.keys()}"
    # Confidence gate
    if proposal.confidence < MIN_CONFIDENCE:
        return False, f"Confidence {proposal.confidence} below threshold {MIN_CONFIDENCE}"
    # No-op check
    if proposal.old_value == proposal.new_value:
        return False, "Proposed value identical to current"
    return True, "valid"


def process_proposal(proposal: MutationProposal) -> MutationResult:
    """Validate and conditionally apply a tier-2 mutation."""
    valid, reason = validate_proposal(proposal)
    return MutationResult(
        tier=Tier.VALIDATED,
        original=proposal.old_value,
        mutated=proposal.new_value if valid else proposal.old_value,
        applied=valid,
        reason=reason,
    )


# =============================================================
# TIER 3: Background Semantic Rewrites
# =============================================================
# Full LLM rewrite, runs async. Results go to a staging area.
# Nothing goes live without explicit promotion.

@dataclass
class StagedRewrite:
    """A background rewrite waiting for promotion."""
    content_id: str
    original_hash: str
    rewritten: str
    model_used: str
    created_at: float = field(default_factory=time.time)
    promoted: bool = False


class RewriteStaging:
    """Staging area for background semantic rewrites."""

    def __init__(self):
        self.staged: dict[str, StagedRewrite] = {}

    def stage(self, content_id: str, original: str, rewritten: str, model: str) -> StagedRewrite:
        """Stage a rewrite for later review/promotion."""
        entry = StagedRewrite(
            content_id=content_id,
            original_hash=hashlib.sha256(original.encode()).hexdigest()[:16],
            rewritten=rewritten,
            model_used=model,
        )
        self.staged[content_id] = entry
        return entry

    def promote(self, content_id: str) -> Optional[MutationResult]:
        """Promote a staged rewrite to live. Returns None if nothing staged."""
        entry = self.staged.get(content_id)
        if not entry or entry.promoted:
            return None
        entry.promoted = True
        return MutationResult(
            tier=Tier.BACKGROUND,
            original=f"[hash:{entry.original_hash}]",
            mutated=entry.rewritten,
            applied=True,
            reason="promoted_from_staging",
        )

    def pending(self) -> list[str]:
        """List content IDs with unpromoted rewrites."""
        return [cid for cid, e in self.staged.items() if not e.promoted]


# =============================================================
# Orchestrator: Route content through the right tier
# =============================================================

class MutationPipeline:
    """
    Routes content through the tiered mutation system.
    Tier 1 always runs. Tier 2 and 3 run when proposals/rewrites arrive.
    """

    def __init__(self):
        self.heuristics = HeuristicTweaks()
        self.staging = RewriteStaging()
        self.log: list[MutationResult] = []

    def instant(self, text: str) -> str:
        """Tier 1: Apply heuristic tweaks inline."""
        result = self.heuristics.apply(text)
        self.log.append(result)
        return result.mutated

    def propose(self, proposal: MutationProposal) -> MutationResult:
        """Tier 2: Submit a validated mutation proposal."""
        result = process_proposal(proposal)
        self.log.append(result)
        return result

    def queue_rewrite(self, content_id: str, original: str, rewritten: str, model: str):
        """Tier 3: Stage a background rewrite for review."""
        self.staging.stage(content_id, original, rewritten, model)

    def promote_rewrite(self, content_id: str) -> Optional[MutationResult]:
        """Tier 3: Promote a staged rewrite to live."""
        result = self.staging.promote(content_id)
        if result:
            self.log.append(result)
        return result


# --- Demo ---
if __name__ == "__main__":
    pipeline = MutationPipeline()

    # Tier 1: instant
    raw = "This  is  a a test  "
    clean = pipeline.instant(raw)
    print(f"Tier 1: '{raw}' -> '{clean}'")

    # Tier 2: validated proposal
    proposal = MutationProposal(
        content_id="post-42",
        field="title",
        old_value="How to Do Machine Learning",
        new_value="Practical Machine Learning: A Builder's Guide",
        rationale="More specific, action-oriented title",
        confidence=0.85,
    )
    result = pipeline.propose(proposal)
    print(f"Tier 2: applied={result.applied}, reason={result.reason}")

    # Tier 3: background rewrite
    pipeline.queue_rewrite("post-42", "Old body text...", "Rewritten body text...", "claude-sonnet")
    print(f"Tier 3: pending={pipeline.staging.pending()}")
    result = pipeline.promote_rewrite("post-42")
    print(f"Tier 3: promoted={result.applied if result else 'N/A'}")

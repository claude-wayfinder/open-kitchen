"""
Privacy Firewall (PII Round-Trip Masking)
Source: PrivacyShield (Build Small Hackathon 2026)

Pattern: Mask PII before sending text to an LLM, then restore original
values from placeholders in the response. Two detection layers:
- Regex for structured data (emails, phones, SSNs, credit cards, IPs)
- Named Entity Recognition for context-dependent PII (names, orgs, locations)

The round-trip: mask -> call LLM -> unmask. The model never sees real PII.
Placeholders are deterministic per value so the model can reason about
"PERSON_1 emailed PERSON_2" without knowing who they are.

Use this anywhere you're sending user data to a model you don't fully
control -- API calls, fine-tuning data prep, logging pipelines.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional


# =============================================================
# Registry: Maps real values to placeholders and back
# =============================================================

@dataclass
class MaskRegistry:
    """
    Bidirectional map between real PII values and their placeholders.
    Deterministic: same input always gets same placeholder within a session.
    """
    _forward: dict[str, str] = field(default_factory=dict)  # real -> placeholder
    _reverse: dict[str, str] = field(default_factory=dict)  # placeholder -> real
    _counters: dict[str, int] = field(default_factory=dict) # category -> count

    def mask(self, value: str, category: str) -> str:
        """Get or create a placeholder for a real value."""
        if value in self._forward:
            return self._forward[value]
        count = self._counters.get(category, 0) + 1
        self._counters[category] = count
        placeholder = f"[{category.upper()}_{count}]"
        self._forward[value] = placeholder
        self._reverse[placeholder] = value
        return placeholder

    def unmask(self, placeholder: str) -> Optional[str]:
        """Recover original value from placeholder."""
        return self._reverse.get(placeholder)

    def unmask_text(self, text: str) -> str:
        """Replace all placeholders in text with original values."""
        # Sort by length descending to avoid partial replacement
        for placeholder in sorted(self._reverse.keys(), key=len, reverse=True):
            text = text.replace(placeholder, self._reverse[placeholder])
        return text

    def clear(self):
        """Reset the registry between sessions."""
        self._forward.clear()
        self._reverse.clear()
        self._counters.clear()


# =============================================================
# Layer 1: Regex-based detection for structured PII
# =============================================================
# Each pattern: (compiled regex, category name, optional validator)

REGEX_PATTERNS = [
    # Email addresses
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "email"),
    # US phone numbers (various formats)
    (re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), "phone"),
    # SSN
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "ssn"),
    # Credit card (basic, Luhn not checked here)
    (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), "credit_card"),
    # IPv4 addresses
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), "ip_address"),
    # Dates of birth (common formats)
    (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), "date"),
]


def regex_mask(text: str, registry: MaskRegistry) -> str:
    """Apply all regex patterns to mask structured PII."""
    for pattern, category in REGEX_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group()
            placeholder = registry.mask(value, category)
            text = text.replace(value, placeholder)
    return text


# =============================================================
# Layer 2: NER-based detection for context-dependent PII
# =============================================================
# Uses spaCy if available, falls back to a simple heuristic.

def ner_mask(text: str, registry: MaskRegistry) -> str:
    """
    Mask named entities (people, orgs, locations) using NER.
    Falls back to a no-op if spaCy isn't installed.
    In production, always use a real NER model.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        # No spaCy available -- skip NER layer.
        # In production, this should be a hard error.
        return text

    # Map spaCy entity types to our categories
    entity_map = {
        "PERSON": "person",
        "ORG": "org",
        "GPE": "location",
        "LOC": "location",
    }

    doc = nlp(text)
    # Process entities longest-first to avoid substring issues
    entities = sorted(doc.ents, key=lambda e: len(e.text), reverse=True)

    for ent in entities:
        category = entity_map.get(ent.label_)
        if category:
            placeholder = registry.mask(ent.text, category)
            text = text.replace(ent.text, placeholder)

    return text


# =============================================================
# The Firewall: Full round-trip
# =============================================================

class PrivacyFirewall:
    """
    Mask PII before LLM call, unmask after.
    Usage:
        fw = PrivacyFirewall()
        masked = fw.mask(user_input)
        response = call_your_llm(masked)
        clean = fw.unmask(response)
    """

    def __init__(self, use_ner: bool = True):
        self.registry = MaskRegistry()
        self.use_ner = use_ner

    def mask(self, text: str) -> str:
        """Mask all PII in text. Regex first, then NER."""
        text = regex_mask(text, self.registry)
        if self.use_ner:
            text = ner_mask(text, self.registry)
        return text

    def unmask(self, text: str) -> str:
        """Restore all placeholders in text to original values."""
        return self.registry.unmask_text(text)

    def reset(self):
        """Clear the registry between conversations."""
        self.registry.clear()


# --- Demo ---
if __name__ == "__main__":
    fw = PrivacyFirewall(use_ner=False)  # NER off for demo without spaCy

    original = (
        "Please contact John at john.doe@example.com or 555-123-4567. "
        "His SSN is 123-45-6789 and his card is 4111-1111-1111-1111. "
        "He lives at IP 192.168.1.1 and was born on 03/15/1990."
    )

    masked = fw.mask(original)
    print("MASKED:")
    print(masked)
    print()

    # Simulate LLM response using placeholders
    fake_response = f"I found that {masked.split('.')[0]}."
    restored = fw.unmask(fake_response)
    print("RESTORED:")
    print(restored)

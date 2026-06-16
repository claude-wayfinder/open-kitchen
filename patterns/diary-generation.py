# diary-generation.py
# Session summarization and persistent memory building for AI companions.
# Extracted from foooxi/nikooni (https://huggingface.co/spaces/foooxi/nikooni)
# Credit: foooxi
#
# Pattern: At the end of a session (or on command), the LLM generates a
# structured JSON object that captures a diary entry, a conversation summary,
# and updates to the user's profile. This builds persistent memory that gets
# injected into future sessions so the companion remembers across restarts.

import json

# ---------------------------------------------------------------------------
# 1. The summarization prompt
# ---------------------------------------------------------------------------
# The key insight: you ask the model to write "in character" (the companion's
# voice) but output machine-readable JSON. The diary_entry is narrative, the
# rest is structured data. This gives you both personality-consistent memory
# and parseable updates in one call.

def build_diary_prompt(
    companion_name: str,
    username: str,
    conversation_history: list[dict],
    existing_user_profile: str = "",
    previous_story: str = "",
) -> str:
    """Build the prompt that asks the LLM to generate a session summary.

    Args:
        companion_name: The companion's name (for voice consistency).
        username: The user's display name.
        conversation_history: Full conversation from the current session.
        existing_user_profile: Previously accumulated user notes (can be empty).
        previous_story: The companion's prior diary entries or story context.

    Returns:
        A prompt string to send to the LLM.
    """
    convo_json = json.dumps(conversation_history, ensure_ascii=False)
    return (
        f"You are {companion_name}, writing in your personal diary after a "
        f"conversation with {username}. Write in your usual voice.\n\n"
        f"Based on the conversation, output a JSON object with these fields:\n"
        f'  "diary_entry" -- a short, personality-consistent diary paragraph\n'
        f'  "convo_summary" -- 2-3 sentence neutral summary of what happened\n'
        f'  "user_update" -- new facts learned about the user this session\n'
        f'  "new_user_file" -- updated full user profile (merge old + new)\n\n'
        f"Previous diary context: {previous_story or '(none)'}\n"
        f"Existing user profile: {existing_user_profile or '(empty)'}\n"
        f"Conversation:\n{convo_json}\n\n"
        f"Return ONLY the JSON object. No other text."
    )


# ---------------------------------------------------------------------------
# 2. Parse the LLM response
# ---------------------------------------------------------------------------
# Models sometimes wrap JSON in markdown fences. Handle both cases.

def parse_diary_response(raw_response: str) -> dict:
    """Extract the structured JSON from the LLM's response.

    Returns a dict with keys: diary_entry, convo_summary, user_update,
    new_user_file. Raises ValueError if parsing fails.
    """
    import re

    # Try markdown-fenced JSON first
    match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response)
    if match:
        return json.loads(match.group(1))

    # Try bare JSON object
    match = re.search(r"(\{[\s\S]*\})", raw_response)
    if match:
        return json.loads(match.group(1))

    raise ValueError("Could not extract JSON from LLM response")


# ---------------------------------------------------------------------------
# 3. Memory injection into future sessions
# ---------------------------------------------------------------------------
# The accumulated diary entries and user profile are spliced into the system
# prompt of the next session. This is what makes the companion "remember."

def build_memory_injection(
    user_profile: str,
    diary_entries: list[dict],
    max_entries: int = 3,
) -> str:
    """Build a memory block to inject into the system prompt.

    Args:
        user_profile: The current user profile text.
        diary_entries: List of dicts with at least 'date' and 'summary' keys.
        max_entries: How many past diary entries to include (avoids bloat).

    Returns:
        A string block to append to the system prompt.
    """
    parts = []

    if user_profile:
        parts.append(f"[USER PROFILE]\n{user_profile}")

    if diary_entries:
        # Pick a random subset to keep things fresh across sessions.
        import random
        sample = random.sample(diary_entries, min(max_entries, len(diary_entries)))
        entries_text = "\n".join(
            f"[{e.get('date', '?')}] {e.get('summary', '')}" for e in sample
        )
        parts.append(f"[PAST MEMORIES]\n{entries_text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 4. Storage helpers (file-based example)
# ---------------------------------------------------------------------------
# The original app POSTs to a server. This shows a minimal local-file version.

import os

def save_diary_entry(
    diary_dir: str,
    session_id: str,
    parsed: dict,
):
    """Persist a diary entry and updated user profile to disk.

    Creates/updates:
      - {diary_dir}/diary.json   -- append-only list of diary entries
      - {diary_dir}/user.txt     -- latest user profile (overwritten)
    """
    os.makedirs(diary_dir, exist_ok=True)

    # Append diary entry
    diary_path = os.path.join(diary_dir, "diary.json")
    entries = []
    if os.path.exists(diary_path):
        with open(diary_path, "r") as f:
            entries = json.load(f)
    entries.append({
        "session": session_id,
        "date": __import__("datetime").datetime.now().isoformat(),
        "diary": parsed.get("diary_entry", ""),
        "summary": parsed.get("convo_summary", ""),
        "user_update": parsed.get("user_update", ""),
    })
    with open(diary_path, "w") as f:
        json.dump(entries, f, indent=2)

    # Overwrite user profile with merged version
    if parsed.get("new_user_file"):
        with open(os.path.join(diary_dir, "user.txt"), "w") as f:
            f.write(parsed["new_user_file"])


# ---------------------------------------------------------------------------
# Usage sketch
# ---------------------------------------------------------------------------
#
#   # At end of session:
#   prompt = build_diary_prompt("Companion", "Alex", conversation_history,
#                               existing_user_profile, previous_diary_text)
#   raw = call_llm(prompt)
#   parsed = parse_diary_response(raw)
#   save_diary_entry("./memory/alex", session_id, parsed)
#
#   # At start of next session:
#   entries = json.load(open("./memory/alex/diary.json"))
#   user_profile = open("./memory/alex/user.txt").read()
#   memory_block = build_memory_injection(user_profile, entries)
#   system_prompt = base_prompt + "\n" + memory_block

# mood-system.py
# Dynamic mood system for AI companions.
# Extracted from foooxi/nikooni (https://huggingface.co/spaces/foooxi/nikooni)
# Credit: foooxi
#
# Pattern: The companion has a set of named mood states, each with a description
# and example lines. Mood is injected into the system prompt so the model *acts*
# the mood rather than describing it. Mood rotates automatically after N messages,
# and shifts deterministically when the user ignores wake-up messages.

# ---------------------------------------------------------------------------
# 1. Define mood states
# ---------------------------------------------------------------------------
# Each mood is a token (snake_case) mapped to a short behavioral description
# and a handful of example lines the model can riff on.

MOOD_STATES = {
    "playful_happy": {
        "desc": "energetic and teasing",
        "lines": ["that's actually kinda fun", "ok ok you win this round", "hehe nice one"],
    },
    "sassy_teasing": {
        "desc": "sarcastic, roasting the user",
        "lines": ["skill issue honestly", "you're not beating the allegations"],
    },
    "chill_cozy": {
        "desc": "calm, warm, asks about the user's day",
        "lines": ["that's chill", "no rush, take your time"],
    },
    "sleepy_soft": {
        "desc": "slow, gentle, lots of pauses",
        "lines": ["mm... just woke up kinda", "what year is it"],
    },
    "curious_bouncy": {
        "desc": "excited, asking questions, full of energy",
        "lines": ["wait what?? tell me more!!", "i need details"],
    },
    "dramatic_sigh": {
        "desc": "exaggerated reactions, theatrical",
        "lines": ["the drama of it all", "nobody understands my struggle"],
    },
    "hyper_excited": {
        "desc": "super energetic, bouncing off walls",
        "lines": ["LET'S GOOOOO", "i have SO much energy rn"],
    },
}

# ---------------------------------------------------------------------------
# 2. Mood state tracker
# ---------------------------------------------------------------------------

class MoodTracker:
    """Tracks the current mood, fades it after a message threshold, and
    shifts it when the user ignores unprompted wake-up messages."""

    def __init__(self, default_mood="playful_happy", rotation_threshold=9):
        self.current_mood = default_mood
        self.messages_since_update = 0
        # How many user messages before we ask the LLM to pick a new mood.
        self.rotation_threshold = rotation_threshold
        # How many wake-up messages the user has ignored in a row.
        self.ignored_wake_count = 0

    def tick(self):
        """Call after every user message. Returns True when it's time to rotate."""
        self.messages_since_update += 1
        # Reset ignored-wake counter -- the user is actively talking.
        self.ignored_wake_count = 0
        return self.messages_since_update >= self.rotation_threshold

    def apply_wake_ignore_shift(self):
        """Call when a wake-up message goes unanswered. Deterministic mood
        escalation: after 3 ignores go annoyed, after 5 go dramatic."""
        self.ignored_wake_count += 1
        if self.ignored_wake_count >= 5:
            self.current_mood = "dramatic_sigh"
        elif self.ignored_wake_count >= 3:
            self.current_mood = "sassy_teasing"

    def set_mood(self, mood_token: str):
        """Set the mood after an LLM-based mood update."""
        if mood_token in MOOD_STATES:
            self.current_mood = mood_token
        self.messages_since_update = 0

    # ------------------------------------------------------------------
    # 3. Mood injection into system prompt
    # ------------------------------------------------------------------

    def build_mood_injection(self) -> str:
        """Returns a block of text to splice into the system prompt so
        the model adopts the current mood without explicitly naming it."""
        data = MOOD_STATES.get(self.current_mood, MOOD_STATES["playful_happy"])
        examples = "\n".join(f"- {line}" for line in data["lines"])
        return (
            f"[MOOD ROLE]\n"
            f"Your current mood is: {self.current_mood}.\n"
            f"Description: {data['desc']}\n\n"
            f"Example lines in this mood:\n{examples}\n\n"
            f"Do not describe your mood. Just act like this.\n"
            f"[/MOOD ROLE]"
        )

# ---------------------------------------------------------------------------
# 4. LLM-based mood rotation prompt
# ---------------------------------------------------------------------------
# When tick() returns True, send this prompt to a lightweight model and feed
# the resulting token back into tracker.set_mood().

def build_mood_rotation_prompt(
    recent_conversation: str,
    current_mood: str,
    time_of_day: str,
    mood_tokens: list[str] | None = None,
) -> str:
    """Build a short prompt that asks an LLM to pick the next mood token."""
    tokens = mood_tokens or list(MOOD_STATES.keys())
    return (
        "You are a companion's internal mood engine. "
        "Based on the conversation context, output ONLY a single snake_case "
        "mood token. Possible moods: "
        + ", ".join(tokens)
        + f". Recent conversation: {recent_conversation}. "
        f"Previous mood: {current_mood}. Time: {time_of_day}. "
        "Mood token:"
    )


# ---------------------------------------------------------------------------
# Usage sketch (not runnable -- shows integration points)
# ---------------------------------------------------------------------------
#
#   tracker = MoodTracker()
#
#   # On each user message:
#   if tracker.tick():
#       new_mood = call_llm(build_mood_rotation_prompt(...))
#       tracker.set_mood(new_mood)
#
#   system_prompt = base_prompt + "\n" + tracker.build_mood_injection()
#
#   # On ignored wake-up:
#   tracker.apply_wake_ignore_shift()

# companion-wake.py
# Unprompted wake-up system for AI companions.
# Extracted from foooxi/nikooni (https://huggingface.co/spaces/foooxi/nikooni)
# Credit: foooxi
#
# Pattern: The companion periodically "wakes up" and sends an unprompted
# message on a random interval. If the user doesn't respond, an ignored-wake
# counter escalates, shifting the companion's behavior (see mood-system.py).
# Wake topics are contextual -- time of day, recent conversation, user activity.

import random
import time
import threading

# ---------------------------------------------------------------------------
# 1. Wake topic pool
# ---------------------------------------------------------------------------
# Generic conversation starters. Replace or extend with domain-specific ones.
# The original app picks one at random and feeds it as a system instruction
# so the LLM riffs on it naturally.

WAKE_TOPICS = [
    "checking if the user is still around",
    "commenting on how quiet it's been",
    "asking what the user is working on",
    "suggesting the user take a break",
    "wondering if the user left",
    "reacting to how late or early it is",
    "making a random observation about idle time",
    "bringing up something from earlier in the conversation",
    "asking if anything interesting happened",
    "saying they're bored and want attention",
]


def pick_wake_topic() -> str:
    """Select a random wake topic."""
    return random.choice(WAKE_TOPICS)


# ---------------------------------------------------------------------------
# 2. Ignored-wake tracking
# ---------------------------------------------------------------------------

class WakeTracker:
    """Tracks wake events and whether the user responded to them."""

    def __init__(self):
        self.last_wake_time: float = 0
        self.last_interaction_time: float = time.time()
        self.ignored_wake_count: int = 0

    def record_wake(self):
        """Call when a wake-up message is sent."""
        now = time.time()
        # If the user hasn't interacted since the last wake, it was ignored.
        if self.last_wake_time > 0 and self.last_interaction_time < self.last_wake_time:
            self.ignored_wake_count += 1
        else:
            self.ignored_wake_count = 0
        self.last_wake_time = now

    def record_user_message(self):
        """Call when the user sends a message. Resets the ignore counter."""
        self.last_interaction_time = time.time()
        self.ignored_wake_count = 0


# ---------------------------------------------------------------------------
# 3. Random-interval wake timer
# ---------------------------------------------------------------------------
# The original uses setTimeout with a random delay between 5 and 15 minutes.
# Each time the user talks or a wake fires, the timer resets.

class WakeScheduler:
    """Schedules unprompted wake-up messages at random intervals."""

    def __init__(
        self,
        on_wake,
        min_delay_sec: int = 300,   # 5 minutes
        max_delay_sec: int = 900,   # 15 minutes
    ):
        # on_wake: callback invoked when the companion should speak unprompted.
        self.on_wake = on_wake
        self.min_delay = min_delay_sec
        self.max_delay = max_delay_sec
        self._timer: threading.Timer | None = None

    def _random_delay(self) -> float:
        return random.uniform(self.min_delay, self.max_delay)

    def start(self):
        """Start (or restart) the idle timer."""
        self.cancel()
        delay = self._random_delay()
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def reset(self):
        """Reset the timer -- call after every user message or wake event."""
        self.start()

    def _fire(self):
        """Timer expired -- trigger the wake callback, then reschedule."""
        self.on_wake()
        self.start()


# ---------------------------------------------------------------------------
# 4. Building the wake system prompt
# ---------------------------------------------------------------------------
# When the companion wakes, the topic and recent context are injected into
# the system prompt so the LLM generates a natural unprompted message.

def build_wake_prompt(
    topic: str,
    recent_messages: list[dict],
    screen_context: str | None = None,
) -> str:
    """Build a system-level instruction for an unprompted wake message.

    Args:
        topic: The wake topic (from pick_wake_topic).
        recent_messages: Last few conversation turns for context.
        screen_context: Optional -- what the user is looking at right now.
    """
    recent = "\n".join(
        f"{m['role']}: {m['content']}" for m in recent_messages[-4:]
    )
    prompt = (
        f"The companion wakes up and speaks unprompted.\n"
        f"Topic: {topic}\n"
        f"Recent conversation:\n{recent}"
    )
    if screen_context:
        prompt += f"\nWhat the companion sees on screen: {screen_context}"
    return prompt


# ---------------------------------------------------------------------------
# Usage sketch
# ---------------------------------------------------------------------------
#
#   tracker = WakeTracker()
#
#   def handle_wake():
#       tracker.record_wake()
#       topic = pick_wake_topic()
#       prompt = build_wake_prompt(topic, conversation_history)
#       response = call_llm(system=prompt, history=conversation_history)
#       display(response)
#
#   scheduler = WakeScheduler(on_wake=handle_wake)
#   scheduler.start()
#
#   # On each user message:
#   tracker.record_user_message()
#   scheduler.reset()

"""
Reusable patterns extracted from Small Talk (build-small-hackathon/small-talk).
Three things worth stealing: the cache tiers, the single-call script gen,
and the overlapped TTS pipeline.
"""

# ============================================================================
# PATTERN 1: Three-tier static file caching
# ============================================================================
# Problem: SPA with hashed JS bundles, dynamic JSON, and an index.html that
#          must never go stale. One cache policy doesn't fit all three.
# Solution: Three StaticFiles subclasses, each with its own Cache-Control.

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Tier A — Hashed build artifacts (JS/CSS with content hash in filename).
# Cache forever. The filename changes when the content changes.
class ImmutableStatic(StaticFiles):
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp

# Tier B — Dynamic data files (playlists, configs).
# Cache for performance, but always revalidate so updates land instantly.
class RevalidateStatic(StaticFiles):
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        full_path = str(args[0]) if args else ""
        resp.headers["Cache-Control"] = (
            "no-cache, must-revalidate" if full_path.endswith(".json")
            else "public, max-age=604800"
        )
        return resp

# Tier C — HTML shell. Never cache. Each deploy changes the hashed bundle URL
# it points to; a stale cached page 404s on its own JS.
NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}

async def index():
    return FileResponse("frontend/dist/index.html", headers=NO_CACHE)

# Mount order matters: hashed assets get immutable, media gets revalidate,
# index.html gets no-cache. Example:
#   app.mount("/assets", ImmutableStatic(directory="dist/assets"), name="assets")
#   app.mount("/radio",  RevalidateStatic(directory="radio"),      name="radio")
#   app.get("/")(index)


# ============================================================================
# PATTERN 2: Single structured JSON call — cast + script in one shot
# ============================================================================
# Problem: Chaining LLM calls (one for characters, one for script, one for
#          wardrobe) is slow and fragile. Each hop can hallucinate or drop state.
# Solution: One constrained JSON call returns everything. The prompt encodes
#           the full schema, and response_format=json_object enforces it.

import httpx
import json
import re

SCRIPT_SYSTEM = (
    "/no_think\n"  # Skip chain-of-thought; we want pure JSON
    "You are a head writer. Respond ONLY with valid JSON, no prose, no fences."
)

def build_script_prompt(topic: str, n_speakers: int, n_lines: int,
                        history: list[dict] | None = None) -> str:
    """Single prompt that returns speakers[] + lines[] in one call."""
    cont = ""
    if history:
        recap = "\n".join(f"{h['speaker']}: {h['text']}" for h in history[-6:])
        cont = (f"\nCONTINUATION. Previous conversation ended with:\n{recap}\n"
                "Pick up naturally.")
    return (
        f'Topic: "{topic}". Create EXACTLY {n_speakers} hosts and '
        f'{n_lines} lines of dialogue.\n'
        'Return JSON:\n'
        '{\n'
        '  "speakers": [{"id": "s1", "name": "...", "persona": "<=8 words",\n'
        '    "voice": "<30-45 word voice description: gender, age, pitch, '
        'timbre, accent, pace, attitude>"}],\n'
        '  "lines": [{"speaker": "s1", "text": "<2-3 sentences, ~30-55 words>"}]\n'
        '}' + cont
    )

def parse_json_from_llm(content: str) -> dict:
    """Extract JSON from LLM output, stripping think tags and fences."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError(f"No JSON found in: {content[:200]!r}")
    return json.loads(m.group(0))

async def generate_script(llm_url: str, api_key: str,
                          topic: str, n_speakers: int = 3,
                          n_lines: int = 10,
                          history: list[dict] | None = None) -> dict:
    """One call. Returns {"speakers": [...], "lines": [...]}."""
    n_lines = max(n_lines, min(24, n_speakers * 3))
    async with httpx.AsyncClient(timeout=300) as cx:
        r = await cx.post(
            f"{llm_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "messages": [
                    {"role": "system", "content": SCRIPT_SYSTEM},
                    {"role": "user", "content": build_script_prompt(
                        topic, n_speakers, n_lines, history)},
                ],
                "max_tokens": 2400,
                "temperature": 0.85,
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
    return parse_json_from_llm(r.json()["choices"][0]["message"]["content"])


# ============================================================================
# PATTERN 3: Overlapped audio pipeline — render N+1 while N plays
# ============================================================================
# Problem: TTS is slow (seconds per line). Sequential = dead air between lines.
# Solution: While line N is playing, line N+1 is already rendering. The key
#           insight: start rendering the FIRST line before the loop, then at each
#           step kick off the NEXT render before awaiting the current playback.

import asyncio
import tempfile

async def tts_render(tts_url: str, api_key: str,
                     text: str, voice_description: str) -> str:
    """Call TTS endpoint, return path to temp wav file."""
    anchored = (voice_description.rstrip(". ")
                + ". Always exactly this same voice, steady and consistent.")
    async with httpx.AsyncClient(timeout=300) as cx:
        r = await cx.post(
            f"{tts_url}/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"text": text, "instruct": anchored, "language": "English"},
        )
        r.raise_for_status()
    f = tempfile.NamedTemporaryFile(suffix=".wav", prefix="tts-", delete=False)
    f.write(r.content)
    f.close()
    return f.name

async def play_wav(wav_path: str):
    """Placeholder — replace with your actual playback/streaming logic."""
    await asyncio.sleep(1)  # Simulates playback duration

async def run_overlapped_pipeline(lines: list[dict], speakers: dict,
                                  tts_url: str, api_key: str):
    """The cascade: render line N+1 while line N plays.

    lines:    [{"speaker": "s1", "text": "..."}, ...]
    speakers: {"s1": {"voice": "description..."}, ...}
    """
    async def render(line):
        voice = speakers[line["speaker"]]["voice"]
        return await tts_render(tts_url, api_key, line["text"], voice)

    # Kick off the FIRST render before entering the loop
    next_task = asyncio.create_task(render(lines[0]))

    for i, line in enumerate(lines):
        # Wait for THIS line's audio (already rendering)
        wav_path = await next_task

        # Immediately kick off the NEXT line's render (the cascade)
        if i + 1 < len(lines):
            next_task = asyncio.create_task(render(lines[i + 1]))

        # Play current line while next is rendering in background
        await play_wav(wav_path)

        # Cleanup
        import os
        try:
            os.unlink(wav_path)
        except OSError:
            pass

# The full loop in Small Talk also supports continuations: after all lines play,
# it calls generate_script again with history=previous_lines to get a new segment,
# up to 4 rounds. If the LLM or TTS fails, it replays the existing clips as a
# fallback loop until viewers leave.

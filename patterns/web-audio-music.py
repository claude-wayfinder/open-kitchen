"""
Generative Web Audio Music Pattern
Extracted from Nightwave (Build Small Hackathon 2026).

Pattern: Music from math. Web Audio API oscillators, zero samples, zero licensing.
Python generates a self-contained HTML page with JS synthesis. No audio files.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# 1. MUSIC THEORY HELPERS — Notes, scales, chords
# ---------------------------------------------------------------------------

# Standard tuning: A4 = 440 Hz. Every note is a frequency.
NOTE_FREQUENCIES = {}
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _build_frequency_table():
    """Build a lookup table of note name -> frequency (Hz)."""
    for midi in range(21, 109):  # A0 to C8
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        octave = (midi // 12) - 1
        name = _NOTE_NAMES[midi % 12]
        NOTE_FREQUENCIES[f"{name}{octave}"] = round(freq, 2)

_build_frequency_table()


def scale(root: str, pattern: list[int]) -> list[str]:
    """
    Generate a scale from a root note and a pattern of semitone intervals.

    Common patterns:
      Major:         [0, 2, 4, 5, 7, 9, 11]
      Minor:         [0, 2, 3, 5, 7, 8, 10]
      Pentatonic:    [0, 2, 4, 7, 9]
      Blues:          [0, 3, 5, 6, 7, 10]
    """
    root_idx = _NOTE_NAMES.index(root[:-1])
    octave = int(root[-1])
    notes = []
    for interval in pattern:
        idx = root_idx + interval
        oct = octave + idx // 12
        note_name = _NOTE_NAMES[idx % 12]
        full_name = f"{note_name}{oct}"
        if full_name in NOTE_FREQUENCIES:
            notes.append(full_name)
    return notes


# ---------------------------------------------------------------------------
# 2. SEQUENCE BUILDER — Define what to play and when
# ---------------------------------------------------------------------------

def build_sequence(
    notes: list[str],
    durations: list[float] | None = None,
    bpm: float = 120.0,
    wave_type: str = "sine",
) -> list[dict]:
    """
    Build a note sequence for the Web Audio player.

    Args:
        notes:      List of note names (e.g. ["C4", "E4", "G4"])
        durations:  Duration of each note in beats (default: all quarter notes)
        bpm:        Tempo in beats per minute
        wave_type:  Oscillator type: "sine", "square", "sawtooth", "triangle"

    Returns:
        List of dicts ready to serialize into the JS player
    """
    if durations is None:
        durations = [1.0] * len(notes)

    beat_duration = 60.0 / bpm  # seconds per beat
    sequence = []
    current_time = 0.0

    for note, dur in zip(notes, durations):
        freq = NOTE_FREQUENCIES.get(note, 440.0)
        sequence.append({
            "freq": freq,
            "start": round(current_time, 4),
            "duration": round(dur * beat_duration, 4),
            "type": wave_type,
        })
        current_time += dur * beat_duration

    return sequence


# ---------------------------------------------------------------------------
# 3. HTML/JS GENERATOR — The actual Web Audio player
# ---------------------------------------------------------------------------

def generate_player_html(
    sequences: dict[str, list[dict]], title: str = "Generative Music",
    loop: bool = True, master_volume: float = 0.3,
) -> str:
    """Generate a self-contained HTML page with Web Audio music player."""
    seq_json = json.dumps(sequences)

    return f"""<!DOCTYPE html>
<html><head><title>{title}</title>
<style>
  body {{ font-family: system-ui; background: #111; color: #eee;
         display: flex; flex-direction: column; align-items: center;
         justify-content: center; height: 100vh; margin: 0; }}
  button {{ padding: 16px 32px; font-size: 18px; cursor: pointer;
           background: #333; color: #eee; border: 1px solid #555;
           border-radius: 8px; }}
  button:hover {{ background: #444; }}
  .status {{ margin-top: 16px; opacity: 0.6; }}
</style></head>
<body>
<h1>{title}</h1>
<button id="playBtn" onclick="togglePlay()">Play</button>
<div class="status" id="status">Click to start</div>
<script>
const SEQUENCES = {seq_json};
const LOOP = {"true" if loop else "false"};
const MASTER_VOL = {master_volume};

let ctx = null;
let playing = false;
let timeoutIds = [];

function playNote(freq, start, duration, type) {{
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  // Envelope: quick attack, sustain, quick release
  const t = ctx.currentTime + start;
  gain.gain.setValueAtTime(0, t);
  gain.gain.linearRampToValueAtTime(MASTER_VOL, t + 0.02);
  gain.gain.setValueAtTime(MASTER_VOL, t + duration - 0.05);
  gain.gain.linearRampToValueAtTime(0, t + duration);
  osc.connect(gain).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + duration);
}}

function playAll() {{
  let maxEnd = 0;
  for (const [layer, notes] of Object.entries(SEQUENCES)) {{
    for (const n of notes) {{
      playNote(n.freq, n.start, n.duration, n.type);
      const end = n.start + n.duration;
      if (end > maxEnd) maxEnd = end;
    }}
  }}
  if (LOOP) {{
    const id = setTimeout(() => {{ if (playing) playAll(); }}, maxEnd * 1000);
    timeoutIds.push(id);
  }}
}}

function togglePlay() {{
  if (!ctx) ctx = new AudioContext();
  if (playing) {{
    playing = false;
    timeoutIds.forEach(clearTimeout);
    timeoutIds = [];
    ctx.close();
    ctx = null;
    document.getElementById("playBtn").textContent = "Play";
    document.getElementById("status").textContent = "Stopped";
  }} else {{
    playing = true;
    playAll();
    document.getElementById("playBtn").textContent = "Stop";
    document.getElementById("status").textContent = "Playing...";
  }}
}}
</script></body></html>"""


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a simple melody using a pentatonic scale
    pentatonic = scale("C4", [0, 2, 4, 7, 9])
    melody = build_sequence(
        notes=pentatonic + list(reversed(pentatonic)),
        bpm=100,
        wave_type="triangle",
    )

    # Build a bass line
    bass_notes = ["C3", "G2", "A2", "F2"]
    bass = build_sequence(
        notes=bass_notes,
        durations=[2.0, 2.0, 2.0, 2.0],  # half notes
        bpm=100,
        wave_type="sine",
    )

    # Generate the HTML player
    html = generate_player_html(
        sequences={"melody": melody, "bass": bass},
        title="Pentatonic Demo",
    )

    # Write to file
    output_path = "music_demo.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Generated {output_path} — open in a browser to listen")
    print(f"Melody notes: {pentatonic}")
    print(f"Note count: {len(melody)} melody + {len(bass)} bass")

"""
Frequency-as-Seed Determinism Pattern
Extracted from Lost Frequency Radio (Build Small Hackathon 2026).

Pattern: Numeric input (frequency) seeds a PRNG. Same input = same output =
shared experience without sync. The frequency IS the content.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# 1. DETERMINISTIC SEED — The core mechanism
# ---------------------------------------------------------------------------

def frequency_to_seed(frequency: float, epoch: str = "") -> int:
    """
    Convert a frequency (or any numeric input) to a deterministic seed.

    Args:
        frequency: The numeric input (91.7, 440.0, any float)
        epoch:     Optional time-bucket string to rotate content.
                   e.g. "2026-06-15" for daily rotation,
                   "2026-06-15-14" for hourly. Empty = permanent.

    Returns:
        Integer seed suitable for random.seed()
    """
    # Combine frequency + epoch into a single deterministic hash
    raw = f"{frequency:.4f}:{epoch}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return int(digest[:16], 16)


def seeded_random(seed: int) -> random.Random:
    """Create an isolated Random instance with the given seed."""
    rng = random.Random(seed)
    return rng


# ---------------------------------------------------------------------------
# 2. DETERMINISTIC CONTENT GENERATORS — Same seed = same output
# ---------------------------------------------------------------------------

def generate_playlist(
    frequency: float,
    track_pool: list[str],
    n_tracks: int = 10,
    epoch: str = "",
) -> list[str]:
    """
    Generate a deterministic playlist from a frequency.
    Anyone tuned to the same frequency gets the same track order.
    """
    seed = frequency_to_seed(frequency, epoch)
    rng = seeded_random(seed)
    # Sample with replacement if pool is smaller than requested
    if len(track_pool) >= n_tracks:
        return rng.sample(track_pool, n_tracks)
    return [rng.choice(track_pool) for _ in range(n_tracks)]


def generate_color_palette(
    frequency: float,
    n_colors: int = 5,
    epoch: str = "",
) -> list[str]:
    """
    Generate a deterministic color palette from a frequency.
    Same frequency = same colors everywhere.
    """
    seed = frequency_to_seed(frequency, epoch)
    rng = seeded_random(seed)
    return [f"#{rng.randint(0, 0xFFFFFF):06x}" for _ in range(n_colors)]


def generate_name(
    frequency: float,
    epoch: str = "",
) -> str:
    """Generate a deterministic name/title from a frequency."""
    adjectives = [
        "Amber", "Crimson", "Silent", "Lunar", "Electric", "Frozen",
        "Golden", "Hollow", "Iron", "Jade", "Neon", "Obsidian",
        "Phantom", "Radiant", "Solar", "Velvet", "Wild", "Zenith",
    ]
    nouns = [
        "Archive", "Beacon", "Circuit", "Dawn", "Echo", "Forge",
        "Garden", "Harbor", "Index", "Junction", "Knot", "Lattice",
        "Meridian", "Nexus", "Orbit", "Prism", "Relay", "Signal",
    ]
    seed = frequency_to_seed(frequency, epoch)
    rng = seeded_random(seed)
    return f"{rng.choice(adjectives)} {rng.choice(nouns)}"


def generate_world_params(
    frequency: float,
    epoch: str = "",
) -> dict[str, Any]:
    """
    Generate deterministic world/game parameters from a frequency.
    Everyone at the same frequency plays in the same world.
    """
    seed = frequency_to_seed(frequency, epoch)
    rng = seeded_random(seed)
    return {
        "terrain": rng.choice(["forest", "desert", "ocean", "mountains", "tundra"]),
        "time_of_day": rng.choice(["dawn", "noon", "dusk", "midnight"]),
        "weather": rng.choice(["clear", "rain", "fog", "storm", "snow"]),
        "difficulty": round(rng.uniform(0.1, 1.0), 2),
        "color_key": f"#{rng.randint(0, 0xFFFFFF):06x}",
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# 3. EPOCH ROTATION — Time-bucketed content without sync
# ---------------------------------------------------------------------------

def get_epoch(granularity: str = "daily") -> str:
    """
    Get the current time bucket for content rotation.

    Args:
        granularity: "hourly", "daily", "weekly", "permanent"

    Same frequency + same epoch = same content.
    When the epoch rolls over, content changes for everyone simultaneously.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    if granularity == "permanent":
        return ""
    elif granularity == "hourly":
        return now.strftime("%Y-%m-%d-%H")
    elif granularity == "daily":
        return now.strftime("%Y-%m-%d")
    elif granularity == "weekly":
        return f"{now.year}-W{now.isocalendar()[1]:02d}"
    else:
        return now.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 4. FREQUENCY DIAL — Map a continuous range to discrete "stations"
# ---------------------------------------------------------------------------

def snap_to_station(
    frequency: float,
    band_start: float = 88.0,
    band_end: float = 108.0,
    station_count: int = 20,
) -> float:
    """
    Snap a continuous frequency to the nearest discrete station.
    Like an FM radio dial with fixed stations.
    """
    step = (band_end - band_start) / station_count
    clamped = max(band_start, min(band_end, frequency))
    station_idx = round((clamped - band_start) / step)
    return round(band_start + station_idx * step, 1)


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    freq = 91.7
    epoch = get_epoch("daily")

    print(f"Frequency: {freq}")
    print(f"Epoch: {epoch}")
    print(f"Seed: {frequency_to_seed(freq, epoch)}")
    print()

    # Same frequency = same name everywhere
    name = generate_name(freq, epoch)
    print(f"Station name: {name}")

    # Same frequency = same colors everywhere
    colors = generate_color_palette(freq, 5, epoch)
    print(f"Palette: {colors}")

    # Same frequency = same world everywhere
    world = generate_world_params(freq, epoch)
    print(f"World: {json.dumps(world, indent=2)}")

    # Snap to nearest station
    raw_dial = 92.3
    station = snap_to_station(raw_dial)
    print(f"\nDial at {raw_dial} -> snapped to station {station}")

    # Prove determinism: run it again, same results
    name2 = generate_name(freq, epoch)
    assert name == name2, "Determinism broken!"
    print("\nDeterminism verified: same input = same output")

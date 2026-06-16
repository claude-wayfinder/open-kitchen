"""
Persistence + Tick Architecture — extracted from Aether Garden
(huggingface.co/spaces/build-small-hackathon/aether-garden)

Pattern: SQLite WAL locally, HuggingFace Dataset as durable backup.
- On cold boot: pull .db from HF Dataset if local copy is missing/corrupt
- On every tick (and after user-triggered writes): push .db back to HF Dataset
- Autonomous sim loop runs N steps per tick, then backs up

Stripped of all garden/fantasy logic. Plug your own domain in.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# 1. DATABASE LAYER — SQLite + WAL mode
# ---------------------------------------------------------------------------

DB_PATH = Path(os.environ.get("DB_PATH", "data/state.db"))

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS world_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    current_tick    INTEGER NOT NULL DEFAULT 1,
    total_items     INTEGER NOT NULL DEFAULT 0,
    last_tick_run   TEXT,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Add your domain tables here.
-- CREATE TABLE IF NOT EXISTS items ( ... );
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_session() -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    with db_session() as conn:
        conn.executescript(SCHEMA_SQL)
        row = conn.execute("SELECT id FROM world_state WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO world_state (id, current_tick) VALUES (1, 1)"
            )


def get_state() -> dict:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM world_state WHERE id = 1").fetchone()
        return dict(row) if row else {}


# ---------------------------------------------------------------------------
# 2. HF DATASET BACKUP — push/pull the whole .db file
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")


def _db_is_valid() -> bool:
    """True when the .db file exists and has our core table."""
    if not DB_PATH.exists() or DB_PATH.stat().st_size < 512:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='world_state'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def backup_database() -> bool:
    """Push local .db to HF Dataset. Call after every tick / mutation."""
    if not HF_TOKEN or not HF_DATASET_REPO or not DB_PATH.exists():
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(
            HF_DATASET_REPO, repo_type="dataset",
            private=True, exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=str(DB_PATH),
            path_in_repo="state.db",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="State backup",
        )
        return True
    except Exception as e:
        print(f"Backup failed: {e}")
        return False


def restore_database() -> bool:
    """Pull .db from HF Dataset on cold boot (only if local copy is bad)."""
    if not HF_TOKEN or not HF_DATASET_REPO:
        return False
    if _db_is_valid():
        return False  # already good, skip download
    DB_PATH.unlink(missing_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="state.db",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=str(DB_PATH.parent),
        )
        dl = Path(downloaded)
        if dl.resolve() != DB_PATH.resolve():
            dl.replace(DB_PATH)
        return _db_is_valid()
    except Exception as e:
        print(f"Restore failed: {e}")
        return False


# ---------------------------------------------------------------------------
# 3. BOOT SEQUENCE — restore, init, seed
# ---------------------------------------------------------------------------

def ensure_ready() -> None:
    """Full cold-boot sequence. Idempotent — safe to call on every request."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _db_is_valid():
        try:
            restore_database()
        except Exception:
            pass
        if DB_PATH.exists() and not _db_is_valid():
            DB_PATH.unlink(missing_ok=True)

    init_database()

    # Plug your seed data here:
    # with db_session() as conn:
    #     seed_defaults(conn)

    if not _db_is_valid():
        raise RuntimeError(f"Database failed to initialize at {DB_PATH.resolve()}")


# ---------------------------------------------------------------------------
# 4. SIMULATION TICK — the autonomous loop
# ---------------------------------------------------------------------------

def execute_tick() -> dict:
    """
    One complete simulation tick. The 7 steps:

    1. ensure_ready()          — restore from HF if cold boot
    2. Read current state      — get tick number, load active items
    3. Generate world event    — your domain logic (AI call, random, etc.)
    4. Run interactions        — pair items, generate outcomes, write results
    5. Update memories/state   — roll up per-item state changes
    6. Advance tick counter    — bump current_tick, set last_tick_run
    7. Backup to HF Dataset   — push .db so next cold boot has it
    """
    ensure_ready()

    summary = {"tick": None, "event": None, "interactions": 0}

    # Step 2: read current state
    with db_session() as conn:
        world = conn.execute(
            "SELECT * FROM world_state WHERE id = 1"
        ).fetchone()
        current_tick = world["current_tick"]

    # Step 3: generate world event (plug your logic)
    # event = generate_event(current_tick)
    # summary["event"] = event

    # Step 4: run interactions (plug your logic)
    # pairs = select_pairs(...)
    # for a, b in pairs:
    #     result = run_interaction(a, b)
    #     write_result(result)
    #     summary["interactions"] += 1

    # Step 5: update per-item state (plug your logic)
    # for item_id, changes in pending_updates.items():
    #     apply_updates(item_id, changes)

    # Step 6: advance tick counter
    with db_session() as conn:
        conn.execute(
            """
            UPDATE world_state SET
                current_tick = current_tick + 1,
                last_tick_run = ?,
                updated_at = datetime('now')
            WHERE id = 1
            """,
            (datetime.now().isoformat(),),
        )

    with db_session() as conn:
        ws = conn.execute(
            "SELECT current_tick FROM world_state WHERE id = 1"
        ).fetchone()
        summary["tick"] = ws["current_tick"] if ws else "?"

    # Step 7: backup to HF Dataset
    try:
        backup_database()
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# 5. WIRING — Gradio app with Timer-driven auto-refresh
# ---------------------------------------------------------------------------

def build_app():
    """
    Minimal Gradio shell showing how the pieces connect.
    Aether Garden uses:
      - gr.Timer(10) for presence heartbeat
      - gr.Timer(30) for UI refresh
      - A manual "Run tick" button that calls execute_tick()
      - backup_database() after every user-triggered write (summon, etc.)
    """
    import gradio as gr

    ensure_ready()

    with gr.Blocks(title="Persistence Skeleton") as demo:
        status = gr.HTML(value=f"<p>Tick: {get_state().get('current_tick', '?')}</p>")
        tick_btn = gr.Button("Run simulation tick")
        tick_output = gr.HTML()

        def run_tick():
            result = execute_tick()
            return (
                f"<p>Tick {result['tick']} complete. "
                f"{result['interactions']} interactions.</p>"
            )

        tick_btn.click(fn=run_tick, outputs=[tick_output]).then(
            fn=lambda: f"<p>Tick: {get_state().get('current_tick', '?')}</p>",
            outputs=[status],
        )

        # Auto-refresh UI every 30s (reads local DB, no tick advancement)
        refresh_timer = gr.Timer(30)
        refresh_timer.tick(
            fn=lambda: f"<p>Tick: {get_state().get('current_tick', '?')}</p>",
            outputs=[status],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)

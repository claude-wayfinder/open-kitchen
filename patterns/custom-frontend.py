"""
Gradio Server Pattern — Custom HTML Frontend + API Routes
Extracted from PitchFight AI (HuggingFace Spaces hackathon entry).

This is the skeleton. No app logic. Just the wiring.

WHAT THIS DOES:
  - Uses gradio.Server instead of gradio.Blocks/Interface
  - Gives you a raw FastAPI-like app (no Gradio widgets)
  - You mount your own HTML/JS/CSS frontend
  - You define REST API routes the frontend calls
  - Deploys on HF Spaces exactly like a normal Gradio app

DIRECTORY STRUCTURE:
  app.py              <-- this file
  frontend/
    index.html        <-- your custom UI
    style.css
    app.js
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PORT = int(os.getenv("PORT", "7860"))
FRONTEND_DIR = Path(__file__).parent / "frontend"

# ---------------------------------------------------------------------------
# Create the app — this is the whole trick.
# gradio.Server = FastAPI app that HF Spaces knows how to run.
# No gr.Blocks, no gr.Interface, no widgets. Just routes.
# ---------------------------------------------------------------------------

app = Server()

# ---------------------------------------------------------------------------
# REST API routes — your frontend hits these with fetch()
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/api/do-something")
def api_do_something(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    POST route. Frontend sends JSON, you return JSON.

    Frontend calls it like:
        const res = await fetch("/api/do-something", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ prompt: "hello" })
        });
        const data = await res.json();
    """
    prompt = payload.get("prompt", "")
    # ... your logic here ...
    return {"result": f"You said: {prompt}"}


@app.post("/api/another-route")
def api_another_route(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Another endpoint. Same pattern. Add as many as you want."""
    return {"ok": True}


# ---------------------------------------------------------------------------
# Gradio @app.api routes — same handlers, but accessible via gradio_client
#
# This lets people call your app programmatically:
#     from gradio_client import Client
#     client = Client("your-space")
#     result = client.predict({"prompt": "hello"}, api_name="/do_something")
#
# Optional. Skip if you don't need gradio_client compatibility.
# ---------------------------------------------------------------------------

@app.api(name="do_something")
def gradio_do_something(payload: dict[str, Any]) -> dict[str, Any]:
    """Same handler, exposed via Gradio's queue/client protocol."""
    prompt = payload.get("prompt", "")
    return {"result": f"You said: {prompt}"}


# ---------------------------------------------------------------------------
# Frontend — serve your custom HTML/JS/CSS
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def homepage() -> HTMLResponse:
    """Serve index.html at the root."""
    index_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# Mount the frontend directory so JS/CSS/images are accessible at /frontend/*
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting on http://127.0.0.1:{PORT}")
    app.launch(show_error=True, server_port=PORT)


# ===========================================================================
# EXAMPLE: frontend/index.html (minimal)
# ===========================================================================
#
# <!DOCTYPE html>
# <html>
# <head>
#   <title>My App</title>
#   <link rel="stylesheet" href="/frontend/style.css">
# </head>
# <body>
#   <h1>My App</h1>
#   <input id="prompt" placeholder="Say something...">
#   <button onclick="go()">Send</button>
#   <pre id="output"></pre>
#
#   <script>
#   async function go() {
#     const prompt = document.getElementById("prompt").value;
#     const res = await fetch("/api/do-something", {
#       method: "POST",
#       headers: {"Content-Type": "application/json"},
#       body: JSON.stringify({ prompt })
#     });
#     const data = await res.json();
#     document.getElementById("output").textContent = JSON.stringify(data, null, 2);
#   }
#   </script>
# </body>
# </html>
# ===========================================================================

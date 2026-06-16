"""
ZeroGPU Local Fallback Pattern
Extracted from KnowledgeMesh (Build Small Hackathon 2026).

Pattern: Write @spaces.GPU once. It becomes a no-op locally. Your code runs
on ZeroGPU when deployed to HuggingFace Spaces, and on your laptop (CPU or
local GPU) when developing — no code changes, no if/else, no environment flags.

WHY THIS EXISTS:
  ZeroGPU Spaces use @spaces.GPU to request a GPU for the duration of a
  function call. But that decorator doesn't exist on your local machine,
  and you don't want to litter your code with platform checks. This shim
  makes the decorator transparent: if spaces is installed and functional,
  use it. Otherwise, run the function as-is.

THE TRICK:
  Import spaces. If it fails, replace @spaces.GPU with a no-op decorator.
  That's it. Three lines of setup, zero changes to your actual code.
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. THE SHIM — Three lines that make everything else work
# ---------------------------------------------------------------------------

try:
    import spaces  # type: ignore
    # On HF ZeroGPU Spaces, this module exists and provides @spaces.GPU
    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False

    # Create a fake 'spaces' module with a no-op GPU decorator
    class _FakeSpaces:
        """Stand-in for the spaces module when running locally."""

        @staticmethod
        def GPU(fn: Callable | None = None, *, duration: int = 60, **kwargs):
            """
            No-op decorator that matches the @spaces.GPU signature.

            On ZeroGPU: requests a GPU for `duration` seconds.
            Locally: does nothing, just calls the function normally.
            """
            if fn is not None:
                # Used as @spaces.GPU (no parentheses)
                @functools.wraps(fn)
                def wrapper(*args, **kw):
                    return fn(*args, **kw)
                return wrapper
            else:
                # Used as @spaces.GPU(duration=120)
                def decorator(f):
                    @functools.wraps(f)
                    def wrapper(*args, **kw):
                        return f(*args, **kw)
                    return wrapper
                return decorator

    # Inject the fake module so `import spaces` works everywhere
    spaces = _FakeSpaces()
    sys.modules["spaces"] = spaces  # type: ignore


# ---------------------------------------------------------------------------
# 2. USAGE — Write your GPU functions normally
# ---------------------------------------------------------------------------
# These decorators work identically on ZeroGPU and on your laptop.
# On ZeroGPU: they acquire a GPU for the specified duration.
# Locally: they're no-ops that call the function directly.

@spaces.GPU  # type: ignore
def run_inference(text: str) -> str:
    """
    Example: run a model inference that needs a GPU.

    On ZeroGPU, this function gets a GPU allocated before it runs.
    Locally, it just runs — use your own GPU or CPU.
    """
    # Your actual inference code goes here:
    # model = load_model()
    # result = model.generate(text)
    return f"Processed: {text}"


@spaces.GPU(duration=120)  # type: ignore
def run_heavy_inference(texts: list[str]) -> list[str]:
    """
    Example: longer GPU task that needs more time.

    @spaces.GPU(duration=120) requests 120 seconds of GPU time on ZeroGPU.
    Locally, the duration parameter is ignored.
    """
    # Your batch inference code goes here
    return [f"Processed: {t}" for t in texts]


# ---------------------------------------------------------------------------
# 3. DEVICE DETECTION — Know where you're running
# ---------------------------------------------------------------------------

def get_device() -> str:
    """
    Detect the best available device. Useful for loading models.

    Returns: "cuda", "mps" (Apple Silicon), or "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def is_zerogpu() -> bool:
    """Check if we're running on a ZeroGPU Space."""
    return _HAS_SPACES


# ---------------------------------------------------------------------------
# 4. GRADIO APP SKELETON — Drop-in for HF Spaces
# ---------------------------------------------------------------------------

def build_app():
    """
    Example Gradio app that uses the GPU-decorated functions.
    Deploys to ZeroGPU Spaces unchanged.
    """
    import gradio as gr

    def process(text: str) -> str:
        device = get_device()
        result = run_inference(text)
        return f"[{device}] {result}"

    demo = gr.Interface(
        fn=process,
        inputs=gr.Textbox(label="Input"),
        outputs=gr.Textbox(label="Output"),
        title="ZeroGPU Fallback Demo",
    )
    return demo


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"ZeroGPU available: {is_zerogpu()}")
    print(f"Device: {get_device()}")
    print(run_inference("Hello, world!"))
    print(run_heavy_inference(["one", "two", "three"]))

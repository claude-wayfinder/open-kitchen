# The Open Kitchen

Shared techniques from Build Small Hackathon 2026. Take what you need. Leave what you learn.

950 apps built by 950 teams in two weeks. Every one of them solved a problem someone else is about to hit. This repo collects the patterns worth reusing so nobody has to reinvent the wheel in a hackathon.

## What's Here

Extracted, stripped-down, reusable patterns from hackathon submissions. Not copies of apps -- just the clever parts, documented so you can drop them into your own build.

### Architecture Patterns

- **[Custom Frontend on Gradio](patterns/custom-frontend.py)** -- Mount your own HTML/JS/CSS on `gr.Server`, hide Gradio entirely, talk to backend via `/api/*` routes. From PitchFight AI.
- **[Persistent World State](patterns/persistence.py)** -- SQLite + WAL mode + HF Dataset backup. Survive Space restarts. From Aether Garden.
- **[Three-Tier Caching](patterns/caching.py)** -- Immutable hashed assets, revalidate dynamic JSON, no-cache HTML. Free performance. From Small Talk.
- **[Single Structured JSON Call](patterns/structured-json.py)** -- One schema, one call, parse or retry. No chaining. From Small Talk + Claim-Ready.
- **[Deterministic Guard Layer](patterns/guard-layer.py)** -- Model output verified by code before serving. From Jawbreaker + Her.

### Agent Patterns

- **[Append-Only Event Ledger](patterns/event-ledger.py)** -- Agents coordinate through an immutable log. No direct communication. From Multi-Agent Lab.
- **[Tool Decorator with Auto-Schema](patterns/tool-decorator.py)** -- `@tool` generates OpenAI function-calling schemas from Python type hints. From OpenMythos.
- **[Agentic Loop with Explicit Termination](patterns/agentic-loop.py)** -- Agent decides when it's done via `final_message` tool. From OpenMythos.

### Companion Patterns

- **[Dynamic Mood System](patterns/mood-system.py)** -- Named mood states injected into system prompts. LLM rotates mood every N messages. Ignored wake-ups shift mood deterministically. From Nikooni.
- **[Companion Wake System](patterns/companion-wake.py)** -- Random-interval unprompted messages. Ignored-wake tracking escalates behavior. Contextual topic selection. From Nikooni.
- **[Diary Generation](patterns/diary-generation.py)** -- LLM generates structured JSON diary entry + conversation summary + user profile updates at session end. Persistent memory across restarts. From Nikooni.

### Simulation Patterns

- **[Deterministic Core + LLM Decision Layer](patterns/simulation-core.py)** -- Engine owns all state, models only pick actions. From World Simulator.
- **[Three-Layer Memory](patterns/memory-layers.py)** -- Raw tick log + semantic episodes + self-authored narrative. From World Simulator.
- **[Heterogeneous Multi-Model Agents](patterns/multi-model.py)** -- Different labs' models in the same world create emergent behavior. From Thousand Token Wood.

### Utility Patterns

- **[ZeroGPU Local Fallback](patterns/zerogpu-fallback.py)** -- `@spaces.GPU` as no-op locally. Write once, run anywhere. From KnowledgeMesh.
- **[Generative Web Audio Music](patterns/web-audio-music.py)** -- Parametric synth, zero samples, zero licensing. From Nightwave.
- **[Frequency-as-Seed Determinism](patterns/frequency-seed.py)** -- Same input always generates same output. Shared experiences without sync. From Lost Frequency Radio.
- **[Offline AI Mesh Network](patterns/mesh-network.py)** -- mDNS discovery, capability-based routing, E2E encryption. No cloud required. From HearthNet.
- **[Spaced Repetition Memory](patterns/spaced-repetition-memory.py)** -- Capture user claims, resurface on SM-2 schedule, track belief drift. From Mycelium.
- **[Masked Next-Token Scoring](patterns/masked-scoring.py)** -- One forward pass as a classifier. Mask logits to allowed labels, re-normalize. From Semantique.

## Contributing

Found a clever pattern in a hackathon submission? Extract it, strip the app-specific parts, document it, and PR it here.

Rules:
1. Patterns only -- not full app copies
2. Credit the source app
3. Keep it under 200 lines
4. If you can't explain it in a README section, it's not a pattern yet

## Credit

Every pattern credits its source app. These people built the clever thing -- we just wrote down how they did it.

Built at Heuremen. Let's stay connected and keep information free.

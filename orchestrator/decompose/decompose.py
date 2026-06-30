"""Stage 1 — user prompt -> skeletal Scene Graph (one Gemini Flash call).

Captures only what is stated/implied; leaves attributes mostly empty (Stage 3
fills them). Sets the GenerationIntent, including `ambiguity_score`, which drives
how aggressively Stage 3 invents detail.
"""

from __future__ import annotations

from ..config import settings
from ..gemini_client import generate_json
from ..graph.schema import SceneGraph

SYSTEM = """You are a film pre-production assistant. Decompose the user's video idea into a \
structured scene graph for downstream cinematic planning.

Rules:
- Identify distinct ENTITIES (characters, objects, locations, props, vehicles, crowds).
- Identify a sensible list of SHOTS (even a one-line idea implies at least one shot; split \
distinct actions/POVs into separate shots). Durations must be 4, 6, or 8 seconds.
- Add EDGES: which entities APPEARS_IN which shot, LOCATED_IN for the setting, NEXT for \
narrative order, INTERACTS_WITH for blocking.
- Capture genre/era hints into the global STYLE stub.
- Do NOT invent rich detail yet (that happens later). Only record what is stated or strongly \
implied, and mark any inferred value with prov="INFERRED".
- Set GenerationIntent: shot_count_hint, total_duration_s, platform (youtube|tiktok|reels|generic), \
aspect_ratio (16:9 for landscape/youtube, 9:16 for tiktok/reels/shorts), wants_dialogue, and \
ambiguity_score in [0,1] (0 = the user fully specified everything, 1 = a bare one-liner)."""


def decompose(user_prompt: str) -> SceneGraph:
    """Run the decomposition call and return a skeletal SceneGraph."""
    graph = generate_json(
        prompt=f"User video idea:\n\n{user_prompt}",
        schema=SceneGraph,
        model=settings.gemini_flash,
        system=SYSTEM,
        temperature=0.4,  # low: extraction, not invention
    )
    graph.user_prompt = user_prompt
    return graph

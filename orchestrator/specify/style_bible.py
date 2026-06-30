"""Stage 3a — fill the global STYLE 'world bible' (one cached Gemini call)."""

from __future__ import annotations

from ..config import settings
from ..gemini_client import generate_json
from ..graph.schema import SceneGraph, Style

SYSTEM = """You are a cinematographer + production designer. Given a video idea and any genre/era \
hints, produce a COMPLETE visual style bible. Be specific and use professional film vocabulary.
Fill: genre, era, palette (color grade), film_stock, lens_kit, lighting_philosophy, mood.
Every value is INFERRED with a confidence. This bible is reused for every shot, so keep it global \
and internally consistent."""


def fill_style(graph: SceneGraph) -> Style:
    """Return a fully-populated Style for the graph (does not mutate in place)."""
    hint = (
        f"Idea: {graph.user_prompt}\n"
        f"Genre hint: {graph.style.genre.value if graph.style.genre else 'unspecified'}\n"
        f"Era hint: {graph.style.era.value if graph.style.era else 'unspecified'}"
    )
    return generate_json(
        prompt=hint, schema=Style, model=settings.gemini_pro,
        system=SYSTEM, temperature=0.8,
    )

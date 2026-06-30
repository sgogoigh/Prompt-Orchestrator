"""Stage 3e — continuity reconciliation across the whole graph.

Two jobs:
1. Structural (deterministic): flag entities appearing in >= 2 shots as needing a
   reference image (topology.mark_recurring_entities).
2. Semantic (Gemini, optional): a global pass that resolves contradictions
   (wardrobe/lighting drift, style violations) across shots.
"""

from __future__ import annotations

from ..config import settings
from ..gemini_client import generate_json
from ..graph.schema import SceneGraph
from ..graph import topology

SYSTEM = """You are a continuity supervisor. Given a fully-drafted scene graph, find and FIX \
contradictions across shots: an entity described differently in different shots, lighting that \
violates the style bible, dialogue attributed to an absent entity, or duration drift. Return the \
corrected graph. Preserve all ids and any USER/RETRIEVED attributes. Change only what is needed for \
consistency."""


def reconcile(graph: SceneGraph, *, semantic: bool = True) -> SceneGraph:
    """Mark recurring entities, then optionally run the semantic reconcile pass."""
    topology.mark_recurring_entities(graph, min_appearances=2)

    if not semantic:
        return graph

    fixed = generate_json(
        prompt=graph.model_dump_json(),
        schema=SceneGraph,
        model=settings.gemini_pro,
        system=SYSTEM,
        temperature=0.3,  # low: we want corrections, not new invention
    )
    # re-apply the structural flags on the returned graph (the model may drop them)
    topology.mark_recurring_entities(fixed, min_appearances=2)
    fixed.user_prompt = graph.user_prompt
    return fixed

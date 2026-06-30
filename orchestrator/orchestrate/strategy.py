"""Stage 4a — choose the Veo 3.1 generation strategy from graph topology.

Pure logic. Decides HOW we drive Veo; the serializer then produces the prompt
text accordingly and veo/client.py dispatches.
"""

from __future__ import annotations

from enum import Enum

from ..graph.schema import EdgeType, SceneGraph
from ..graph import topology


class Strategy(str, Enum):
    SINGLE = "single"               # one shot, <= 8s  -> single T2V/I2V call
    TIMESTAMP = "timestamp"         # multi-shot within 8s -> one generation, timestamp prompting
    SCENE_EXTENSION = "scene_extension"  # > 8s continuous -> chain on last ~1s / 24 frames
    FIRST_LAST = "first_last"       # defined start+end states -> First & Last Frame interpolation


def select_strategy(graph: SceneGraph) -> Strategy:
    """Pick the single best strategy for this graph."""
    total = topology.total_duration(graph)
    n_shots = len(graph.shots)

    if topology.has_edge_type(graph, EdgeType.TRANSITIONS_TO) and n_shots >= 2:
        return Strategy.FIRST_LAST
    if total > 8 or topology.has_edge_type(graph, EdgeType.CONTINUES_FROM):
        return Strategy.SCENE_EXTENSION
    if n_shots > 1:
        return Strategy.TIMESTAMP
    return Strategy.SINGLE


def reference_images(graph: SceneGraph, limit: int = 3) -> list[str]:
    """Up to `limit` reference-image paths for Veo Ingredients (recurring entities first)."""
    counts = topology.appearance_counts(graph)
    recurring = sorted(
        (e for e in graph.entities if e.reference_image),
        key=lambda e: counts.get(e.id, 0),
        reverse=True,
    )
    return [e.reference_image for e in recurring[:limit] if e.reference_image]

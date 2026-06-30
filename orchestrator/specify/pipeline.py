"""Stage 3 — the fill-in DAG runner.

   ┌─ 3a style_bible ─┐
   │                  ├─► 3c shot_design (parallel) ─► 3e continuity ─► 3f reference_images
   └─ 3b entity_enrich ┘
        (parallel)

Style bible is computed once and cached, then entity-enrichment and shot-design
fan out concurrently (I/O-bound Gemini calls -> threads). `ambiguity_score`
gates how much we invent.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import settings
from ..graph.schema import SceneGraph
from . import continuity, entity_enrich, reference_images, shot_design, style_bible

_MAX_WORKERS = 8


def specify(graph: SceneGraph) -> SceneGraph:
    """Run the full Stage-3 enrichment and return the fully-specified graph."""
    # 3a — global style bible (once, then cached on the graph)
    graph.style = style_bible.fill_style(graph)

    # 3b — per-entity enrichment (parallel)
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        graph.entities = list(
            pool.map(lambda e: entity_enrich.enrich_entity(e, graph), graph.entities)
        )

    # 3c/3d — per-shot design + soundstage (parallel)
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        graph.shots = list(
            pool.map(lambda s: shot_design.design_shot(s, graph), graph.shots)
        )

    # 3e — continuity reconcile (+ flag recurring entities)
    graph = continuity.reconcile(graph, semantic=True)

    # 3f — reference images for recurring entities (Veo Ingredients)
    if settings.enable_reference_images:
        graph = reference_images.generate_reference_images(graph)

    return graph

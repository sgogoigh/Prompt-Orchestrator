"""Stage 3b — per-entity enrichment (parallelizable: one Gemini Flash call per entity)."""

from __future__ import annotations

from ..config import settings
from ..gemini_client import generate_json
from ..graph.schema import Entity, SceneGraph

SYSTEM = """You are a character/production designer. Given one entity and the global style bible, \
flesh out its concrete, filmable details and return the SAME entity with its `attributes` map \
populated. For a character fill: appearance, age, build, wardrobe, distinguishing_features. For an \
object/prop/vehicle fill: material, condition, color, scale. For a location fill: era, decor, \
lighting_env. Stay consistent with the style bible. Mark every value prov="INFERRED" with a \
confidence. Do NOT overwrite any attribute whose prov is "USER" or "RETRIEVED"."""


def enrich_entity(entity: Entity, graph: SceneGraph) -> Entity:
    """Return the entity with attributes filled. (Call once per entity, in parallel.)"""
    style = graph.style.model_dump_json()
    prompt = f"Style bible:\n{style}\n\nEntity to enrich:\n{entity.model_dump_json()}"
    enriched = generate_json(
        prompt=prompt, schema=Entity, model=settings.gemini_flash,
        system=SYSTEM, temperature=0.8,
    )
    # preserve identity fields the model might drift on
    enriched.id, enriched.type, enriched.name = entity.id, entity.type, entity.name
    enriched.canonical = entity.canonical
    # keep USER/RETRIEVED attributes authoritative
    for a in entity.attributes:
        if a.prov.value in ("USER", "RETRIEVED"):
            enriched.set_attr(a)
    return enriched

"""Stage 3f — generate a canonical reference image per recurring entity.

These images are what Veo's 'Ingredients' feature uses to keep a character/object
consistent across shots. We build a tight visual prompt from the entity's filled
attributes + the style bible, generate with Gemini 2.5 Flash Image, and store the
path on `entity.reference_image`.
"""

from __future__ import annotations

import os

from ..config import settings
from ..gemini_client import generate_image
from ..graph.schema import Entity, SceneGraph


def _image_prompt(entity: Entity, graph: SceneGraph) -> str:
    attrs = "; ".join(f"{a.key}: {a.value}" for a in entity.attributes)
    s = graph.style
    style = "; ".join(
        v.value for v in (s.palette, s.film_stock, s.lens_kit, s.mood) if v
    )
    return (
        f"Character/asset reference sheet, neutral pose, plain background. "
        f"{entity.name}. {attrs}. Visual style: {style}. "
        f"Single subject, clear full view, consistent canonical appearance."
    )


def generate_reference_images(graph: SceneGraph) -> SceneGraph:
    """For each entity flagged `needs_reference`, generate + attach a reference image."""
    os.makedirs(os.path.join(settings.asset_store_dir, "refs"), exist_ok=True)
    for ent in graph.entities:
        if not ent.needs_reference or ent.reference_image:
            continue
        out = os.path.join(settings.asset_store_dir, "refs", f"{ent.id}.png")
        try:
            generate_image(_image_prompt(ent, graph), out_path=out, model=settings.gemini_image)
            ent.reference_image = out
        except Exception as exc:  # non-fatal: degrade to text-only consistency
            print(f"[reference_images] WARN: {ent.id} image gen failed: {exc}")
    return graph

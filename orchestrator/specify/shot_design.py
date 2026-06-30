"""Stage 3c — per-shot cinematography design (parallelizable: one call per shot).

Also folds in Stage 3d soundstage in the same call (cheaper, and dialogue/SFX are
tightly coupled to the action). Split into soundstage.py if you want them separate.
"""

from __future__ import annotations

from ..config import settings
from ..gemini_client import generate_json
from ..graph.schema import SceneGraph, Shot

SYSTEM = """You are a film director + sound designer. Design ONE shot and return the SAME shot \
fully specified.

Fill `cinematography`: shot_size, camera_move, lens (focal length + depth of field), angle — using \
professional terms. Fill `action` with ONE clear primary action. Fill shot-specific `lighting` and \
the `emotion` beat. Fill `soundstage`: in-character `dialogue` lines (keep them short), event-locked \
`sfx`, an `ambient` bed, and optional `music`. Respect the style bible and the entities present. \
Mark inferred values prov="INFERRED" with a confidence. Keep duration in {4,6,8}."""


def design_shot(shot: Shot, graph: SceneGraph) -> Shot:
    """Return the shot with cinematography + soundstage filled."""
    present = [e.model_dump() for e in graph.entities_in_shot(shot.id)]
    prompt = (
        f"Style bible:\n{graph.style.model_dump_json()}\n\n"
        f"Entities present in this shot:\n{present}\n\n"
        f"Shot to design:\n{shot.model_dump_json()}"
    )
    designed = generate_json(
        prompt=prompt, schema=Shot, model=settings.gemini_pro,
        system=SYSTEM, temperature=0.85,
    )
    designed.id, designed.order = shot.id, shot.order
    if designed.duration_s not in (4, 6, 8):
        designed.duration_s = 8
    return designed

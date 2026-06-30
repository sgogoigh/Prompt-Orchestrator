"""Stage 3d — soundstage design (OPTIONAL standalone).

By default soundstage is produced inside shot_design.py (dialogue/SFX are coupled
to the action). Use this module only if you want a dedicated, separately-tunable
audio pass — e.g. to enforce a consistent musical motif across shots.

TODO (P1): implement a per-shot or whole-graph audio pass that fills
`shot.soundstage` and harmonizes music across the NEXT chain.
"""

from __future__ import annotations

from ..graph.schema import SceneGraph, Shot


def design_soundstage(shot: Shot, graph: SceneGraph) -> Shot:
    raise NotImplementedError(
        "Standalone soundstage pass not wired — soundstage is filled by shot_design.py. "
        "Implement here only if you split audio into its own tunable stage."
    )

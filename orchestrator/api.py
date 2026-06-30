"""Public entrypoint — wires the full pipeline.

    user prompt → structured graph → retrieval → specification → orchestrated prompt → Veo 3.1 → output

Usage:
    from orchestrator.api import generate
    result = generate("a detective meets a mysterious woman in his office, noir", dry_run=True)
    print(result.veo_prompt)        # inspect the orchestrated prompt without spending Veo credits
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .config import settings
from .decompose.decompose import decompose
from .gemini_client import generate_image
from .graph import topology
from .graph.schema import SceneGraph
from .orchestrate import strategy as strat
from .orchestrate.critique import critique_prompt
from .orchestrate.serializer import serialize, serialize_per_shot, serialize_shot
from .orchestrate.strategy import Strategy
from .specify.pipeline import specify
from .store.store import store


@dataclass
class OrchestrationResult:
    graph: SceneGraph                     # the fully-specified scene graph
    veo_prompt: str                       # the final NL prompt (joined, for display)
    strategy: str                         # which Veo strategy was chosen
    reference_images: list[str] = field(default_factory=list)
    per_shot_prompts: list[str] = field(default_factory=list)  # used by multi-clip strategies
    clips: list[str] = field(default_factory=list)             # output MP4 paths (empty on dry_run)


# strategies that render one clip PER shot and stitch them
_MULTI = (Strategy.SCENE_EXTENSION, Strategy.FIRST_LAST)


def _endpoint_frames(graph: SceneGraph) -> tuple[str, str]:
    """FIRST_LAST: render a start still (first shot) and an end still (last shot)."""
    shots = topology.ordered_shots(graph)
    refs_dir = os.path.join(settings.asset_store_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    first = os.path.join(refs_dir, "frame_first.png")
    last = os.path.join(refs_dir, "frame_last.png")
    generate_image("Cinematic film still. " + serialize_shot(graph, shots[0]), out_path=first)
    generate_image("Cinematic film still. " + serialize_shot(graph, shots[-1]), out_path=last)
    return first, last


def generate(
    user_prompt: str,
    *,
    dry_run: bool = False,
    persist: bool = True,
) -> OrchestrationResult:
    """Run the orchestrator end-to-end.

    dry_run=True stops before the (paid) Veo call and returns the orchestrated
    prompt(s) + graph — ideal for inspecting/iterating cheaply.
    """
    # 1 — DECOMPOSE: prompt -> skeletal graph
    graph = decompose(user_prompt)

    # 2 — RETRIEVE: bind to canonical assets for continuity (if any)
    graph = store.retrieve(graph)

    # 3 — SPECIFY: fill in everything (style, entities, shots, sound, continuity, refs)
    graph = specify(graph)

    # 4 — ORCHESTRATE: graph -> Veo-native prompt(s) (+ self-critique)
    chosen = strat.select_strategy(graph)
    refs = strat.reference_images(graph)

    if chosen in _MULTI:
        # one prompt per shot — chained (extension) or interpolated (first/last)
        per_shot = [critique_prompt(p) for p in serialize_per_shot(graph)]
        display = "\n\n--- next shot ---\n\n".join(per_shot)
    else:
        per_shot = []
        display = critique_prompt(serialize(graph))

    result = OrchestrationResult(
        graph=graph, veo_prompt=display, strategy=chosen.value,
        reference_images=refs, per_shot_prompts=per_shot,
    )
    if dry_run:
        return result

    # 5 — VEO: dispatch by strategy
    from .veo.client import (
        VeoRequest,
        generate as veo_generate,
        generate_chain,
        generate_transition,
    )

    req = VeoRequest(
        prompt=display,
        reference_images=refs or None,
        aspect_ratio=graph.intent.aspect_ratio or settings.aspect_ratio,
        duration_s=min(graph.intent.total_duration_s or settings.duration_s, 8),
    )

    if chosen == Strategy.SCENE_EXTENSION:
        durations = [min(s.duration_s, 8) for s in topology.ordered_shots(graph)]
        result.clips = generate_chain(per_shot, req, durations=durations)
    elif chosen == Strategy.FIRST_LAST:
        first_img, last_img = _endpoint_frames(graph)
        result.clips = generate_transition(first_img, last_img, per_shot[0], req)
    else:  # SINGLE / TIMESTAMP
        result.clips = veo_generate(req)

    # TODO (P5): run qc.assess on each clip; regenerate up to MAX_QC_RETRIES.

    if persist:
        store.save_graph(graph)   # "construct a graph relation to use it later"
    return result

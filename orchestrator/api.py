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
    qc: list = field(default_factory=list)                     # QCVerdict(s) for the chosen clip(s)
    candidates: list = field(default_factory=list)             # best-of-N: (path, adherence, accept)


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

    # P5 — QC: gate + best-of-N select. Only the trustworthy signals gate
    # (adherence/subject); artifacts/lip-sync stay advisory. See qc/qc.py.
    from .qc.qc import assess, passes_gate, pick_best

    def _qc_gate(clip_path: str, shot_prompt: str) -> bool:
        try:
            return passes_gate(assess(clip_path, shot_prompt))
        except Exception as exc:  # QC must never block delivery of a rendered clip
            print(f"[qc] WARN: assessment skipped ({exc})")
            return True

    if chosen == Strategy.SCENE_EXTENSION:
        durations = [min(s.duration_s, 8) for s in topology.ordered_shots(graph)]
        result.clips = generate_chain(
            per_shot, req, durations=durations,
            qc_gate=_qc_gate if settings.enable_qc_loop else None,
            qc_retries=settings.max_qc_retries if settings.enable_qc_loop else 0,
        )
    elif chosen == Strategy.FIRST_LAST:
        first_img, last_img = _endpoint_frames(graph)
        result.clips = generate_transition(first_img, last_img, per_shot[0], req)
    else:  # SINGLE / TIMESTAMP — best-of-N select over Fast candidates, bounded re-roll
        # Developer API only renders 1 video per call, so N candidates = N calls.
        n = max(1, settings.num_videos) if settings.enable_qc_loop else 1
        candidates: list[str] = []
        for _ in range(n):
            candidates.extend(veo_generate(req))
        if settings.enable_qc_loop and candidates:
            scored = [(p, assess(p, display)) for p in candidates]
            result.candidates = [(p, v.adherence, v.accept) for p, v in scored]
            best = pick_best(scored)
            tries = 0
            while not passes_gate(best[1]) and tries < settings.max_qc_retries:
                tries += 1
                more_scored = [(p, assess(p, display)) for p in veo_generate(req)]
                result.candidates += [(p, v.adherence, v.accept) for p, v in more_scored]
                best = pick_best([best, *more_scored])
            result.clips = [best[0]]
            result.qc = [best[1]]
        else:
            result.clips = candidates

    if persist:
        store.save_graph(graph)   # "construct a graph relation to use it later"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Requirement-graph core path (single clip) with a PRE-GENERATION eval-gate.
#
# This is the wired-in Prompt Requirement Graph (orchestrator/graph/prompt_*):
#   user prompt -> build_prompt (map -> dependency-fill -> evaluate -> serialize)
#   -> EVAL-GATE (refuse / revise / warn) -> Veo (only if the PROMPT is judged good).
#
# The gate is the payoff: Veo credits are spent only on prompts the graph already
# scored complete + coherent. Multi-shot chaining still lives in generate() above;
# extending the ontology to shot sequences is the planned refinement step.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GatedResult:
    user_prompt: str
    veo_prompt: str
    api_params: dict = field(default_factory=dict)
    evaluation: object = None            # graph.prompt_eval.EvalReport
    revised: bool = False                # was the prompt auto-revised by the gate?
    gated: bool = False                  # True = refused on quality/defects, no generation
    needs_confirmation: bool = False     # True = too much was inferred; confirm before spending
    inferred_fields: list = field(default_factory=list)   # [(label, value)] auto-invented
    gate_reason: str = ""
    clips: list = field(default_factory=list)
    qc: list = field(default_factory=list)


def generate_gated(
    user_prompt: str,
    *,
    min_quality: float = 0.65,
    gate_policy: str = "revise",   # "revise" | "refuse" | "warn"
    max_invention: float = 0.85,   # if a larger share was auto-inferred, ask to confirm first
    confirm: bool = False,         # set True to proceed despite high invention
    dry_run: bool = False,
    persist: bool = False,
) -> GatedResult:
    """Build + evaluate the prompt via the requirement graph, gate, then generate.

    Two gates before any Veo spend:
      1. QUALITY/DEFECT gate (min_quality + coherence conflicts / missing-required):
         - "refuse": return without generating
         - "revise": incorporate suggestions, rebuild once, keep the better
         - "warn":   proceed anyway
      2. INVENTION gate: if invention_ratio > max_invention (the graph invented most of
         the film), stop with needs_confirmation + the inferred fields, so the caller can
         confirm/edit before spending. Pass confirm=True to override.
    """
    from .graph.prompt_flow import build_prompt
    from .graph.prompt_graph import Source

    build = build_prompt(user_prompt)
    report = build.evaluation
    revised, reason = False, ""

    # Gate fires on a low score OR a real defect (coherence conflict / missing required).
    def _fails(r) -> bool:
        return bool(r and (r.overall < min_quality or r.has_defects))

    if _fails(report):
        if gate_policy == "refuse":
            return GatedResult(
                user_prompt=user_prompt, veo_prompt=build.veo_prompt,
                api_params=build.api_params, evaluation=report, gated=True,
                gate_reason=(f"overall {report.overall} < {min_quality} or defects "
                             f"(missing={report.missing_required}, conflicts={report.conflicts}); "
                             f"not generating."),
            )
        if gate_policy == "revise":
            guidance = "; ".join(report.suggestions + report.conflicts) or \
                "Make every vague field concrete and resolve any conflicts."
            build2 = build_prompt(f"{user_prompt}\n\nIncorporate this guidance: {guidance}")
            if build2.evaluation and build2.evaluation.overall >= report.overall:
                build, report, revised = build2, build2.evaluation, True
                reason = f"auto-revised: {report.overall} (defects={report.has_defects})"
        # "warn": fall through and generate

    result = GatedResult(
        user_prompt=user_prompt, veo_prompt=build.veo_prompt,
        api_params=build.api_params, evaluation=report, revised=revised, gate_reason=reason,
    )

    # Invention gate: too much was auto-inferred -> ask to confirm before spending.
    if report and report.invention_ratio > max_invention and not confirm:
        result.needs_confirmation = True
        result.inferred_fields = [
            (n.label, n.value) for n in build.graph.nodes.values()
            if n.source is Source.INFERRED and n.filled
        ]
        result.gate_reason = (result.gate_reason + " | " if result.gate_reason else "") + (
            f"invention_ratio {report.invention_ratio} > {max_invention}: "
            f"{len(result.inferred_fields)} fields auto-inferred — confirm before spending."
        )

    if dry_run:
        return result
    if result.needs_confirmation:   # block spend until the caller confirms
        return result

    # Passed both gates -> generate a single clip (+ optional best-of-N QC select)
    from .veo.client import VeoRequest, generate as veo_generate

    req = VeoRequest(
        prompt=build.veo_prompt,
        aspect_ratio=build.api_params.get("aspect_ratio", settings.aspect_ratio),
        resolution=build.api_params.get("resolution", settings.resolution),
        duration_s=build.api_params.get("duration_s", settings.duration_s),
    )
    n = max(1, settings.num_videos) if settings.enable_qc_loop else 1
    candidates: list[str] = []
    for _ in range(n):
        candidates.extend(veo_generate(req))
    if settings.enable_qc_loop and candidates:
        from .qc.qc import assess, pick_best
        scored = [(p, assess(p, build.veo_prompt)) for p in candidates]
        best = pick_best(scored)
        result.clips, result.qc = [best[0]], [best[1]]
    else:
        result.clips = candidates
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Multi-shot core path: requirement graph per shot (shared globals) + gate + chain.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class StoryResult:
    user_prompt: str
    shot_prompts: list = field(default_factory=list)
    durations: list = field(default_factory=list)
    evaluations: list = field(default_factory=list)     # per-shot EvalReport
    worst_overall: float | None = None
    avg_invention: float | None = None
    gated: bool = False
    needs_confirmation: bool = False
    gate_reason: str = ""
    clips: list = field(default_factory=list)


def generate_story(
    user_prompt: str,
    *,
    min_quality: float = 0.6,
    max_invention: float = 0.9,
    confirm: bool = False,
    dry_run: bool = False,
) -> StoryResult:
    """Plan -> per-shot requirement graphs (shared globals) -> gate -> chained render.

    Gates on the WORST shot's quality/defects and the AVERAGE invention ratio before
    spending. Dispatches to veo.generate_chain (last-frame chaining; 1080p-capable).
    """
    from .graph.prompt_story import build_story

    sb = build_story(user_prompt)
    worst, avg_inv = sb.worst_overall, sb.avg_invention
    result = StoryResult(
        user_prompt=user_prompt, shot_prompts=sb.prompts, durations=sb.durations,
        evaluations=[s.evaluation for s in sb.shots], worst_overall=worst, avg_invention=avg_inv,
    )

    if sb.has_defects or (worst is not None and worst < min_quality):
        result.gated = True
        result.gate_reason = f"worst shot {worst} < {min_quality} or per-shot defects present."
    if avg_inv is not None and avg_inv > max_invention and not confirm:
        result.needs_confirmation = True
        result.gate_reason = (result.gate_reason + " | " if result.gate_reason else "") + \
            f"avg invention {avg_inv} > {max_invention}: confirm before spending."

    if dry_run or result.gated or result.needs_confirmation:
        return result

    # Passed -> chained render (last-frame chaining), QC-gating each clip
    from .qc.qc import assess, passes_gate
    from .veo.client import VeoRequest, generate_chain

    ap = sb.shots[0].api_params
    req = VeoRequest(
        prompt=sb.prompts[0],
        aspect_ratio=ap.get("aspect_ratio", settings.aspect_ratio),
        resolution=ap.get("resolution", settings.resolution),
    )

    def _qc_gate(clip_path: str, shot_prompt: str) -> bool:
        try:
            return passes_gate(assess(clip_path, shot_prompt))
        except Exception as exc:
            print(f"[qc] WARN: {exc}")
            return True

    result.clips = generate_chain(
        sb.prompts, req, durations=sb.durations,
        qc_gate=_qc_gate if settings.enable_qc_loop else None,
        qc_retries=settings.max_qc_retries if settings.enable_qc_loop else 0,
    )
    return result

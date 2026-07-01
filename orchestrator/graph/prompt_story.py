"""Multi-shot support — a StoryBuild of per-shot Prompt Requirement Graphs.

A single requirement graph models ONE shot. For a multi-shot narrative we:
  1. plan_story(idea)  -> Gemini splits the idea into N continuous shots + extracts the
     SHARED GLOBALS (genre, visual_style, mood, color_palette, setting) that must stay
     consistent across every shot.
  2. per shot: build a PromptGraph, SEED the story-globals (so per-shot dependency-fill
     produces a consistent look), map the shot-specific content, fill the rest, evaluate.
  3. expose per-shot Veo prompts + durations -> fed to veo.generate_chain (last-frame
     chaining, which supports 1080p).

Cross-shot consistency is enforced structurally by seeding the shared globals into every
shot's graph as authoritative (Source.USER), rather than hoping N independent fills agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..config import settings
from ..gemini_client import generate_json
from .prompt_eval import EvalReport, evaluate
from .prompt_fill import MapResult, fill_graph, fill_technical, map_user_prompt
from .prompt_graph import GLOBAL_KEYS, PromptGraph, Source
from .prompt_serialize import api_params, to_veo_prompt


# ── plan ──────────────────────────────────────────────────────────────────────
class ShotIdea(BaseModel):
    idea: str
    duration_s: int = 8


class StoryPlan(BaseModel):
    shots: list[ShotIdea] = Field(default_factory=list)
    # shared globals for cross-shot consistency
    genre: str = ""
    visual_style: str = ""
    mood: str = ""
    color_palette: str = ""
    setting: str = ""
    platform: str = "generic"


_PLAN_SYS = """You are a director breaking a video idea into a short sequence of CONTINUOUS shots
(2-5) that form a coherent narrative — each shot flows into the next. For each shot give a concise
`idea` (subject + action + framing intent) and a duration (4, 6, or 8s). ALSO extract the shared
globals that MUST stay consistent across all shots: genre, visual_style, mood, color_palette, and
the (continuous) setting, plus the target platform. Keep shots in narrative order."""


def plan_story(user_prompt: str) -> StoryPlan:
    plan = generate_json(user_prompt, StoryPlan, model=settings.gemini_pro,
                         system=_PLAN_SYS, temperature=0.5)
    if not plan.shots:  # degrade to a single shot
        plan.shots = [ShotIdea(idea=user_prompt, duration_s=settings.duration_s)]
    return plan


# ── build ─────────────────────────────────────────────────────────────────────
@dataclass
class ShotBuild:
    idea: str
    graph: PromptGraph
    veo_prompt: str
    api_params: dict
    evaluation: EvalReport | None = None


@dataclass
class StoryBuild:
    plan: StoryPlan
    shots: list[ShotBuild] = field(default_factory=list)

    @property
    def prompts(self) -> list[str]:
        return [s.veo_prompt for s in self.shots]

    @property
    def durations(self) -> list[int]:
        return [int(s.api_params.get("duration_s", 8)) for s in self.shots]

    @property
    def worst_overall(self) -> float | None:
        evs = [s.evaluation.overall for s in self.shots if s.evaluation]
        return min(evs) if evs else None

    @property
    def avg_invention(self) -> float | None:
        ivs = [s.evaluation.invention_ratio for s in self.shots if s.evaluation]
        return round(sum(ivs) / len(ivs), 3) if ivs else None

    @property
    def has_defects(self) -> bool:
        return any(s.evaluation and s.evaluation.has_defects for s in self.shots)


def _seed_globals(graph: PromptGraph, plan: StoryPlan) -> None:
    """Force the shared story-globals into a shot graph as authoritative (consistency)."""
    for key in GLOBAL_KEYS:
        val = getattr(plan, key, "") if hasattr(plan, key) else ""
        if val:
            graph.set(key, val.strip(), Source.USER, 0.9, "story global (cross-shot consistency)")


def build_story(user_prompt: str, *, evaluate_shots: bool = True) -> StoryBuild:
    plan = plan_story(user_prompt)
    builds: list[ShotBuild] = []
    for si in plan.shots:
        g = PromptGraph.build()
        map_user_prompt(g, si.idea)        # shot-specific content/camera
        _seed_globals(g, plan)             # enforce shared look AFTER map (globals win)
        fill_graph(g, si.idea)             # dependency-ordered fill of the rest
        fill_technical(g, MapResult(platform=plan.platform, duration_s=si.duration_s))
        report = evaluate(g) if evaluate_shots else None
        builds.append(ShotBuild(idea=si.idea, graph=g, veo_prompt=to_veo_prompt(g),
                                api_params=api_params(g), evaluation=report))
    return StoryBuild(plan=plan, shots=builds)

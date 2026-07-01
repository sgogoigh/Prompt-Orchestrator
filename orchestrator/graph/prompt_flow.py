"""End-to-end Prompt Requirement Graph flow (prototype).

    user prompt
      -> MAP (Gemini)         which requirement nodes did the user specify
      -> FILL (Gemini)        fill gaps in dependency order, parents as constraints
      -> TECH (deterministic) aspect/resolution/duration from platform + Veo rules
      -> EVALUATE (Gemini)    score the PROMPT (completeness/coherence/specificity)
      -> SERIALIZE            graph -> Veo-native natural language + API params

Later this replaces the ad-hoc decompose/specify/serialize in the core pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .prompt_eval import EvalReport, evaluate
from .prompt_fill import fill_graph, fill_technical, map_user_prompt
from .prompt_graph import PromptGraph
from .prompt_serialize import api_params, to_veo_prompt


@dataclass
class PromptBuild:
    user_prompt: str
    graph: PromptGraph
    veo_prompt: str
    api_params: dict = field(default_factory=dict)
    evaluation: EvalReport | None = None


def build_prompt(user_prompt: str, *, evaluate_prompt: bool = True) -> PromptBuild:
    """Run the full requirement-graph flow and return the built + evaluated prompt."""
    graph = PromptGraph.build()

    mapped = map_user_prompt(graph, user_prompt)   # 1. MAP
    fill_graph(graph, user_prompt)                 # 2. FILL (dependency-ordered)
    fill_technical(graph, mapped)                  # 3. TECH (deterministic)

    report = evaluate(graph) if evaluate_prompt else None   # 4. EVALUATE (pre-generation)

    veo_prompt = to_veo_prompt(graph)              # 5. SERIALIZE
    return PromptBuild(
        user_prompt=user_prompt,
        graph=graph,
        veo_prompt=veo_prompt,
        api_params=api_params(graph),
        evaluation=report,
    )

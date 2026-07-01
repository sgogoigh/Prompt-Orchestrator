"""Evaluate the COMPLETED prompt graph — BEFORE any Veo generation.

This is the evaluation the user actually wanted: judge the *prompt*, not the video.
- completeness: computed in code from node tiers (deterministic, trustworthy).
- coherence + specificity: judged by Gemini over the dependency edges (does lighting
  match mood/style? is each value concrete rather than vague?).
Returns a report with an overall score, conflicts, and actionable suggestions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import settings
from ..gemini_client import generate_json
from .prompt_graph import PromptGraph, Source, TECHNICAL_KEYS, Tier


class PromptEval(BaseModel):
    coherence: float = Field(ge=0.0, le=1.0)     # do dependent fields agree?
    specificity: float = Field(ge=0.0, le=1.0)   # concrete vs vague?
    conflicts: list[str] = Field(default_factory=list)     # e.g. "mood=cheerful vs lighting=dark"
    weak_fields: list[str] = Field(default_factory=list)   # vague/underspecified node keys
    suggestions: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    completeness: float          # code-computed (degenerate post-fill; informational)
    coherence: float
    specificity: float
    invention_ratio: float       # code-computed: share of the spec Gemini inferred (vs user)
    overall: float
    has_defects: bool = False    # missing-required OR coherence conflicts -> gate-worthy
    missing_required: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    weak_fields: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


_EVAL_SYS = """You are a Veo prompt reviewer. You are given a completed shot specification and
its dependency rules (which fields should agree with which). Judge:
- coherence (0..1): do dependent fields agree? Flag conflicts like a cheerful mood with dark,
  oppressive lighting, or a static camera for a high-energy chase.
- specificity (0..1): is each value concrete and filmable, or vague ("nice lighting")?
List concrete conflicts, weak/vague field keys, and short actionable suggestions. Do NOT rewrite
the prompt — only assess it."""


def _completeness(graph: PromptGraph) -> tuple[float, list[str]]:
    """Weighted completeness from node tiers (required weigh most)."""
    weights = {Tier.REQUIRED: 3.0, Tier.RECOMMENDED: 1.5, Tier.OPTIONAL: 0.5}
    total = got = 0.0
    for n in graph.nodes.values():
        w = weights[n.tier]
        total += w
        if n.filled:
            got += w
    missing_req = graph.missing_required()
    return (got / total if total else 1.0), missing_req


def _invention_ratio(graph: PromptGraph) -> float:
    """Share of filled non-technical nodes that were INFERRED (vs USER-specified).

    High ratio = the spec is mostly the model's creative choices, not the user's —
    a fidelity/transparency signal, not necessarily a defect.
    """
    filled = [n for k, n in graph.nodes.items()
              if k not in TECHNICAL_KEYS and n.filled]
    if not filled:
        return 0.0
    inferred = sum(1 for n in filled if n.source is Source.INFERRED)
    return round(inferred / len(filled), 3)


def evaluate(graph: PromptGraph) -> EvalReport:
    completeness, missing_req = _completeness(graph)

    # dependency rules the judge should check
    dep_lines = "\n".join(f"  {c} should agree with: {', '.join(graph.parents(c))}"
                          for c in graph.nodes if graph.parents(c))
    prompt = (
        f"Completed shot specification:\n{graph.as_readable(only_filled=True)}\n\n"
        f"Dependency rules:\n{dep_lines}\n\n"
        "Assess coherence and specificity."
    )
    try:
        pe = generate_json(prompt, PromptEval, model=settings.gemini_pro,
                           system=_EVAL_SYS, temperature=0.2)
    except Exception as exc:  # never block on a failed eval
        print(f"[prompt_eval] WARN: judge skipped ({exc})")
        pe = PromptEval(coherence=0.7, specificity=0.7)

    # `completeness` is near-1.0 after a competent fill, so it can't discriminate —
    # weight it low. Coherence + specificity carry the signal. Real DEFECTS (coherence
    # conflicts or a missing required node) apply a hard penalty so the gate fires on them.
    invention = _invention_ratio(graph)
    # 'requires'-edge violations (e.g. dialogue with no speaker) are hard, code-detected defects
    req_violations = graph.requires_violations()
    conflicts = list(pe.conflicts) + req_violations

    overall = 0.15 * completeness + 0.50 * pe.coherence + 0.35 * pe.specificity
    overall -= 0.12 * min(len(conflicts), 3)   # each concrete conflict hurts
    if missing_req:
        overall *= 0.5
    overall = max(0.0, min(1.0, overall))
    has_defects = bool(missing_req or conflicts)
    return EvalReport(
        completeness=round(completeness, 3),
        coherence=round(pe.coherence, 3),
        specificity=round(pe.specificity, 3),
        invention_ratio=invention,
        overall=round(overall, 3),
        has_defects=has_defects,
        missing_required=missing_req,
        conflicts=conflicts,
        weak_fields=pe.weak_fields,
        suggestions=pe.suggestions,
    )

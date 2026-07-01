"""Fill the Prompt Requirement Graph.

Two Gemini stages + one deterministic stage:
  1. MAP   — read the user prompt onto the ontology (which nodes are USER-specified).
  2. FILL  — fill unfilled nodes in DEPENDENCY order; each node is filled given its
             already-resolved PARENTS as constraints (so lighting respects mood+style, etc.).
  3. TECH  — technical nodes (aspect/resolution/duration) filled deterministically from
             platform + Veo pairing rules (no model call).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import settings
from ..gemini_client import generate_json
from .prompt_graph import PromptGraph, Source, TECHNICAL_KEYS


# ── Gemini I/O (keyed lists — no free-form dicts, Gemini-structured-output safe) ──
class NodeValue(BaseModel):
    key: str
    value: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = ""


class MapResult(BaseModel):
    specified: list[NodeValue] = Field(default_factory=list)  # only nodes the user gave
    platform: str = "generic"        # youtube | tiktok | reels | generic
    duration_s: int = 8              # 4 | 6 | 8


class FillResult(BaseModel):
    filled: list[NodeValue] = Field(default_factory=list)


def _valid_keys(graph: PromptGraph) -> set[str]:
    return set(graph.nodes) - TECHNICAL_KEYS


# ── Stage 1: MAP ──────────────────────────────────────────────────────────────
_MAP_SYS = """You map a user's video idea onto a fixed set of prompt-requirement fields.
Return ONLY the fields the user explicitly stated or STRONGLY implied — do not invent detail
yet (that happens in a later step). For each, give the field `key`, a concise `value`, a
confidence, and a short rationale. Also infer the target `platform` and intended `duration_s`
(4, 6, or 8). Valid keys are provided; ignore anything that doesn't map cleanly."""


def map_user_prompt(graph: PromptGraph, user_prompt: str) -> MapResult:
    keys = sorted(_valid_keys(graph))
    catalog = "\n".join(f"  {graph.nodes[k].key}: {graph.nodes[k].description}" for k in keys)
    prompt = (
        f"User video idea:\n{user_prompt}\n\n"
        f"Valid requirement fields:\n{catalog}\n\n"
        "Map the idea onto these fields (only what the user gave)."
    )
    res = generate_json(prompt, MapResult, model=settings.gemini_flash, system=_MAP_SYS, temperature=0.3)
    valid = _valid_keys(graph)
    for nv in res.specified:
        if nv.key in valid:
            graph.set(nv.key, nv.value.strip(), Source.USER, nv.confidence, nv.rationale)
    return res


# ── Stage 2: FILL (dependency-ordered) ────────────────────────────────────────
_FILL_SYS = """You are a film director + cinematographer completing a shot specification.
Fill ONLY the requested empty fields. CRITICAL: each value must be consistent with the
already-decided fields provided as CONSTRAINTS (e.g. lighting must match the mood + visual
style + time of day; camera movement must match the action's energy). Be specific and use
professional film language. Return each field's `key`, `value`, confidence, and a one-line
rationale explaining how it follows from the constraints."""


def _fill_layer(graph: PromptGraph, layer_keys: list[str], user_prompt: str) -> None:
    targets = [k for k in layer_keys if k not in TECHNICAL_KEYS and not graph.nodes[k].filled]
    if not targets:
        return
    # constraints = every already-filled node (especially the parents of the targets)
    constraints = graph.as_readable(only_filled=True) or "(none yet)"
    to_fill = "\n".join(
        f"  {graph.nodes[k].key}: {graph.nodes[k].description}"
        f"  [depends on: {', '.join(graph.parents(k)) or 'none'}]"
        for k in targets
    )
    prompt = (
        f"Original idea: {user_prompt}\n\n"
        f"Already-decided fields (CONSTRAINTS — stay consistent with these):\n{constraints}\n\n"
        f"Fill these empty fields:\n{to_fill}"
    )
    res = generate_json(prompt, FillResult, model=settings.gemini_pro, system=_FILL_SYS, temperature=0.8)
    target_set = set(targets)
    for nv in res.filled:
        if nv.key in target_set and not graph.nodes[nv.key].filled:
            graph.set(nv.key, nv.value.strip(), Source.INFERRED, nv.confidence, nv.rationale)


def fill_graph(graph: PromptGraph, user_prompt: str) -> None:
    """Fill all unfilled non-technical nodes, layer by layer in dependency order."""
    for layer in graph.topo_layers():
        _fill_layer(graph, layer, user_prompt)


# ── Stage 3: TECHNICAL (deterministic, Veo constraints) ───────────────────────
_VERTICAL = {"tiktok", "reels", "shorts", "instagram"}


def fill_technical(graph: PromptGraph, mapped: MapResult) -> None:
    platform = (mapped.platform or "generic").lower()
    graph.set("platform", platform, Source.USER if graph.nodes["platform"].source is Source.USER
              else Source.DEFAULT, 0.9)
    aspect = "9:16" if any(v in platform for v in _VERTICAL) else "16:9"
    graph.set("aspect_ratio", aspect, Source.DEFAULT, 1.0,
              f"{platform} -> {aspect}")

    resolution = settings.resolution  # e.g. 1080p
    graph.set("resolution", resolution, Source.DEFAULT, 1.0)

    # Veo pairing: 1080p reliably supports only the native 8s clip length.
    dur = 8 if resolution == "1080p" else (mapped.duration_s if mapped.duration_s in (4, 6, 8) else 8)
    graph.set("duration", str(dur), Source.DEFAULT, 1.0,
              "1080p pairs with 8s" if resolution == "1080p" else "from user/default")

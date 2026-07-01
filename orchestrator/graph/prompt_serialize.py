"""Serialize the completed Prompt Requirement Graph into a Veo-native NL prompt.

Deterministic. Composes the 5-part formula from node values, then appends the
soundstage and any negative constraints. Technical nodes (aspect/res/duration)
are returned SEPARATELY as API params — they are not prose.
"""

from __future__ import annotations

from .prompt_graph import PromptGraph


def _v(graph: PromptGraph, key: str) -> str:
    n = graph.node(key)
    if not (n and n.filled and n.value):
        return ""
    # Fills may come back as full sentences; strip trailing punctuation so the
    # comma-joined 5-part formula doesn't produce ".," / ".." artifacts.
    return n.value.strip().rstrip(".; ")


def _join(parts: list[str]) -> str:
    return ", ".join(p for p in (x.strip() for x in parts) if p)


def to_veo_prompt(graph: PromptGraph) -> str:
    cinematography = _join([
        _v(graph, "shot_size"), _v(graph, "camera_movement"), _v(graph, "camera_angle"),
        _v(graph, "lens"), _v(graph, "depth_of_field"),
    ])
    subject = _v(graph, "subject")
    action = _v(graph, "action")
    context = _join([_v(graph, "setting"), _v(graph, "time_of_day"), _v(graph, "lighting")])
    style = _join([
        _v(graph, "visual_style"), _v(graph, "color_palette"),
        _v(graph, "film_stock"), _v(graph, "mood"),
    ])

    visual = _join([cinematography, subject, action, context, style])
    visual = (visual[:1].upper() + visual[1:] + ".") if visual else ""

    # soundstage
    sound_bits: list[str] = []
    if _v(graph, "dialogue"):
        sound_bits.append(_v(graph, "dialogue"))       # expected already-quoted by the fill
    if _v(graph, "sfx"):
        sound_bits.append(f"SFX: {_v(graph, 'sfx')}")
    if _v(graph, "ambient"):
        sound_bits.append(f"Ambient: {_v(graph, 'ambient')}")
    if _v(graph, "music"):
        sound_bits.append(f"Music: {_v(graph, 'music')}")
    sound = " ".join(sound_bits)

    neg = _v(graph, "negative_constraints")
    neg_clause = f" Avoid: {neg}." if neg else ""

    return f"{visual} {sound}{neg_clause}".strip()


def api_params(graph: PromptGraph) -> dict:
    """Technical nodes as Veo API parameters (not part of the prose prompt)."""
    dur = _v(graph, "duration")
    return {
        "aspect_ratio": _v(graph, "aspect_ratio") or "16:9",
        "resolution": _v(graph, "resolution") or "1080p",
        "duration_s": int(dur) if dur.isdigit() else 8,
    }

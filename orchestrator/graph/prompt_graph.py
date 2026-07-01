"""Prompt Requirement Graph — the ontology of what an *ideal* Veo prompt needs.

PROTOTYPE living in `graph/`. Models the *requirements* of a good prompt as typed
requirement nodes + logical dependency edges. Gemini fills unspecified nodes in
dependency order; an evaluation pass scores the completed graph before generation.

Refined ontology (v2): adds genre, pacing, weather, wardrobe, composition; edges now
carry a KIND — 'informs' (soft guidance), 'constrains' (limits valid values),
'requires' (child is meaningless without the parent).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class Source(str, Enum):
    USER = "USER"           # stated/strongly implied by the user (or a story-global)
    INFERRED = "INFERRED"   # filled by Gemini under constraints
    DEFAULT = "DEFAULT"     # deterministic default (technical nodes)
    OMITTED = "OMITTED"     # deliberately empty


# edge kinds
INFORMS = "informs"        # parent guides the child's ideal value
CONSTRAINS = "constrains"  # parent limits the child's valid values
REQUIRES = "requires"      # child is only meaningful if the parent is present


@dataclass
class PromptNode:
    key: str
    label: str
    category: str            # CONTENT | CAMERA | STYLE | AUDIO | CONSTRAINT | TECHNICAL
    tier: Tier
    description: str
    value: str | None = None
    source: Source | None = None
    confidence: float = 0.0
    rationale: str = ""

    @property
    def filled(self) -> bool:
        return bool(self.value) and self.source is not Source.OMITTED


# ── Requirement nodes (key, label, category, tier, description) ────────────────
_NODES: list[tuple] = [
    # CONTENT
    ("subject", "Subject", "CONTENT", Tier.REQUIRED,
     "The main character/object/focal point — who or what the video is about."),
    ("action", "Action", "CONTENT", Tier.REQUIRED,
     "What the subject does — ONE clear primary action."),
    ("setting", "Setting", "CONTENT", Tier.REQUIRED,
     "The location/environment where it takes place."),
    ("time_of_day", "Time of day", "CONTENT", Tier.RECOMMENDED,
     "Time of day / natural light condition (dawn, noon, dusk, night)."),
    ("weather", "Weather", "CONTENT", Tier.OPTIONAL,
     "Weather/atmospherics (rain, fog, snow, clear haze) — affects light, mood, ambient."),
    ("wardrobe", "Wardrobe / appearance", "CONTENT", Tier.OPTIONAL,
     "The subject's clothing and physical appearance — anchors identity and era."),
    ("pacing", "Pacing / energy", "CONTENT", Tier.OPTIONAL,
     "Energy of the shot (languid, steady, frenetic) — drives camera and cut rhythm."),
    # CAMERA
    ("shot_size", "Shot size", "CAMERA", Tier.RECOMMENDED,
     "Framing: wide / medium / close-up / extreme close-up."),
    ("camera_movement", "Camera movement", "CAMERA", Tier.RECOMMENDED,
     "Camera motion: static, slow dolly-in, tracking, crane, pan, handheld."),
    ("camera_angle", "Camera angle", "CAMERA", Tier.OPTIONAL,
     "Angle: eye-level, low, high, over-the-shoulder, Dutch tilt."),
    ("composition", "Composition", "CAMERA", Tier.OPTIONAL,
     "Framing rule: centered, rule-of-thirds, symmetrical, strong negative space."),
    ("lens", "Lens", "CAMERA", Tier.OPTIONAL,
     "Lens / focal length (24mm wide, 50mm normal, 85mm portrait)."),
    ("depth_of_field", "Depth of field", "CAMERA", Tier.OPTIONAL,
     "Focus depth: shallow (bokeh) or deep focus."),
    # STYLE
    ("genre", "Genre", "STYLE", Tier.RECOMMENDED,
     "Genre/register (noir, sci-fi, documentary, fantasy, commercial) — sets style/mood/music priors."),
    ("visual_style", "Visual style", "STYLE", Tier.RECOMMENDED,
     "Overall aesthetic (film noir, photorealistic, anime, claymation, 90s VHS)."),
    ("lighting", "Lighting", "STYLE", Tier.RECOMMENDED,
     "Lighting design (low-key chiaroscuro, soft golden hour, harsh fluorescent)."),
    ("color_palette", "Color palette", "STYLE", Tier.RECOMMENDED,
     "Color grade/palette (desaturated teal-orange, monochrome, warm pastels)."),
    ("film_stock", "Film stock / texture", "STYLE", Tier.OPTIONAL,
     "Medium/texture (35mm grain, digital clean, Super 8, black-and-white)."),
    ("mood", "Mood", "STYLE", Tier.RECOMMENDED,
     "Emotional tone (tense, melancholic, joyful, ominous, whimsical)."),
    # AUDIO
    ("dialogue", "Dialogue", "AUDIO", Tier.OPTIONAL,
     "Spoken lines in quotes with the speaker, if any speech occurs."),
    ("sfx", "Sound effects", "AUDIO", Tier.RECOMMENDED,
     "Key sound effects locked to on-screen events."),
    ("ambient", "Ambient sound", "AUDIO", Tier.RECOMMENDED,
     "Background ambient soundscape/bed."),
    ("music", "Music", "AUDIO", Tier.OPTIONAL,
     "Musical score/underscore, if any."),
    # CONSTRAINT
    ("negative_constraints", "Negative constraints", "CONSTRAINT", Tier.OPTIONAL,
     "What to AVOID, phrased as positive absence (e.g. 'an empty street with no cars')."),
    # TECHNICAL (filled deterministically from platform + Veo rules)
    ("platform", "Platform", "TECHNICAL", Tier.RECOMMENDED,
     "Target platform (youtube, tiktok, reels, generic) — drives aspect ratio."),
    ("aspect_ratio", "Aspect ratio", "TECHNICAL", Tier.REQUIRED,
     "16:9 (landscape) or 9:16 (vertical)."),
    ("resolution", "Resolution", "TECHNICAL", Tier.REQUIRED,
     "720p or 1080p (Veo 3.1 constraint: 1080p pairs with 8s)."),
    ("duration", "Duration", "TECHNICAL", Tier.REQUIRED,
     "Clip length in seconds: 4, 6, or 8."),
]

# ── Dependency edges (parent, child, kind) ────────────────────────────────────
_EDGES: list[tuple[str, str, str]] = [
    # genre priors
    ("genre", "visual_style", INFORMS), ("genre", "mood", INFORMS),
    ("genre", "color_palette", INFORMS), ("genre", "music", INFORMS),
    ("genre", "pacing", INFORMS),
    # setting derivatives
    ("setting", "time_of_day", INFORMS), ("setting", "weather", INFORMS),
    ("setting", "ambient", INFORMS), ("setting", "sfx", INFORMS),
    ("setting", "wardrobe", INFORMS),
    # weather / time -> light + atmosphere
    ("weather", "lighting", INFORMS), ("weather", "ambient", INFORMS),
    ("weather", "mood", INFORMS), ("time_of_day", "lighting", INFORMS),
    # style -> look
    ("visual_style", "lighting", INFORMS), ("visual_style", "color_palette", INFORMS),
    ("visual_style", "film_stock", INFORMS), ("visual_style", "depth_of_field", INFORMS),
    ("visual_style", "wardrobe", INFORMS), ("visual_style", "composition", INFORMS),
    ("visual_style", "negative_constraints", INFORMS),
    # mood -> look + feel
    ("mood", "lighting", INFORMS), ("mood", "color_palette", INFORMS),
    ("mood", "music", INFORMS), ("mood", "camera_angle", INFORMS),
    ("mood", "composition", INFORMS), ("mood", "pacing", INFORMS),
    ("mood", "negative_constraints", INFORMS),
    # subject
    ("subject", "shot_size", INFORMS), ("subject", "camera_angle", INFORMS),
    ("subject", "wardrobe", REQUIRES), ("subject", "dialogue", REQUIRES),
    # action
    ("action", "camera_movement", INFORMS), ("action", "shot_size", INFORMS),
    ("action", "sfx", INFORMS), ("action", "dialogue", INFORMS),
    ("action", "pacing", INFORMS),
    # pacing
    ("pacing", "camera_movement", INFORMS), ("pacing", "duration", INFORMS),
    # camera chain
    ("shot_size", "lens", INFORMS), ("shot_size", "composition", INFORMS),
    ("lens", "depth_of_field", INFORMS),
    # technical constraints
    ("platform", "aspect_ratio", CONSTRAINS), ("resolution", "duration", CONSTRAINS),
]

TECHNICAL_KEYS = {"platform", "aspect_ratio", "resolution", "duration"}
# nodes that define cross-shot consistency (seeded as story-globals in multi-shot)
GLOBAL_KEYS = ["genre", "visual_style", "mood", "color_palette", "film_stock", "setting"]


@dataclass
class PromptGraph:
    nodes: dict[str, PromptNode] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    @classmethod
    def build(cls) -> "PromptGraph":
        nodes = {
            k: PromptNode(key=k, label=lbl, category=cat, tier=tier, description=desc)
            for (k, lbl, cat, tier, desc) in _NODES
        }
        return cls(nodes=nodes, edges=list(_EDGES))

    # ── queries ──
    def node(self, key: str) -> PromptNode | None:
        return self.nodes.get(key)

    def parents(self, key: str) -> list[str]:
        return [p for (p, c, _k) in self.edges if c == key]

    def children(self, key: str) -> list[str]:
        return [c for (p, c, _k) in self.edges if p == key]

    def requires_parents(self, key: str) -> list[str]:
        return [p for (p, c, k) in self.edges if c == key and k == REQUIRES]

    def set(self, key: str, value: str, source: Source, confidence: float = 0.6,
            rationale: str = "") -> None:
        n = self.nodes.get(key)
        if n is not None:
            n.value, n.source, n.confidence, n.rationale = value, source, confidence, rationale

    def unfilled(self, keys: list[str] | None = None) -> list[str]:
        ks = keys if keys is not None else list(self.nodes)
        return [k for k in ks if not self.nodes[k].filled]

    def missing_required(self) -> list[str]:
        return [k for k, n in self.nodes.items() if n.tier is Tier.REQUIRED and not n.filled]

    def requires_violations(self) -> list[str]:
        """Filled nodes whose 'requires' parent is empty (logical defect)."""
        bad = []
        for k, n in self.nodes.items():
            if n.filled:
                for p in self.requires_parents(k):
                    if not self.nodes[p].filled:
                        bad.append(f"{k} requires {p}")
        return bad

    def topo_layers(self) -> list[list[str]]:
        indeg = {k: 0 for k in self.nodes}
        kids: dict[str, list[str]] = defaultdict(list)
        for p, c, _k in self.edges:
            kids[p].append(c)
            indeg[c] += 1
        layers: list[list[str]] = []
        frontier = sorted(k for k in self.nodes if indeg[k] == 0)
        while frontier:
            layers.append(frontier)
            nxt: list[str] = []
            for k in frontier:
                for c in kids[k]:
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        nxt.append(c)
            frontier = sorted(nxt)
        return layers

    def as_readable(self, only_filled: bool = True) -> str:
        lines = []
        for k, n in self.nodes.items():
            if only_filled and not n.filled:
                continue
            src = n.source.value if n.source else "-"
            lines.append(f"- {n.label} [{n.category}/{n.tier.value}, {src}]: {n.value}")
        return "\n".join(lines)

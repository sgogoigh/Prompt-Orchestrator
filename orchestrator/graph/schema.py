"""Scene Graph data model — the single source of truth for the orchestrator.

A typed, serializable graph of the video to be generated. Mirrors the GitNexus
philosophy: separate typed node kinds + one typed-edge set, every derived value
carrying provenance and confidence (the analog of GitNexus's confidence-scored
edges).

IMPORTANT: this graph is an *internal* representation. It is NEVER serialized to
Veo. `orchestrate/serializer.py` flattens it into cinematic natural language;
that prose is the only thing the model sees.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Provenance ────────────────────────────────────────────────────────────────
class Provenance(str, Enum):
    """Where an attribute value came from — lets the user override Gemini's fill-ins."""

    USER = "USER"            # stated/implied by the user prompt
    RETRIEVED = "RETRIEVED"  # bound from a canonical asset in the Store (continuity)
    INFERRED = "INFERRED"    # invented by a Gemini fill-in call


class Attribute(BaseModel):
    """A single filled value with provenance + confidence (0..1)."""

    value: str
    prov: Provenance = Provenance.INFERRED
    conf: float = Field(default=0.6, ge=0.0, le=1.0)


class NamedAttribute(Attribute):
    """An Attribute that carries its own key.

    Used for the open-ended entity attribute set. We use a *list* of these rather
    than a dict because the Gemini Developer API rejects `additionalProperties`
    (free-form maps) in a response_schema.
    """

    key: str


# ── Node types ──────────────────────────────────────────────────────────────--
class EntityType(str, Enum):
    CHARACTER = "character"
    OBJECT = "object"
    LOCATION = "location"
    PROP = "prop"
    VEHICLE = "vehicle"
    CROWD = "crowd"


class Entity(BaseModel):
    """A character / object / location / prop that can appear in shots."""

    id: str
    type: EntityType
    name: str
    canonical: bool = False  # reusable, continuity-tracked asset?
    attributes: list[NamedAttribute] = Field(default_factory=list)
    reference_image: Optional[str] = None   # asset:// path, for Veo Ingredients
    needs_reference: bool = False            # set by continuity when APPEARS_IN >= 2 shots
    embedding: Optional[list[float]] = None  # for retrieval (Stage 2)

    # ── attribute helpers (list is the storage; these give dict-like ergonomics) ──
    def attr(self, key: str) -> Optional[NamedAttribute]:
        return next((a for a in self.attributes if a.key == key), None)

    def attr_value(self, key: str, default: str = "") -> str:
        a = self.attr(key)
        return a.value if a else default

    def set_attr(self, attribute: NamedAttribute) -> None:
        """Upsert by key (later write wins)."""
        for i, a in enumerate(self.attributes):
            if a.key == attribute.key:
                self.attributes[i] = attribute
                return
        self.attributes.append(attribute)


class Style(BaseModel):
    """The global 'world bible' — one per generation. Filled by specify/style_bible.py."""

    id: str = "style_global"
    genre: Attribute | None = None
    era: Attribute | None = None
    palette: Attribute | None = None
    film_stock: Attribute | None = None
    lens_kit: Attribute | None = None
    lighting_philosophy: Attribute | None = None
    mood: Attribute | None = None


class Cinematography(BaseModel):
    shot_size: Attribute | None = None     # wide / medium / close-up / extreme close-up
    camera_move: Attribute | None = None   # dolly / tracking / crane / pan / POV / static
    lens: Attribute | None = None          # focal length + DoF
    angle: Attribute | None = None         # eye-level / low / high / OTS


class DialogueLine(BaseModel):
    speaker_id: str          # Entity.id
    line: str                # the spoken words (rendered in quotes by the serializer)


class Soundstage(BaseModel):
    dialogue: list[DialogueLine] = Field(default_factory=list)
    sfx: list[str] = Field(default_factory=list)       # event-locked sound effects
    ambient: Attribute | None = None                   # background bed
    music: Attribute | None = None


class Shot(BaseModel):
    """A unit of generation (or one timestamp segment within a generation)."""

    id: str
    order: int = 0
    duration_s: int = 8                     # 4 | 6 | 8
    cinematography: Cinematography = Field(default_factory=Cinematography)
    action: Attribute | None = None         # ONE clear primary action
    lighting: Attribute | None = None       # shot-specific lighting
    soundstage: Soundstage = Field(default_factory=Soundstage)
    emotion: Attribute | None = None

    @property
    def time_range(self) -> str:
        """Inclusive [start-end] label for timestamp prompting (filled by topology)."""
        return getattr(self, "_time_range", f"[00:00-00:{self.duration_s:02d}]")


class Scene(BaseModel):
    """A Leiden-style grouping of shots into a coherent scene."""

    id: str
    label: str = ""
    shot_ids: list[str] = Field(default_factory=list)


# ── Edges ───────────────────────────────────────────────────────────────────--
class EdgeType(str, Enum):
    APPEARS_IN = "APPEARS_IN"          # entity -> shot   (drives reference-image routing)
    LOCATED_IN = "LOCATED_IN"          # shot -> location
    INTERACTS_WITH = "INTERACTS_WITH"  # entity -> entity (blocking)
    NEXT = "NEXT"                      # shot -> shot     (narrative flow / the "process")
    TRANSITIONS_TO = "TRANSITIONS_TO"  # shot -> shot     (drives First+Last Frame)
    CONTINUES_FROM = "CONTINUES_FROM"  # shot -> shot     (drives Scene Extension)
    MEMBER_OF = "MEMBER_OF"            # shot -> scene
    STYLED_BY = "STYLED_BY"            # shot/scene -> style
    DERIVED_FROM = "DERIVED_FROM"      # entity -> entity (continuity: "same as")


class Edge(BaseModel):
    type: EdgeType
    src: str
    dst: str
    note: Optional[str] = None   # free-text basis (kept as a string, not a map)


# ── Generation intent ─────────────────────────────────────────────────────────
class GenerationIntent(BaseModel):
    shot_count_hint: int = 1
    total_duration_s: int = 8
    platform: str = "generic"          # youtube|tiktok|reels|generic -> aspect ratio
    aspect_ratio: str = "16:9"
    wants_dialogue: bool = False
    ambiguity_score: float = 0.5       # 0 = fully specified, 1 = one-liner; drives fill-in depth


# ── The graph ─────────────────────────────────────────────────────────────────
class SceneGraph(BaseModel):
    """The complete (or skeletal) scene representation passed through the pipeline."""

    user_prompt: str = ""
    intent: GenerationIntent = Field(default_factory=GenerationIntent)
    style: Style = Field(default_factory=Style)
    entities: list[Entity] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    # ── convenience accessors ──
    def entity(self, entity_id: str) -> Optional[Entity]:
        return next((e for e in self.entities if e.id == entity_id), None)

    def shot(self, shot_id: str) -> Optional[Shot]:
        return next((s for s in self.shots if s.id == shot_id), None)

    def entities_in_shot(self, shot_id: str) -> list[Entity]:
        ids = {e.src for e in self.edges
               if e.type == EdgeType.APPEARS_IN and e.dst == shot_id}
        return [e for e in self.entities if e.id in ids]

    def add_edge(self, type_: EdgeType, src: str, dst: str, note: str | None = None) -> None:
        self.edges.append(Edge(type=type_, src=src, dst=dst, note=note))

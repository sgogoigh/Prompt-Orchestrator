"""Stage 4b — flatten the Scene Graph into Veo-native cinematic natural language.

DETERMINISTIC by design (same graph -> same prompt; debuggable, diffable). This
is where the iron rule lives: we emit PROSE ONLY, using the 5-part formula
  [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
plus quoted dialogue and explicit SFX:/Ambient:/Music: cues. We NEVER emit JSON
or graph syntax — that is out-of-distribution for Veo and degrades output.
"""

from __future__ import annotations

from ..graph.schema import Attribute, EdgeType, Entity, SceneGraph, Shot
from ..graph import topology
from .strategy import Strategy, select_strategy


def _v(attr: Attribute | None, default: str = "") -> str:
    return attr.value if attr else default


def _join(parts: list[str]) -> str:
    return ", ".join(p.strip() for p in parts if p and p.strip())


def _entity_phrase(e: Entity) -> str:
    """A compact descriptive noun phrase for an entity, from its filled attributes."""
    desc = []
    for key in ("appearance", "age", "build", "wardrobe", "material"):
        val = e.attr_value(key)
        if val:
            desc.append(val)
    inner = _join(desc)
    return f"{e.name} ({inner})" if inner else e.name


def _style_clause(graph: SceneGraph) -> str:
    s = graph.style
    return _join([_v(s.palette), _v(s.film_stock), _v(s.lens_kit), _v(s.mood)])


def _cinematography_clause(shot: Shot) -> str:
    c = shot.cinematography
    return _join([_v(c.shot_size), _v(c.camera_move), _v(c.lens), _v(c.angle)])


def _soundstage_clause(graph: SceneGraph, shot: Shot) -> str:
    ss = shot.soundstage
    bits: list[str] = []
    for d in ss.dialogue:
        speaker = graph.entity(d.speaker_id)
        who = speaker.name if speaker else "A voice"
        bits.append(f'{who} says, "{d.line}"')          # dialogue in quotes
    if ss.sfx:
        bits.append("SFX: " + "; ".join(ss.sfx))
    if ss.ambient:
        bits.append("Ambient: " + ss.ambient.value)
    if ss.music:
        bits.append("Music: " + ss.music.value)
    return " ".join(bits)


def serialize_shot(graph: SceneGraph, shot: Shot) -> str:
    """One shot -> a single 5-part natural-language sentence (+ soundstage)."""
    subjects = _join([_entity_phrase(e) for e in graph.entities_in_shot(shot.id)])
    # context = location + lighting
    location = next(
        (graph.entity(e.dst) for e in graph.edges
         if e.type == EdgeType.LOCATED_IN and e.src == shot.id and graph.entity(e.dst)),
        None,
    )
    context = _join([
        _entity_phrase(location) if location else "",
        _v(shot.lighting),
    ])

    visual = _join([
        _cinematography_clause(shot),
        subjects,
        _v(shot.action),
        context,
        _style_clause(graph),
    ])
    visual = visual[:1].upper() + visual[1:] + "." if visual else ""
    sound = _soundstage_clause(graph, shot)
    return f"{visual} {sound}".strip()


def serialize(graph: SceneGraph) -> str:
    """Full graph -> the final Veo prompt string, shaped by the chosen strategy."""
    topology.assign_time_ranges(graph)
    shots = topology.ordered_shots(graph)
    strat = select_strategy(graph)

    if strat == Strategy.TIMESTAMP and len(shots) > 1:
        # multi-shot in one generation: prefix each with its [mm:ss-mm:ss] range
        return "\n".join(f"{s.time_range} {serialize_shot(graph, s)}" for s in shots)

    # SINGLE / SCENE_EXTENSION / FIRST_LAST all serialize shot-by-shot;
    # veo/client.py decides how many calls to make and how to chain them.
    return serialize_shot(graph, shots[0]) if shots else graph.user_prompt


def serialize_per_shot(graph: SceneGraph) -> list[str]:
    """One prompt string per shot — for SCENE_EXTENSION chaining / FIRST_LAST."""
    topology.assign_time_ranges(graph)
    return [serialize_shot(graph, s) for s in topology.ordered_shots(graph)]

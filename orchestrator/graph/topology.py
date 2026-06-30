"""Pure graph-topology helpers over a SceneGraph.

No I/O, no model calls — just structural queries the rest of the pipeline relies
on (the analog of GitNexus's graph traversals).
"""

from __future__ import annotations

from .schema import EdgeType, SceneGraph


def appearance_counts(graph: SceneGraph) -> dict[str, int]:
    """entity_id -> number of distinct shots it APPEARS_IN."""
    counts: dict[str, int] = {}
    for e in graph.edges:
        if e.type == EdgeType.APPEARS_IN:
            counts[e.src] = counts.get(e.src, 0) + 1
    return counts


def mark_recurring_entities(graph: SceneGraph, min_appearances: int = 2) -> list[str]:
    """Flag entities that appear in >= N shots as needing a reference image.

    This is the key continuity signal: anything recurring must be locked with a
    Veo Ingredients reference image (analog of GitNexus's call graph driving
    safe `rename`). Returns the list of flagged entity ids.
    """
    counts = appearance_counts(graph)
    flagged: list[str] = []
    for ent in graph.entities:
        if counts.get(ent.id, 0) >= min_appearances:
            ent.needs_reference = True
            flagged.append(ent.id)
    return flagged


def ordered_shots(graph: SceneGraph):
    """Shots in narrative order: follow NEXT edges if present, else by `order`."""
    succ = {e.src: e.dst for e in graph.edges if e.type == EdgeType.NEXT}
    preds = set(succ.values())
    heads = [s for s in graph.shots if s.id not in preds]
    if not succ or not heads:
        return sorted(graph.shots, key=lambda s: s.order)

    by_id = {s.id: s for s in graph.shots}
    start = min(heads, key=lambda s: s.order)
    seq, cur, seen = [], start.id, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in by_id:
            seq.append(by_id[cur])
        cur = succ.get(cur)
    # append any shots not reached by NEXT chain
    seq += [s for s in sorted(graph.shots, key=lambda s: s.order) if s.id not in seen]
    return seq


def assign_time_ranges(graph: SceneGraph) -> None:
    """Stamp inclusive [mm:ss-mm:ss] ranges onto shots for timestamp prompting."""
    t = 0
    for shot in ordered_shots(graph):
        start, end = t, t + shot.duration_s
        object.__setattr__(shot, "_time_range", f"[{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}]")
        t = end


def total_duration(graph: SceneGraph) -> int:
    return sum(s.duration_s for s in graph.shots)


def has_edge_type(graph: SceneGraph, t: EdgeType) -> bool:
    return any(e.type == t for e in graph.edges)

"""Stage 2 — Asset & Continuity Store: retrieval + persistence (Graph-RAG).

This is the GitNexus role: persist scene graphs + canonical entities so later
generations stay consistent ("the same detective"), and retrieve them with a
hybrid name + vector match fused by Reciprocal Rank Fusion (RRF).

Current scaffold: a JSON-file-backed store with name matching. Vector retrieval
and RRF are stubbed (P3).
"""

from __future__ import annotations

import json
import os

from ..config import settings
from ..graph.schema import EdgeType, Entity, Provenance, SceneGraph

RRF_K = 60  # standard RRF constant (same as GitNexus hybrid search)


class AssetStore:
    def __init__(self, path: str | None = None):
        self.path = path or settings.scene_graph_db
        self.canonical: list[Entity] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.canonical = [Entity.model_validate(e) for e in data.get("canonical_entities", [])]

    def save_graph(self, graph: SceneGraph) -> None:
        """Persist canonical entities + the final graph (the 'use it later' step)."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        for ent in graph.entities:
            if ent.canonical or ent.needs_reference:
                # lock attributes as RETRIEVED-grade for next time
                ent.attributes = [
                    a.model_copy(update={"prov": Provenance.RETRIEVED})
                    for a in ent.attributes
                ]
                self._upsert_canonical(ent)
        payload = {"canonical_entities": [e.model_dump() for e in self.canonical]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # TODO: also persist the full graph + clip metadata (SynthID) per project.

    def _upsert_canonical(self, ent: Entity) -> None:
        for i, e in enumerate(self.canonical):
            if e.name.lower() == ent.name.lower():
                self.canonical[i] = ent
                return
        ent.canonical = True
        self.canonical.append(ent)

    # ── retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, graph: SceneGraph) -> SceneGraph:
        """Bind skeletal entities to canonical assets when the user implies continuity.

        Only runs when ENABLE_RETRIEVAL. For each skeletal entity, try an exact/alias
        name match (and, TODO, vector similarity fused via RRF); above threshold,
        copy the canonical attributes + reference image and add a DERIVED_FROM edge.
        """
        if not settings.enable_retrieval or not self.canonical:
            return graph
        for ent in graph.entities:
            match = self._best_match(ent)
            if match is None:
                continue
            for a in match.attributes:
                ent.set_attr(a)
            ent.reference_image = match.reference_image
            ent.canonical = True
            graph.add_edge(EdgeType.DERIVED_FROM, ent.id, match.id, note="name-match")
        return graph

    def _best_match(self, ent: Entity) -> Entity | None:
        # P3: replace with hybrid name + embedding cosine, fused by RRF (K=60).
        for c in self.canonical:
            if c.name.lower() == ent.name.lower():
                return c
        return None


store = AssetStore()

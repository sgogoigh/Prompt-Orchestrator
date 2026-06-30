# Veo 3.1 Prompt Orchestrator — Architecture Plan

> A **scene-graph-driven prompt orchestration system** that turns a sparse user prompt into a
> fully-specified, continuity-aware, Veo-native generation request. It borrows the **GitNexus**
> philosophy — *model the thing as a typed graph of entities and relationships, persist it, and
> retrieve from it* — and applies it to video. **Gemini** API calls do the heavy "fill in the
> world" work; the graph is persisted so later generations stay consistent; the final output is
> always **natural language** (never graph syntax), because that is the only thing Veo eats.

**Target pipeline (as specified by the user):**
```
user prompt → structured graph → retrieval (if needed) → specification → prompt orchestrated → Veo 3.1 → output
                     │                   │                     │                                          │
                     └──────────── persisted Scene-Graph + Asset/Continuity store ─────────────── feedback loop
```

**Stack:** Python 3.10+, `google-genai` (Gemini text/image + Veo), `pydantic` v2 (schema +
structured outputs), `python-dotenv`.

---

## 0. Design principles — the GitNexus → Veo mapping

| GitNexus concept | Orchestrator analog | File(s) |
|---|---|---|
| Typed code knowledge graph (symbols + edges) | **Scene Graph** (entities + relationships + shots) | `graph/schema.py` |
| Confidence-scored edges | **Provenance + confidence per attribute** (`USER`/`RETRIEVED`/`INFERRED`) | `graph/schema.py` |
| LadybugDB persistence + registry | **Asset & Continuity Store** (graph JSON + vector index + blobs) | `store/store.py` |
| Communities (Leiden) | **Scene / sequence grouping** | `graph/schema.py` (`Scene`) |
| Processes (execution flows, BFS) | **Narrative beat sequence** (`NEXT` shot edges) | `graph/topology.py` |
| Hybrid retrieval (BM25 + vector + RRF) | **Entity retrieval** (name + vector, RRF-fused) | `store/store.py` |
| Ingestion phase DAG | **Gemini enrichment DAG** | `specify/pipeline.py` |
| MCP tools = thin shell over graph logic | **Veo API = thin renderer** over the graph | `veo/client.py` |

**Iron rule:** the Scene Graph is an *internal* representation. Veo 3.1 was trained on
Gemini-written **cinematic natural language**, so the orchestrator's final job is to *flatten*
the graph into prose (5-part formula) and route structure into Veo's **native** features
(Ingredients, First/Last Frame, Scene Extension). **Graph/JSON syntax is never sent to Veo.**

---

## 1. Pipeline stages

### Stage 1 — Decompose  (`decompose/decompose.py`)
One Gemini Flash call, structured output → a **skeletal Scene Graph**: entities, relationships,
draft shot list, genre/style stub, and a `GenerationIntent` (shot count, duration, platform→aspect,
`ambiguity_score`). Only captures what is stated/implied; everything inferred is tagged
`prov=INFERRED`. `ambiguity_score` drives how aggressive Stage 3 fill-in is.

### Stage 2 — Retrieve if needed  (`store/store.py`)
Graph-RAG. Skipped unless the prompt references something that must stay consistent ("the same
detective", a named recurring character/brand, a saved style bible). Hybrid match (exact name +
vector similarity, fused with RRF, K=60); above threshold → **bind** the skeletal entity to the
stored canonical one (copy locked attributes + reference image, add `DERIVED_FROM` edge).

### Stage 3 — Specify  (`specify/*`) — the "fill in EVERYTHING" engine
A **DAG of Gemini calls**, parallelized where independent:
```
        ┌─ 3a style_bible ─┐
skeletal│                  ├─► 3c shot_design ─► 3e continuity ─► 3f reference_images
 graph ─┤                  │      (parallel)      reconcile         (parallel, recurring)
        └─ 3b entity_enrich ┘   (+3d soundstage)
```
| Phase | Module | Model | Fills in |
|---|---|---|---|
| 3a | `style_bible.py` | Pro/Flash | palette, film stock, lens kit, lighting philosophy, mood |
| 3b | `entity_enrich.py` | Flash | physical desc, wardrobe, age, materials, features |
| 3c | `shot_design.py` | Pro | shot size, camera move, lens, angle, blocking, lighting, emotion, timing |
| 3d | `soundstage.py` | Flash | dialogue lines, `SFX:` cues, ambient bed, music |
| 3e | `continuity.py` | Pro | resolve contradictions, enforce style, flag entities in ≥2 shots as `needs_reference` |
| 3f | `reference_images.py` | Gemini Image | one canonical reference PNG per recurring entity → Ingredients |

Many small focused calls (not one mega-prompt) → better quality, parallelism, cacheable style
bible, per-value provenance. Output: the **fully-specified Scene Graph**.

### Stage 4 — Orchestrate  (`orchestrate/*`)
- `strategy.py` — pick the Veo strategy from graph topology:
  | Graph signal | Veo strategy |
  |---|---|
  | 1 shot ≤ 8 s | single T2V/I2V |
  | multi-shot, Σ ≤ 8 s | **timestamp prompting** in one generation |
  | Σ > 8 s + `CONTINUES_FROM` | **Scene Extension** chain (last ~1 s / 24 frames) |
  | `TRANSITIONS_TO` w/ start+end | **First & Last Frame** |
  | entity `APPEARS_IN ≥ 2` shots | attach reference image via **Ingredients** (≤ 3) |
- `serializer.py` — deterministic graph → 5-part NL prompt. Dialogue in quotes; `SFX:`/`Ambient:`
  cues; negatives phrased positively; one primary action per shot; multi-shot → `[00:00-00:02]…`
  timestamp blocks. **Prose only — never JSON.**
- `critique.py` — optional Gemini self-critique against the 5-part formula + Veo's known limits
  (no on-screen text, fragile hands/faces past ~3 s, short-dialogue inconsistency); revises.

### Stage 5 — Veo dispatch + feedback  (`veo/client.py`, `qc/qc.py`)
1. Dispatch per strategy (single / timestamp / extension chain / frames).
2. Draft on **Veo Fast**, promote chosen take to **Standard** (+ 4K upscale) on approval.
3. QC (optional): Gemini **vision** checks adherence/artifacts → score → accept or regenerate
   (bounded retries).
4. **Persist** clip + final graph + reference images back to the Store (SynthID recorded) so
   "the same detective" is retrievable next time.

---

## 2. Scene Graph data model (see `graph/schema.py`)

- **Nodes:** `Entity` (character/object/location/prop/vehicle/crowd), `Location`, `Style`
  (global world bible), `Shot` (with `Cinematography` + `Soundstage`), `Scene`.
- **Attribute** = `{ value, prov: USER|RETRIEVED|INFERRED, conf: float }` — provenance + confidence
  on every filled value (GitNexus's confidence-scored edges).
- **Edges (typed):** `APPEARS_IN`, `LOCATED_IN`, `INTERACTS_WITH`, `NEXT`, `TRANSITIONS_TO`,
  `CONTINUES_FROM`, `MEMBER_OF`, `STYLED_BY`, `DERIVED_FROM`.
- **Key signal:** any entity with `APPEARS_IN ≥ 2` shots is auto-promoted to "needs reference
  image" → Veo Ingredients keeps it consistent (analog of GitNexus's call graph driving `rename`).

---

## 3. Module map

```
orchestrator/
├── config.py            # env + model IDs + defaults
├── gemini_client.py     # shared google-genai wrapper (text JSON, image, helpers)
├── api.py               # generate(user_prompt, opts) — wires all 5 stages
├── graph/
│   ├── schema.py        # pydantic Scene Graph (the data model) — backbone
│   └── topology.py      # APPEARS_IN fan-out, NEXT ordering, strategy inputs
├── decompose/decompose.py        # Stage 1
├── store/store.py                # Stage 2 retrieval + persistence
├── specify/
│   ├── pipeline.py      # Stage 3 DAG runner (parallel)
│   ├── style_bible.py · entity_enrich.py · shot_design.py
│   ├── soundstage.py · continuity.py · reference_images.py
├── orchestrate/
│   ├── strategy.py · serializer.py · critique.py
├── veo/client.py                 # Stage 5 dispatch
└── qc/qc.py                      # Stage 5 vision QC loop
examples/run_noir.py              # end-to-end worked example
```

---

## 4. Cross-cutting concerns
- **Provenance & user control** — every inferred value is tagged + confidence-scored; the
  fully-specified graph is the human-in-the-loop edit surface before Stage 4.
- **Cost/latency** — Gemini Flash fill-ins are cheap + parallel; cache the style bible; draft on
  Veo Fast, finalize on Standard. Budget is dominated by Veo seconds, not Gemini tokens.
- **Determinism** — Stage 4 serialization is deterministic (same graph → same prompt). Only
  Gemini stages are stochastic.
- **Guardrails** — serializer can't emit graph syntax (template-only NL); self-critique pre-empts
  Veo limitations; continuity handled by reference images + extension, not text alone; SynthID
  recorded on every stored clip.

---

## 5. Build phases → see [CHECKLIST.md](CHECKLIST.md)
P0 skeleton · P1 fill-in engine · P2 consistency · P3 persistence/RAG · P4 long-form · P5 QC loop.

*Design intent: the orchestrator is to Veo what GitNexus is to a coding agent — a structured,
persistent, retrieval-backed context engine that makes a powerful but context-blind generator
reliable. The graph is the brain; Gemini fills the world; Veo only renders; natural language is
the contract between them.*

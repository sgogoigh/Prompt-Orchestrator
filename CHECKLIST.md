# Build Checklist — Veo 3.1 Prompt Orchestrator

Legend: `[ ]` todo · `[~]` stubbed (signature + docstring, not implemented) · `[x]` done

> Current state: **P0–P2 verified against the live API.** `python -m examples.run_noir` (dry run)
> runs the full Gemini pipeline end-to-end: decompose -> retrieve -> specify (style/entity/shot/
> continuity) -> reference-image generation -> orchestrated Veo prompt. Models pinned to
> `gemini-3.5-flash` / `gemini-3.1-pro-preview` / `gemini-3.1-flash-image` (verified accessible).
> The Veo dispatch path is fixed + validated against the SDK surface but **not yet run** (a real
> `--generate` call spends Veo credits). Store retrieval (vector/RRF), scene-extension chaining,
> and the QC loop remain stubbed.

### Verified working (live)
- [x] Key loads from `.env`; account has Veo 3.1 + Gemini 3.x access (checked via models.list)
- [x] Schema is Gemini-structured-output safe (no `additionalProperties`; keyed-list attributes)
- [x] Stage 1 decompose -> skeletal graph (real call)
- [x] Stage 3 specify: style bible + per-entity + per-shot + continuity (real calls, parallel)
- [x] Stage 3f reference image generated via `gemini-3.1-flash-image`
- [x] Stage 4 serialize + strategy selection -> rich 5-part Veo prompt
- [x] Veo client arg names fixed: `VideoGenerationReferenceImage` (ASSET) wrapping, `Image.from_file`,
      `duration_seconds`, `files.download`/`Video.save` — all confirmed against the SDK
- [x] Dropped Vertex-only `generate_audio` flag (Developer API rejects it; audio is native by default)
- [x] **Veo 3.1 generation confirmed end-to-end** -> `assets/clips/*.mp4` (14.5 MB, valid MP4 header)

---

## P0 — Skeleton (end-to-end on one shot, no fill-in)
- [x] Project scaffold, dirs, `.env` / `.env.example`, `requirements.txt`
- [x] `config.py` — load env, model IDs, defaults
- [x] `graph/schema.py` — Scene Graph pydantic models (Entity/Location/Style/Shot/Scene/Edge)
- [x] `gemini_client.py` — shared SDK wrapper (text-JSON, image, vision)
- [~] `decompose/decompose.py` — user prompt → skeletal graph (prompt + schema present)
- [x] `orchestrate/serializer.py` — graph → 5-part NL prompt (deterministic)
- [x] `orchestrate/strategy.py` — choose Veo strategy from topology
- [x] `veo/client.py` — single T2V generation + poll + download (verified live)
- [x] `api.py` — wire stages 1→4→5
- [x] **Milestone:** `examples/run_noir.py --generate` produced a real 14.5 MB MP4 (Veo 3.1 Fast)

## P1 — Fill-in engine (Stage 3)
- [~] `specify/style_bible.py` — global world bible (3a)
- [~] `specify/entity_enrich.py` — per-entity enrichment (3b)
- [~] `specify/shot_design.py` — per-shot cinematography (3c)
- [~] `specify/soundstage.py` — dialogue / SFX / ambient / music (3d)
- [~] `specify/pipeline.py` — DAG runner, parallel fan-out, style-bible caching
- [ ] Provenance tagging (`INFERRED` + confidence) on every filled value
- [ ] `ambiguity_score` throttles fill-in depth
- [ ] **Milestone:** terse prompt → richly-specified graph

## P2 — Consistency (reference images + multi-shot)
- [~] `specify/continuity.py` — contradiction reconcile + flag recurring entities (3e)
- [~] `specify/reference_images.py` — Gemini image gen per recurring entity (3f)
- [ ] `graph/topology.py` — `APPEARS_IN ≥ 2` → `needs_reference`
- [ ] Ingredients routing in `veo/client.py` (attach ≤ 3 reference images)
- [ ] Timestamp prompting for multi-shot ≤ 8 s in `serializer.py`
- [ ] **Milestone:** same character consistent across 2 shots

## P3 — Persistence / Retrieval (Stage 2)
- [~] `store/store.py` — Asset & Continuity Store (graph JSON + blobs)
- [ ] Entity embeddings + vector index
- [ ] Hybrid retrieval (name + vector, RRF) + binding via `DERIVED_FROM`
- [ ] Persist final graph + clip + refs after generation
- [ ] **Milestone:** "the same detective" retrieves the prior canonical entity

## P4 — Long-form  ✅ (verified live)
- [x] Scene-extension chaining in `veo/client.py` — via **last-frame chaining**, NOT native
      `video=` extension. **Finding: native scene extension is NOT supported on the Gemini
      Developer API for Veo 3.1** ("Your use case is currently not supported"); it's a
      Vertex/Flow feature. Workaround: extract clip N's last frame (OpenCV) → feed as clip N+1's
      first frame (image-to-video, 1080p-capable).
- [x] First & Last Frame transitions (`generate_transition`) + endpoint-frame gen in `api.py`
- [x] `_normalize_duration` — enforce valid Veo pairings (1080p→8s; 720p→4/6/8)
- [x] **Milestone:** 2-shot chain rendered = ~16 s continuous 1080p narrative, character carried
      across the seam (chain0 + chain1, 1920x1080 @24fps, 8 s each). Scales to 60 s+ with more shots.
- [ ] Follow-up: optional audio continuity (native extension has it; last-frame chaining does not —
      each clip generates fresh audio at the seam)
- [ ] Follow-up: 1080p/4K upscale pass (Veo upscaling) if a 720p path is ever used

## P5 — Closing the loop (QC)
- [~] `qc/qc.py` — Gemini-vision adherence/artifact scoring
- [ ] Bounded auto-regeneration on low score (`MAX_QC_RETRIES`)
- [ ] Veo Fast draft → Standard promotion + 4K upscale on approval
- [ ] **Milestone:** auto-reject + retry a failed take without human input

---

## Cross-cutting (any phase)
- [ ] `orchestrate/critique.py` — self-critique against Veo limitations
- [ ] Cost/latency telemetry (Gemini tokens, Veo seconds)
- [ ] Unit tests: serializer determinism, strategy selection, schema validation
- [ ] Structured logging (`rich`)
- [ ] Error handling: Gemini JSON parse retries, Veo polling timeouts, rate limits

## Setup (do this first)
- [ ] `python -m venv venv && source venv/Scripts/activate` (Windows: `venv\Scripts\activate`)
- [ ] `pip install -r requirements.txt`
- [ ] Paste your key into `.env` → `GEMINI_API_KEY=...`
- [ ] `python -m examples.run_noir`

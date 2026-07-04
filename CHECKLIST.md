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

## P5 — Closing the loop (QC)  ✅ (core verified live; regen path wired)
- [x] `qc/qc.py` — Gemini-vision QC via Files API (`analyze_video`): scores adherence, subject
      presence, artifacts, lip-sync. **Verified discriminating live**: correct prompt → 1.0/accept,
      wrong prompt → 0.0/reject; and across two real clips 1.0 vs 0.5 → correct pick.
- [x] `passes_gate` + `pick_best` — best-of-N selection; gates ONLY on the trustworthy signals
      (adherence + subject). Artifacts/lip-sync are advisory (vision models unreliable there).
- [x] Bounded auto-regeneration (`MAX_QC_RETRIES`): single/timestamp path does best-of-N over Fast
      candidates + re-roll on hard-gate fail; `generate_chain` gates each clip before it seeds the
      next (a broken clip never poisons the chain).
- [x] **Milestone:** auto-reject + re-pick without human input — proven (rejected 0.5 clip, picked 1.0).
- [~] Fast draft → "Standard promotion" = re-render winning prompt on Standard model (a NEW take,
      not an upscale). Wire as opt-in if desired.
- [ ] ~~4K upscale on approval~~ — **not available on the Gemini Developer API** (no video-upscale
      method; Vertex/Flow only). `upscale_image` exists but is images-only.

### P5 upside verdict
Significant upside for **(a) gross-failure gating** and **(b) best-of-N selection over cheap Fast
candidates** — especially in chains, where a bad clip cascades. Marginal/negative for subtle-artifact
or lip-sync gating (false-reject risk → wasted re-rolls). QC as a *selector* (near-zero extra cost)
beats QC as a blind *re-roll trigger* (unbounded cost). Not live-tested: the paid regen/best-of-N Veo
runs (they reuse the already-proven `veo_generate`); only the QC decision logic was exercised (free).

---

## Prompt Requirement Graph + core eval-gate  ✅ (verified live, no Veo spend)
The thing the project was really after: evaluate/complete the PROMPT before generation.
- [x] `graph/prompt_graph.py` — requirement ontology (23 nodes, tiers, 25 dependency edges, topo layers)
- [x] `graph/prompt_fill.py` — MAP (user) + dependency-ordered FILL (parents constrain children) + deterministic technical
- [x] `graph/prompt_eval.py` — pre-generation prompt scoring (coherence/specificity/completeness + `invention_ratio` + defect flag)
- [x] `graph/prompt_serialize.py` — graph → 5-part NL prompt + API params
- [x] `graph/prompt_flow.py` — `build_prompt(idea)` runs all five stages
- [x] `api.generate_gated()` — wires the graph into the core with an EVAL-GATE (refuse / revise / warn)
- [x] Demo `examples/run_promptgraph.py` (build+eval) and `examples/run_gated.py` (the gate)
- **Finding:** scoring `completeness` is degenerate post-fill (the fill makes almost everything
  complete), so the gate leans on **coherence + specificity + real defects (conflicts / missing-
  required)**, with `invention_ratio` as a transparency signal. Verified: refused a weak prompt
  (0.767, invention 0.895) with no spend; auto-revised a rich prompt to 0.982.

### Refinement pass ✅ (verified live, no Veo spend)
- [x] **Ontology refined** — 23→**28 nodes** (added genre, pacing, weather, wardrobe, composition),
      25→**44 edges** now carrying a KIND: `informs` / `constrains` / `requires`.
- [x] **`requires` logic is a hard defect** (e.g. dialogue with no subject); soft coherence tensions
      only penalize the score. Fixes over-strict gating with the richer ontology (barista story:
      all shots defects=False, gate passes; hiker story: worst 0.59 → gated).
- [x] **Invention-ratio policy** — `generate_gated(max_invention, confirm)`: if the graph invented
      more than the threshold, returns `needs_confirmation` + the `inferred_fields` list (confirm
      before spending) instead of silently generating an all-invented film.
- [x] **Multi-shot** — `graph/prompt_story.py` (`plan_story` → per-shot graphs SEEDED with shared
      globals for cross-shot consistency → per-shot eval) + `api.generate_story()` gates on the
      WORST shot + AVERAGE invention, then dispatches to `generate_chain`. Demo: `examples/run_story.py`.
      Verified: 4-shot barista story, consistent macro-coffee look across shots, gate passed.
- [ ] Optional next: cross-shot narrative-continuity Gemini check; live chained story render.

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

# Prompt-Orchestrator

Running a prompt generator to enhance the performance of models to generate video — in this case
**Veo 3.1** — using **graph structures** for implementation and structure.

It takes a sparse user prompt, builds a typed **Scene Graph**, uses **Gemini** to fill in every
unspecified detail (lighting, wardrobe, lens, sound…), persists the graph for continuity, then
flattens it into a Veo-native cinematic prompt and renders. Think of it as a **GitNexus-style
context engine for video generation**: the graph is the brain, Gemini fills the world, Veo only
renders, and natural language is the contract between them.

```
user prompt → structured graph → retrieval (if needed) → specification → orchestrated prompt → Veo 3.1 → output
```

## Quickstart

```bash
# 1. install
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 2. add your key
#    edit .env  ->  GEMINI_API_KEY=...   (get one at https://aistudio.google.com/apikey)

# 3. run the worked example
python -m examples.run_noir
```

## Documentation
- **[PLAN.md](PLAN.md)** — full architecture: stages, Scene Graph model, GitNexus->Veo mapping.
- **[CHECKLIST.md](CHECKLIST.md)** — build status and the P0–P5 roadmap.

## Layout
```
orchestrator/
├── config.py · gemini_client.py · api.py
├── graph/        schema.py · topology.py          # the Scene Graph (data model)
├── decompose/    decompose.py                      # Stage 1: prompt -> skeletal graph
├── store/        store.py                          # Stage 2: retrieval + persistence
├── specify/      style_bible · entity_enrich · shot_design · soundstage · continuity · reference_images
├── orchestrate/  strategy · serializer · critique  # Stage 4: graph -> Veo prompt
├── veo/          client.py                          # Stage 5: dispatch to Veo 3.1
└── qc/           qc.py                              # Stage 5: vision QC loop
examples/run_noir.py
```

> Status: **scaffold** — see [CHECKLIST.md](CHECKLIST.md). The data model, config, Gemini wrapper,
> serializer, and strategy logic are implemented; Gemini-call stages carry real prompt templates
> and need wiring/tuning; Veo dispatch and the store need full implementations.

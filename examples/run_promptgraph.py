"""Demo: build + evaluate a Veo prompt from a sparse idea via the Prompt Requirement Graph.

No Veo spend — this only exercises the graph flow (Gemini text calls) and prints:
  - the filled requirement graph (value + source per node)
  - the pre-generation prompt evaluation
  - the final natural-language Veo prompt + API params

Run:
    python -m examples.run_promptgraph
    python -m examples.run_promptgraph "a chef plating dessert in a busy kitchen"
"""

from __future__ import annotations

import sys

from orchestrator.graph.prompt_flow import build_prompt

DEFAULT = "a detective in his office, noir"


def main() -> None:
    idea = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(f"\n=== USER PROMPT ===\n{idea}\n")

    b = build_prompt(idea)

    print("=== FILLED REQUIREMENT GRAPH ===")
    for cat in ("CONTENT", "CAMERA", "STYLE", "AUDIO", "CONSTRAINT", "TECHNICAL"):
        rows = [n for n in b.graph.nodes.values() if n.category == cat and n.filled]
        if not rows:
            continue
        print(f"  [{cat}]")
        for n in rows:
            src = n.source.value if n.source else "-"
            print(f"    {n.label:18} ({src:8}) {n.value}")

    if b.evaluation:
        e = b.evaluation
        print("\n=== PROMPT EVALUATION (pre-generation) ===")
        print(f"  overall      : {e.overall}")
        print(f"  completeness : {e.completeness}")
        print(f"  coherence    : {e.coherence}")
        print(f"  specificity  : {e.specificity}")
        if e.missing_required:
            print(f"  MISSING REQUIRED: {e.missing_required}")
        if e.conflicts:
            print(f"  conflicts    : {e.conflicts}")
        if e.weak_fields:
            print(f"  weak fields  : {e.weak_fields}")
        if e.suggestions:
            print(f"  suggestions  : {e.suggestions}")

    print(f"\n=== VEO PROMPT ===\n{b.veo_prompt}\n")
    print(f"=== API PARAMS ===\n{b.api_params}\n")


if __name__ == "__main__":
    main()

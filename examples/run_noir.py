"""Worked example — the film-noir scene from PLAN.md.

Run:
    python -m examples.run_noir            # dry run: prints the orchestrated prompt (no Veo spend)
    python -m examples.run_noir --generate # full run: also calls Veo 3.1 and downloads the clip

Requires GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import sys

from orchestrator.api import generate

PROMPT = "a detective meets a mysterious woman in his office, film noir, two shots, for YouTube"


def main() -> None:
    do_generate = "--generate" in sys.argv
    print(f"\n=== USER PROMPT ===\n{PROMPT}\n")

    result = generate(PROMPT, dry_run=not do_generate)

    print(f"=== STRATEGY ===\n{result.strategy}\n")
    print(f"=== ENTITIES ({len(result.graph.entities)}) ===")
    for e in result.graph.entities:
        flag = " [reference]" if e.needs_reference else ""
        print(f"  - {e.name} ({e.type.value}){flag}")
    if result.per_shot_prompts:
        print(f"\n=== ORCHESTRATED VEO PROMPTS ({len(result.per_shot_prompts)} shots, chained) ===")
        for i, p in enumerate(result.per_shot_prompts):
            print(f"\n  [shot {i + 1}]  {p}")
        print()
    else:
        print(f"\n=== ORCHESTRATED VEO PROMPT ===\n{result.veo_prompt}\n")
    if result.reference_images:
        print(f"=== REFERENCE IMAGES (Ingredients) ===\n  " + "\n  ".join(result.reference_images) + "\n")
    if result.clips:
        print(f"=== OUTPUT CLIPS ===\n  " + "\n  ".join(result.clips) + "\n")
    elif not do_generate:
        print("(dry run -- re-run with --generate to call Veo 3.1)\n")


if __name__ == "__main__":
    main()

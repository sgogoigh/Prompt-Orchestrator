"""Multi-shot demo: idea -> planned shots (shared globals) -> per-shot prompts -> gate.

Dry by default (no Veo spend) — shows the planned shots, each shot's requirement-graph
prompt + eval, cross-shot globals, and the story-level gate decision. Add --generate to
render the chain (last-frame chaining).

Run:
    python -m examples.run_story
    python -m examples.run_story "a barista makes latte art, a customer smiles, they clink cups"
    python -m examples.run_story --generate
"""

from __future__ import annotations

import sys

from orchestrator.api import generate_story

DEFAULT = ("a lone hiker reaches a mountain ridge at dawn, gazes at the valley below, "
           "then raises her arms in triumph")


def main() -> None:
    gen = "--generate" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--generate"]
    idea = args[0] if args else DEFAULT

    print(f"\n=== STORY IDEA ===\n{idea}\n")
    r = generate_story(idea, dry_run=not gen)

    print(f"=== STORY-LEVEL GATE ===")
    print(f"  worst_overall={r.worst_overall}  avg_invention={r.avg_invention}")
    print(f"  gated={r.gated}  needs_confirmation={r.needs_confirmation}")
    if r.gate_reason:
        print(f"  reason: {r.gate_reason}")

    print(f"\n=== SHOTS ({len(r.shot_prompts)}) ===")
    for i, (p, d) in enumerate(zip(r.shot_prompts, r.durations)):
        e = r.evaluations[i]
        score = f"overall={e.overall} invention={e.invention_ratio} defects={e.has_defects}" if e else ""
        print(f"\n  [shot {i + 1}] {d}s  {score}")
        print(f"    {p[:220]}...")

    if r.clips:
        print(f"\n=== CHAINED CLIPS ===")
        for c in r.clips:
            print(f"  {c}")
    elif not gen:
        print("\n(dry run -- add --generate to render the chain)")


if __name__ == "__main__":
    main()

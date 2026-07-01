"""Demo the wired-in requirement-graph core path with the pre-generation EVAL-GATE.

Shows, for a deliberately WEAK prompt:
  - refuse policy  -> no Veo spend, returns why + suggestions
  - revise policy  -> auto-improves the prompt, re-scores, proceeds
and for a RICH prompt -> passes the gate directly.

All runs here are dry (no Veo spend). Add --generate to actually render the
revise-path result.

Run:
    python -m examples.run_gated
    python -m examples.run_gated --generate
"""

from __future__ import annotations

import sys

from orchestrator.api import generate_gated

WEAK = "a guy walks"
RICH = ("A weary detective in a rain-soaked trench coat lights a cigarette under a flickering "
        "neon sign in a 1940s alley at night, film noir, slow dolly-in.")


def _show(tag, r):
    e = r.evaluation
    print(f"\n===== {tag} =====")
    print(f"  gated (refused): {r.gated}   revised: {r.revised}")
    if r.gate_reason:
        print(f"  gate: {r.gate_reason}")
    if e:
        print(f"  scores: overall={e.overall} coherence={e.coherence} "
              f"specificity={e.specificity} | invention={e.invention_ratio} defects={e.has_defects}")
        if e.conflicts:
            print(f"  conflicts: {e.conflicts}")
        if e.suggestions:
            print(f"  suggestions: {e.suggestions}")
    print(f"  veo_prompt: {r.veo_prompt[:180]}...")
    if r.clips:
        print(f"  CLIPS: {r.clips}")


def main() -> None:
    gen = "--generate" in sys.argv

    # 1) weak prompt, REFUSE (strict bar) -> should not generate
    _show("WEAK + refuse", generate_gated(WEAK, min_quality=0.90, gate_policy="refuse", dry_run=True))

    # 2) weak prompt, REVISE -> should auto-improve and (dry) proceed
    _show("WEAK + revise", generate_gated(WEAK, min_quality=0.90, gate_policy="revise", dry_run=True))

    # 3) rich prompt -> passes the gate directly
    _show("RICH (passes)", generate_gated(RICH, min_quality=0.65, gate_policy="revise", dry_run=True))

    # 4) optional: actually render the rich one
    if gen:
        _show("RICH + GENERATE", generate_gated(RICH, min_quality=0.65, gate_policy="revise", dry_run=False))


if __name__ == "__main__":
    main()

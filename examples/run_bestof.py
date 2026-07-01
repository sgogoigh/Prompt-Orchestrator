"""Best-of-N demo: ONE single-shot prompt, N Fast candidates, QC picks the winner.

Run:
    DEFAULT_NUM_VIDEOS=3 MAX_QC_RETRIES=0 python -m examples.run_bestof

Uses a deliberately atomic (single-shot) prompt so the strategy resolves to
SINGLE -> the best-of-N path (chains force one clip per shot instead).
"""

from __future__ import annotations

from orchestrator.api import generate

PROMPT = (
    "A single continuous cinematic shot: a lone detective in a fedora strikes a match and "
    "lights a cigarette in his dark office, rain streaking the window behind him. "
    "High-contrast black-and-white film noir, slow push-in."
)


def main() -> None:
    print(f"\n=== USER PROMPT ===\n{PROMPT}\n")
    result = generate(PROMPT, dry_run=False, persist=False)

    print(f"=== STRATEGY ===\n{result.strategy}\n")
    print(f"=== ORCHESTRATED VEO PROMPT ===\n{result.veo_prompt}\n")

    print(f"=== QC CANDIDATE SCORES ({len(result.candidates)}) ===")
    winner = result.clips[0] if result.clips else None
    for path, adherence, accept in result.candidates:
        mark = "  <-- WINNER" if path == winner else ""
        print(f"  adherence={adherence:>4}  accept={accept!s:<5}  {path.split(chr(92))[-1]}{mark}")

    if result.qc:
        v = result.qc[0]
        print(f"\n=== WINNER QC NOTE ===\n  {(v.notes or '')[:260]}")
    print(f"\n=== WINNING CLIP ===\n  {winner}\n")


if __name__ == "__main__":
    main()

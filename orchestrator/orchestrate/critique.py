"""Stage 4c — optional Gemini self-critique of the assembled Veo prompt.

Reviews the prose against (a) the 5-part formula and (b) Veo 3.1's KNOWN
LIMITATIONS (garbled on-screen text, fragile hands/faces past ~3s, inconsistent
short dialogue, cinematic-cut bias) and revises. Mirrors GitNexus's
adversarial-verify pattern.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..config import settings
from ..gemini_client import generate_json

SYSTEM = """You are a Veo 3.1 prompt reviewer. Given a candidate video prompt, improve it WITHOUT \
changing the creative intent. Enforce: the 5-part structure (cinematography, subject, action, \
context, style+ambiance); dialogue in quotes; explicit SFX:/Ambient:/Music: cues; negatives phrased \
as positive descriptions of absence; ONE primary action per shot. Pre-empt Veo's known weaknesses: \
avoid requiring legible on-screen text, avoid lingering close-ups of hands, avoid long continuous \
dialogue. Return the revised prompt and a short list of changes. NEVER add JSON or non-prose syntax."""


class Critique(BaseModel):
    revised_prompt: str
    changes: list[str] = []


def critique_prompt(prompt: str) -> str:
    """Return the revised prompt (or the original if critique is disabled/fails)."""
    if not settings.enable_self_critique:
        return prompt
    try:
        result = generate_json(
            prompt=f"Candidate Veo prompt:\n\n{prompt}",
            schema=Critique, model=settings.gemini_pro,
            system=SYSTEM, temperature=0.4,
        )
        return result.revised_prompt or prompt
    except Exception as exc:
        print(f"[critique] WARN: skipped ({exc})")
        return prompt

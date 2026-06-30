"""Stage 5 (optional) — vision QC loop.

After Veo returns a clip, ask Gemini (multimodal) to score prompt adherence and
flag artifacts (missing entities, bad lip-sync, extra fingers, garbled text). If
the score is below threshold, the caller adjusts the prompt and regenerates
(bounded by MAX_QC_RETRIES). This closing loop is what turns a one-shot prompt
into a self-correcting system — the real GitNexus-style value.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..gemini_client import analyze_video


class QCVerdict(BaseModel):
    adherence: float          # 0..1 — how well the clip matches the prompt
    entities_present: bool
    artifacts: list[str] = []
    lip_sync_ok: bool = True
    accept: bool = True
    notes: str = ""


QC_QUESTION = (
    "Compare this video to the intended prompt. Score prompt adherence 0..1, list visual "
    "artifacts (extra fingers, warped faces, garbled on-screen text), confirm the intended "
    "subjects appear, and judge lip-sync. Set accept=false if adherence < 0.6 or there are "
    "severe artifacts."
)


def assess(video_path: str, intended_prompt: str) -> QCVerdict:
    """Run the multimodal QC check on one clip. (Depends on gemini_client.analyze_video — P5.)"""
    question = f"{QC_QUESTION}\n\nIntended prompt:\n{intended_prompt}"
    return analyze_video(video_path, question, schema=QCVerdict)

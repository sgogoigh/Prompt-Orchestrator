"""Shared Gemini wrapper over the google-genai SDK.

One place for: structured-JSON text calls (the workhorse of every fill-in stage),
image generation (reference images for Ingredients), and multimodal vision (QC).

Docs: https://ai.google.dev/gemini-api/docs  ·  SDK: `google-genai`
"""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from .config import settings

try:  # keep import-light so scaffolding can be inspected without the dep installed
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

T = TypeVar("T", bound=BaseModel)

_client = None


def client():
    """Lazily build a singleton genai client (validates the API key on first use)."""
    global _client
    if genai is None:
        raise RuntimeError("google-genai not installed. Run: pip install -r requirements.txt")
    if _client is None:
        _client = genai.Client(api_key=settings.require_key())
    return _client


def generate_json(
    prompt: str,
    schema: Type[T],
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int = 32768,
    retries: int = 2,
) -> T:
    """Run a Gemini text call constrained to `schema` and parse into the pydantic model.

    Uses the SDK's native structured-output mode (`response_schema`), so the model
    is forced to emit JSON matching `schema` — no brittle string parsing.

    Robust to flaky generations: a model can occasionally ramble past the output
    cap and return truncated (invalid) JSON. We bound output with
    `max_output_tokens` and retry a fresh sample up to `retries` times, preferring
    the SDK's already-parsed object.
    """
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        system_instruction=system,
        max_output_tokens=max_output_tokens,
    )

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        resp = client().models.generate_content(
            model=model or settings.gemini_flash,
            contents=prompt,
            config=cfg,
        )
        # The SDK returns a parsed object when response_schema is a pydantic type.
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            text = resp.text
            if text:
                return schema.model_validate_json(text)
            last_err = ValueError("empty response (possibly blocked or truncated)")
        except Exception as e:  # truncated/invalid JSON -> retry a fresh sample
            last_err = e

    raise RuntimeError(
        f"generate_json: model returned no valid JSON for {schema.__name__} "
        f"after {retries + 1} attempts. Last error: {last_err}"
    )


def generate_image(prompt: str, *, out_path: str, model: str | None = None) -> str:
    """Generate one reference image (Gemini 2.5 Flash Image / 'Nano Banana') and save it.

    Returns the path written. Used by specify/reference_images.py to make the
    canonical look of a recurring entity for Veo Ingredients.
    """
    resp = client().models.generate_content(
        model=model or settings.gemini_image,
        contents=prompt,
    )
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            with open(out_path, "wb") as f:
                f.write(part.inline_data.data)
            return out_path
    raise RuntimeError("No image returned by the model.")


def analyze_video(video_path: str, question: str, schema: Type[T], *, model: str | None = None) -> T:
    """Multimodal QC: ask Gemini about a generated clip (adherence, artifacts, lip-sync).

    Used by qc/qc.py. TODO: upload via Files API and attach as a video part.
    """
    raise NotImplementedError("QC vision call — wire up Files API upload + video part (P5).")

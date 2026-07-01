"""Stage 5 — dispatch the orchestrated prompt to Veo 3.1 and download the result.

Wraps google-genai `generate_videos` (long-running op -> poll -> download).
Handles the four strategies; SCENE_EXTENSION / FIRST_LAST chaining are stubbed
for P4.

Veo config (per VIDEO-MODELS.md): aspect_ratio 16:9|9:16, resolution 720p|1080p|4k,
duration 4|6|8, generate_audio, number_of_videos 1-4, plus reference_images
(Ingredients), image (first frame), last_frame, video (extension).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ..config import settings
from ..gemini_client import client

try:
    from google.genai import types
except ImportError:  # pragma: no cover
    types = None

_POLL_SECONDS = 10


@dataclass
class VeoRequest:
    prompt: str
    reference_images: list[str] | None = None   # Ingredients (<=3)
    first_frame: str | None = None
    last_frame: str | None = None
    extend_video: str | None = None
    aspect_ratio: str = settings.aspect_ratio
    resolution: str = settings.resolution
    duration_s: int = settings.duration_s
    generate_audio: bool = True
    num_videos: int = settings.num_videos
    fast: bool = settings.draft_with_fast


def _one_image(path: str | None):
    """Load a single Image from disk, or None if missing."""
    if path and os.path.exists(path):
        return types.Image.from_file(location=path)
    return None


def _normalize_duration(resolution: str, duration: int) -> int:
    """Snap to a (resolution, duration) pairing Veo 3.1 actually accepts.

    The Developer API rejects e.g. 1080p @ 6s ("1080p is not supported for a
    duration of 6 seconds"). 1080p reliably supports the native 8s clip length, so
    we snap 1080p to 8s; 720p accepts 4/6/8.
    """
    if resolution == "1080p":
        return 8
    return duration if duration in (4, 6, 8) else 8


def _ref_images(paths: list[str] | None):
    """Wrap reference-image files as VideoGenerationReferenceImage (Ingredients).

    Veo's `reference_images` expects VideoGenerationReferenceImage objects
    (image + reference_type), NOT raw Image objects. ASSET = keep this subject
    consistent (character/object); STYLE = match this look.
    """
    if not paths:
        return None
    refs = []
    for p in paths:
        img = _one_image(p)
        if img is not None:
            refs.append(
                types.VideoGenerationReferenceImage(
                    image=img,
                    reference_type=types.VideoGenerationReferenceType.ASSET,
                )
            )
    return refs or None


def generate(req: VeoRequest, *, out_dir: str | None = None) -> list[str]:
    """Submit one generation, poll to completion, download MP4(s). Returns file paths.

    NOTE: this is the single-call path (Strategy.SINGLE / TIMESTAMP). Multi-clip
    narratives use `generate_chain` (SCENE_EXTENSION) or `generate_transition`
    (FIRST_LAST) instead.
    """
    out_dir = out_dir or os.path.join(settings.asset_store_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    model = settings.veo_fast if req.fast else settings.veo_standard

    # NOTE: `generate_audio`, `fps`, `seed`, `output_gcs_uri`, `compression_quality`,
    # `labels`, `mask` are Vertex-only — passing them on the Gemini Developer API
    # (api-key path) raises. Veo 3.1 generates native audio by default here, so we
    # simply omit the flag. `reference_images` (Ingredients) + `last_frame` ARE
    # supported on the Developer API.
    # Developer API constraint: sampleCount must be 1 ("out of bound … between 1
    # and 1"). Multiple candidates = multiple calls (best-of-N loops in api.py).
    config = types.GenerateVideosConfig(
        aspect_ratio=req.aspect_ratio,
        resolution=req.resolution,
        duration_seconds=_normalize_duration(req.resolution, req.duration_s),
        number_of_videos=1,
        reference_images=_ref_images(req.reference_images),
        last_frame=_one_image(req.last_frame),
    )

    op = client().models.generate_videos(
        model=model,
        prompt=req.prompt,
        image=_one_image(req.first_frame),
        config=config,
    )
    paths, _ = _await_and_save(op, out_dir, tag="veo")
    return paths


def _await_and_save(op, out_dir: str, tag: str = "veo"):
    """Poll a generate_videos op to completion, then download + save every clip.

    Returns (paths, videos). The returned Video objects are what a chain feeds
    forward as `video=` for the next extension step.
    """
    while not op.done:
        time.sleep(_POLL_SECONDS)
        op = client().operations.get(op)

    paths: list[str] = []
    videos = []
    for i, gen in enumerate(op.response.generated_videos):
        path = os.path.join(out_dir, f"{tag}_{int(time.time())}_{i}.mp4")
        client().files.download(file=gen.video)
        gen.video.save(path)
        paths.append(path)
        videos.append(gen.video)
    return paths, videos


def _last_frame_png(video_path: str, out_dir: str, idx: int) -> str | None:
    """Extract the final frame of a clip as a PNG (for last-frame chaining).

    Uses OpenCV. Seeks to the last frame; falls back to a sequential read if the
    codec doesn't support precise seeking. Returns the PNG path, or None on failure.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = None
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            ok, frame = cap.read()
            if not ok:
                frame = None
        if frame is None:  # sequential fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frame = f
        if frame is None:
            return None
        out = os.path.join(out_dir, f"frame_chain{idx}.png")
        cv2.imwrite(out, frame)
        return out
    finally:
        cap.release()


def generate_chain(
    prompts: list[str],
    req: VeoRequest,
    *,
    durations: list[int] | None = None,
    out_dir: str | None = None,
    qc_gate=None,        # optional callable(clip_path, shot_prompt) -> bool (passes?)
    qc_retries: int = 0,  # re-roll a clip up to N times if it fails the gate
) -> list[str]:
    """SCENE_EXTENSION via LAST-FRAME CHAINING (Developer-API compatible).

    Native scene extension (passing the prior clip as `video=`) is NOT available on
    the Gemini Developer API for Veo 3.1 ("Your use case is currently not
    supported"). So we get continuity the supported way:

      clip 0 = text/image-to-video (+ Ingredients reference images)
      clip i = image-to-video whose FIRST frame is the LAST frame of clip i-1

    The shared frame visually welds the seam; reference images on clip 0 anchor the
    character look. Image-to-video supports 1080p, so the chain stays full-res.

    NOTE vs native extension: there is no audio carryover across the seam and no
    motion continuity beyond the shared still — each clip generates fresh audio.

    Returns clip paths in narrative order (chain[0], chain[1], ...).
    """
    if not prompts:
        return []
    out_dir = out_dir or os.path.join(settings.asset_store_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    model = settings.veo_fast if req.fast else settings.veo_standard

    all_paths: list[str] = []
    prev_frame: str | None = req.first_frame  # optional starting still for clip 0
    for i, prompt in enumerate(prompts):
        dur = durations[i] if (durations and i < len(durations)) else req.duration_s
        dur = _normalize_duration(req.resolution, dur)
        # Reference images (Ingredients) only on clip 0; later clips inherit the look
        # from the seeded first frame (avoids combining first-frame + refs, which the
        # API may reject).
        config = types.GenerateVideosConfig(
            aspect_ratio=req.aspect_ratio,
            resolution=req.resolution,
            duration_seconds=dur,
            number_of_videos=1,  # a chain renders one canonical clip per shot
            reference_images=_ref_images(req.reference_images) if i == 0 else None,
        )
        # Generate this clip; if a QC gate is provided, re-roll it (bounded) until
        # it passes — so a broken clip never seeds the rest of the chain.
        attempt = 0
        clip = None
        while True:
            op = client().models.generate_videos(
                model=model,
                prompt=prompt,
                image=_one_image(prev_frame),
                config=config,
            )
            paths, _ = _await_and_save(op, out_dir, tag=f"chain{i}")
            clip = paths[-1] if paths else None
            if not clip or qc_gate is None or attempt >= qc_retries:
                break
            if qc_gate(clip, prompt):
                break
            attempt += 1  # failed gate, retries remain -> re-roll this shot

        if clip:
            all_paths.append(clip)
            prev_frame = _last_frame_png(clip, out_dir, i) or prev_frame
    return all_paths


def generate_transition(
    first_frame: str,
    last_frame: str,
    prompt: str,
    req: VeoRequest,
    *,
    out_dir: str | None = None,
) -> list[str]:
    """FIRST & LAST FRAME — interpolate one clip between two stills (+ native audio).

    `first_frame` / `last_frame` are image paths (e.g. produced by the Gemini image
    model). The prompt describes the motion/camera move between them.
    """
    out_dir = out_dir or os.path.join(settings.asset_store_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    model = settings.veo_fast if req.fast else settings.veo_standard
    config = types.GenerateVideosConfig(
        aspect_ratio=req.aspect_ratio,
        resolution=req.resolution,
        duration_seconds=_normalize_duration(req.resolution, req.duration_s),
        number_of_videos=1,
        last_frame=_one_image(last_frame),
    )
    op = client().models.generate_videos(
        model=model, prompt=prompt,
        image=_one_image(first_frame), config=config,
    )
    paths, _ = _await_and_save(op, out_dir, tag="transition")
    return paths

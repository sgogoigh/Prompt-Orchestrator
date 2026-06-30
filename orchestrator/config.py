"""Central configuration — loads .env and exposes model IDs + generation defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Anchor .env to the project root (parent of the `orchestrator` package), so the
# key loads no matter what the current working directory is.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # ── credentials ──
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()

    # ── model IDs ──
    # Defaults pinned to the newest models available on this account (verified via
    # models.list()). NOTE: gemini-3.5-pro / gemini-3.5-flash-image do NOT exist —
    # newest Pro is gemini-3.1-pro-preview, newest flash image is gemini-3.1-flash-image.
    gemini_flash: str = os.getenv("GEMINI_TEXT_FLASH", "gemini-3.5-flash")
    gemini_pro: str = os.getenv("GEMINI_TEXT_PRO", "gemini-3.1-pro-preview")
    gemini_image: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
    veo_standard: str = os.getenv("VEO_MODEL_STANDARD", "veo-3.1-generate-preview")
    veo_fast: str = os.getenv("VEO_MODEL_FAST", "veo-3.1-fast-generate-preview")

    # ── generation defaults ──
    aspect_ratio: str = os.getenv("DEFAULT_ASPECT_RATIO", "16:9")
    resolution: str = os.getenv("DEFAULT_RESOLUTION", "1080p")
    duration_s: int = int(os.getenv("DEFAULT_DURATION_S", "8"))
    num_videos: int = int(os.getenv("DEFAULT_NUM_VIDEOS", "2"))
    draft_with_fast: bool = _b("DRAFT_WITH_FAST", True)

    # ── pipeline toggles ──
    enable_retrieval: bool = _b("ENABLE_RETRIEVAL", True)
    enable_reference_images: bool = _b("ENABLE_REFERENCE_IMAGES", True)
    enable_self_critique: bool = _b("ENABLE_SELF_CRITIQUE", True)
    enable_qc_loop: bool = _b("ENABLE_QC_LOOP", True)
    max_qc_retries: int = int(os.getenv("MAX_QC_RETRIES", "2"))

    # ── storage ──
    asset_store_dir: str = os.getenv("ASSET_STORE_DIR", "./assets")
    scene_graph_db: str = os.getenv("SCENE_GRAPH_DB", "./assets/scene_graphs.json")

    def require_key(self) -> str:
        if not self.gemini_api_key or self.gemini_api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Edit the .env file and paste your key "
                "(get one at https://aistudio.google.com/apikey)."
            )
        return self.gemini_api_key


settings = Settings()

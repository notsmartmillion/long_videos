"""Image captioning stub for QA (Phase 6)

Replace with BLIP/LLava integration when available. Current version returns a
simple filename-based caption as a placeholder to keep the pipeline cohesive.
"""

from __future__ import annotations

from pathlib import Path


def caption_image(image_path: str) -> str:
    p = Path(image_path)
    # Placeholder: turn filename tokens into a pseudo-caption
    # e.g., topic_timestamp_keywords -> "topic keywords"
    name = p.stem.replace('_', ' ')
    return f"Image of {name}"



"""Timeline Builder (Phase 7 stub)

Converts beat-mapped prompts into a simple assembly timeline description that
the existing video assembler can consume or that can be translated to segments.
"""

from __future__ import annotations

from typing import List, Dict, Any


def build_timeline(enhanced_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a simple timeline list of clips with start/end and image path placeholder.

    The assembler can later replace placeholders with real image paths after
    generation and set per-clip transitions.
    """
    timeline = []
    for p in enhanced_prompts:
        entry = {
            "beat_id": p.get("beat_id"),
            "shot_type": p.get("shot_type"),
            "start_s": p.get("start_s", 0.0),
            "end_s": p.get("end_s", 0.0),
            "image_path": None,  # to be filled after image generation
            "transition": {"type": "crossfade", "duration": 0.5},
        }
        timeline.append(entry)
    return timeline



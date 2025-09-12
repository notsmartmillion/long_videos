"""Alignment utilities (Phase 4 stub)

Provides forced alignment wrappers and mapping from narration token spans to
audio timecodes. This is a stub implementation designed to be safe to import
without external tools installed. In production, integrate Aeneas/Gentle/MFA.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


def force_align(audio_path: str, script_text: str, mode: str = "heuristic") -> Dict[str, Any]:
    """Return alignment data.

    Stub: returns sentence-level boundaries via simple heuristics if an external
    aligner is not available. Structure is intentionally generic.
    """
    # Heuristic fallback
    sentences = [s.strip() for s in script_text.replace('\n', ' ').split('.') if s.strip()]
    num = max(1, len(sentences))
    # Without audio decoding here, we return indexes only; mapping to time is a
    # separate step that will use the actual audio duration in the pipeline.
    spans = []
    token_cursor = 0
    for s in sentences:
        tokens = s.split()
        start = token_cursor
        end = token_cursor + max(1, len(tokens))
        spans.append({"start_token": start, "end_token": end, "text": s})
        token_cursor = end + 1
    return {"granularity": "sentence", "spans": spans}


def map_beats_to_times(beats: List[Dict[str, Any]], alignment_data: Dict[str, Any], total_audio_seconds: float) -> List[Dict[str, Any]]:
    """Map beat narration spans to approximate start/end times.

    If we have sentence spans, assign each beat a time window proportional to
    its token length. Ensures non-overlapping, contiguous coverage of audio.
    """
    # Compute token lengths per beat
    total_tokens = 0
    beat_tokens = []
    for b in beats:
        span = b.get("narration_span", {})
        length = max(1, int(span.get("end_token", 0) - span.get("start_token", 0)))
        beat_tokens.append(length)
        total_tokens += length

    # Avoid division by zero
    total_tokens = max(1, total_tokens)

    # Assign times proportionally
    result = []
    t = 0.0
    for b, tok in zip(beats, beat_tokens):
        dur = (tok / total_tokens) * total_audio_seconds
        start_s = t
        end_s = min(total_audio_seconds, t + dur)
        out = dict(b)
        out["start_s"] = float(start_s)
        out["end_s"] = float(end_s)
        result.append(out)
        t = end_s

    # Ensure last beat ends at exact audio end
    if result:
        result[-1]["end_s"] = float(total_audio_seconds)
    return result



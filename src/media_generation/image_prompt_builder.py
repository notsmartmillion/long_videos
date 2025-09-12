"""Image Prompt Builder (Phase 5 stub)

Transforms a VisualPlan + style template + optional topic adapter into
sanitized, class-based prompts with deterministic seeds and timing metadata.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Any, List

# Global negatives and shot normalization
NEG_CORE = (
    "statue, marble, plaster, engraving, sketch, cartoon, anime, text, watermark, signature, "
    "lowres, blurry, jpeg artifacts, deformed, mutated, extra fingers, extra limbs, fused limbs, "
    "mangled hands, duplicate face, disfigured, poorly drawn face, doll, wax figure"
)

SHOT_MAP = {
    "map": "establishing_wide_landscape",
    "diagram": "hero_prop_close",
}

def _normalize_shot(shot: str) -> str:
    if not shot:
        return "establishing"
    s = shot.strip().lower()
    return SHOT_MAP.get(s, s)


def seed_from(*parts: Any, mod: int = 2 ** 32) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) % mod


def sanitize_prompt(text: str, max_tokens: int = 75) -> str:
    # Very simple CLIP-friendly truncation by words
    words = text.split()
    return " ".join(words[:max_tokens])


def build_prompts(*, visual_plan: Dict[str, Any], style_template: Dict[str, str], topic: str,
                  adapters: Dict[str, Any], seed_namespace: str) -> List[Dict[str, Any]]:
    """Return a list of enhanced prompts with seeds and timestamps if present.

    Output items: { prompt, negatives, seed, shot_type, beat_id, start_s?, end_s? }
    """
    adapter = adapters.get(topic, adapters.get("default", {})) or {}
    style_base = style_template.get("base_style", "")
    style_colors = style_template.get("colors", "")
    style_mood = style_template.get("mood", "")

    results: List[Dict[str, Any]] = []
    beats = visual_plan.get("beats", [])
    for beat in beats:
        beat_id = beat.get("id", "beat")
        seed_group = beat.get("seed_group", beat_id)
        shot_type = beat.get("shot_type", "establishing")
        narration_span = beat.get("narration_span")
        for idx, v in enumerate(beat.get("prompts", [])):
            base_prompt = v.get("prompt", "").strip()
            negatives = v.get("negatives", "")

            # Enrich with style locks (drop labels like colors:/mood:)
            style_bits = ", ".join(x for x in [style_base, style_colors, style_mood] if x)
            if style_bits:
                full_prompt = sanitize_prompt(f"{base_prompt}, {style_bits}")
            else:
                full_prompt = sanitize_prompt(base_prompt)

            # Deterministic seed
            seed = seed_from(seed_namespace, topic, seed_group, idx)

            # Merge negatives (adapter + incoming + core)
            neg_list: List[str] = []
            adapter_negs = adapter.get("negatives", []) or []
            if isinstance(adapter_negs, list):
                neg_list.extend(adapter_negs)
            elif isinstance(adapter_negs, str):
                neg_list.append(adapter_negs)
            if negatives:
                neg_list.append(negatives)
            neg_list.append(NEG_CORE)
            negatives_merged = ", ".join([n for n in neg_list if n])

            out = {
                "beat_id": beat_id,
                "shot_type": _normalize_shot(shot_type),
                "prompt": full_prompt,
                "negatives": negatives_merged,
                "seed": int(seed),
            }
            # Carry mapped times if present
            if "start_s" in beat and "end_s" in beat:
                out["start_s"] = float(beat["start_s"])
                out["end_s"] = float(beat["end_s"])
            if narration_span:
                out["narration_span"] = narration_span
            results.append(out)
    return results



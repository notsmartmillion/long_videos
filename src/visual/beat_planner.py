from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib
import re

DEFAULT_NEGATIVES = [
    "logos", "watermarks", "text", "subtitles", "frames", "borders",
    "lowres", "blurry", "jpeg artifacts", "oversaturated",
    "marble statue skin", "plastic skin", "mannequin", "doll", "wax figure",
    "engraving", "sketch", "cartoon", "anime",
    "deformed hands", "extra fingers", "extra limbs", "fused limbs",
    "distorted face", "duplicate face",
    "celebrity likeness", "modern clothing", "anachronisms",
    # extra anachronisms
    "digital watch", "headphones", "smartphone", "neon signage",
    "modern city skyline", "sci-fi UI"
]

SHOT_TEMPLATES: Dict[str, str] = {
    "establishing":
        ("Dawn over {PRIMARY_LOCALE} under dramatic clouds; {MAIN_SUBJECT} as a living human figure on a high promontory, "
         "laurel crown, ancient robes; wide vista of peaks and mist; low-angle, rule of thirds, crisp edges; "
         "living human figures, realistic anatomy, cinematic volumetric light, {STYLE_CORE}"),
    "establishing_wide_landscape":
        ("Grand landscape: coastline, olive groves, distant marble cities beneath storm light; omen bird circling above as sign of {MAIN_SUBJECT}; "
         "atmospheric perspective, grand scale, crisp edges; living human figures, realistic anatomy, cinematic volumetric light, {STYLE_CORE}"),
    "medium_detail":
        ("Medium portrait of {MAIN_SUBJECT}: calm, intelligent eyes catching distant lightning; windswept hair; "
         "chiton with gold trim; hand resting on ceremonial scepter; soft rim light; painterly yet crisp edges; "
         "living human figures, realistic anatomy, cinematic volumetric light, {STYLE_CORE}"),
    "insert":
        ("Insert close-up: {MAIN_SUBJECT} hand; faint electricity arcing across bronze knuckles; laurel leaves beaded with rain; "
         "shallow depth, cinematic bokeh; tactile metal/skin textures; living human figures, realistic anatomy, "
         "cinematic volumetric light, {STYLE_CORE}"),
    "hero_prop_close":
        ("Hero prop close: ceremonial artifact (e.g., scepter, tablet) resting on carved stone; filigree and ancient runes glint; "
         "extreme detail macro, shallow depth, crisp edges; {STYLE_CORE}"),
    "archival":
        ("Archival plate: faded parchment illustration of {SECONDARY_FIGURE} and {MOTIF}; ink wash, foxing, creases; "
         "limited palette (sienna/charcoal); photographed like museum artifact with raking light; {STYLE_CORE_ARCHIVAL}"),
    "reenactment":
        ("Staged reenactment: {SECONDARY_FIGURE} on a crag at night, torch flaring against storm clouds; distant glimmer of {PRIMARY_LOCALE}; "
         "low-angle dynamic pose, sparks trailing; dramatic yet grounded realism; "
         "living human figures, realistic anatomy, cinematic volumetric light, {STYLE_CORE}"),
    "abstract":
        ("Abstract metaphor: a human iris swirling with storm clouds and micro lightning filaments—power and conscience entwined; "
         "macro eye texture, subtle mountain reflections; elegant, minimal, poetic; crisp edges, controlled palette; {STYLE_CORE}"),
    "map":
        ("Cartographic interlude: hand-painted map of {REGION} on parchment; coastline lines, mountain hatching, laurel cartouche; "
         "aged paper texture lit by grazing light; {STYLE_CORE_ARCHIVAL}"),
    "diagram":
        ("Ancient diagram: symbolic schema of lineage/events related to {MAIN_SUBJECT}; painted glyphs, simple arrows, "
         "subtle gold leaf accents on parchment; studio-lit artifact aesthetic; {STYLE_CORE_ARCHIVAL}")
}

# Optional shot-type style locks (can be appended to prompt or exposed separately)
SHOT_STYLE_LOCKS: Dict[str, List[str]] = {
    "establishing": ["epic, mystical, ancient", "rich golds, deep blues, ivory whites"],
    "establishing_wide_landscape": ["grand scale", "atmospheric perspective"],
    "medium_detail": ["soft rim light", "cinematic volumetric light"],
    "insert": ["macro texture detail", "shallow depth of field"],
    "hero_prop_close": ["extreme macro", "studio-lit artifact"],
    "archival": ["aged paper texture", "museum raking light"],
    "map": ["parchment cartography", "ink hatching"],
    "diagram": ["parchment schematic", "flat ink glyphs"],
    "reenactment": ["dynamic low-angle", "sparks and storm haze"],
    "abstract": ["minimal, elegant", "controlled palette"],
}


@dataclass
class GlobalStyle:
    topic: str
    style_template_key: str
    aspect_ratio: str
    color_profile: str
    tone: str


@dataclass
class Entity:
    id: str
    kind: str
    descriptor: str
    persist_across_beats: bool = True


def _style_core(global_style: GlobalStyle) -> str:
    palette = "rich golds, deep blues, ivory whites"
    base = f"{global_style.style_template_key}, {palette}, {global_style.tone}"
    base = base.replace("but not statues", "").strip()
    return base


def _style_core_archival(global_style: GlobalStyle) -> str:
    return "neoclassical scholarly mood, restrained palette, museum-lit artifact, " + _style_core(global_style)


def _topic_negatives(topic: str) -> List[str]:
    t = (topic or "").lower()
    if any(k in t for k in ["ancient", "myth", "greece", "rome", "medieval", "renaissance"]):
        return ["concrete freeway", "skyscraper", "street light", "plastic", "synthetic fabric"]
    return []


def dedupe_negatives(*lists: List[str]) -> str:
    seen = set()
    out = []
    for lst in lists:
        if not lst:
            continue
        for n in lst:
            key = (n or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(n)
    return ", ".join(out)


def stable_seed(seed_group: str, beat_id: str) -> int:
    key = f"{seed_group}:{beat_id}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def stable_variant_seed(base_seed: int, variant_index: int) -> int:
    return (base_seed + (variant_index * 9973)) & 0xFFFFFFFF


def choose_subjects_from_context(
    text: str,
    entities: Dict[str, Entity],
    fallback_subject: str = "the central figure",
    fallback_locale: str = "a grand setting",
) -> Dict[str, str]:
    txt = (text or "").lower()

    def pick_by_ids(candidates: List[str], fallback: str) -> str:
        for cid in candidates:
            if cid in entities:
                return entities[cid].id
        for e in entities.values():
            if e.id.lower() in txt:
                return e.id
        return fallback

    main_subject = pick_by_ids(["zeus", "apollo", "athena"], fallback_subject)
    secondary = pick_by_ids(["prometheus", "hera"], "a key figure")
    primary_locale = "Mount Olympus" if "olymp" in txt else fallback_locale
    related_event = "the Olympic Games" if ("olymp" in txt or "games" in txt) else "a sacred festival"
    motif = "bringing fire to humanity" if ("fire" in txt or secondary == "prometheus") else "an origin myth"
    region = "the Mediterranean" if any(w in txt for w in ["aegean", "crete", "greece"]) else "the known world"

    return {
        "MAIN_SUBJECT": main_subject,
        "SECONDARY_FIGURE": secondary,
        "PRIMARY_LOCALE": primary_locale,
        "RELATED_EVENT": related_event,
        "MOTIF": motif,
        "REGION": region,
    }


def _sanitize_prompt(text: str) -> str:
    # Remove repeated generic openers and fix commas
    txt = re.sub(r'^\s*establishing,\s*', '', (text or '').strip(), flags=re.IGNORECASE)
    txt = re.sub(r'(,\s*){2,}', ', ', txt)
    # Deduplicate comma-separated descriptors (basic pass)
    parts = [p.strip() for p in txt.split(',')]
    seen = set()
    out: List[str] = []
    for p in parts:
        key = p.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return ', '.join(out).strip()


def _strip_live_figure_terms(prompt: str) -> str:
    low = prompt.lower()
    if any(k in low for k in ("archival", "map", "diagram")):
        out = re.sub(r'\bliving human figures\b', '', prompt, flags=re.IGNORECASE)
        out = re.sub(r'\brealistic anatomy\b', '', out, flags=re.IGNORECASE)
        out = re.sub(r'(,\s*){2,}', ', ', out).strip(', ')
        return out
    return prompt


def build_prompt_for_shot(
    shot_type: str,
    global_style: GlobalStyle,
    placeholders: Dict[str, str]
) -> str:
    tpl = SHOT_TEMPLATES.get(shot_type, SHOT_TEMPLATES["establishing"])
    prompt = tpl.format(
        STYLE_CORE=_style_core(global_style),
        STYLE_CORE_ARCHIVAL=_style_core_archival(global_style),
        **placeholders
    )
    if "living human figures" not in prompt and shot_type not in ("archival", "map", "diagram", "abstract"):
        prompt += ", living human figures, realistic anatomy"
    prompt = _sanitize_prompt(prompt)
    prompt = _strip_live_figure_terms(prompt)
    return prompt


def build_beat(
    beat_id: str,
    shot_type: str,
    narration_text: str,
    global_style: GlobalStyle,
    entities: Dict[str, Entity],
    seed_group: Optional[str] = None,
    base_negatives: Optional[List[str]] = None,
    start_s: float = 0.0,
    end_s: float = 0.0,
    narration_span: Optional[Tuple[int, int]] = None
) -> Dict:
    placeholders = choose_subjects_from_context(narration_text, entities)
    prompt = build_prompt_for_shot(shot_type, global_style, placeholders)
    negatives = dedupe_negatives(DEFAULT_NEGATIVES, _topic_negatives(global_style.topic), base_negatives or [])
    seed = stable_seed(seed_group or beat_id, beat_id)

    return {
        "beat_id": beat_id,
        "shot_type": shot_type,
        "prompt": prompt,
        "negatives": negatives,
        "seed": seed,
        "style_locks": SHOT_STYLE_LOCKS.get(shot_type, []),
        "start_s": start_s,
        "end_s": end_s,
        "narration_span": {
            "start_token": narration_span[0] if narration_span else 0,
            "end_token": narration_span[1] if narration_span else 0
        }
    }


def tokens_to_duration_s(st: int, et: int, wps: float = 2.5) -> float:
    count = max(0, (et - st))
    return count / max(0.1, wps)


def _window_tokens(tokens: List[str], st: int, et: int, window: int = 80) -> str:
    if et - st >= 12:
        return " ".join(tokens[st:et])
    lo = max(0, st - window)
    hi = min(len(tokens), et + window)
    return " ".join(tokens[lo:hi])


def _avoid_repeats(shot: str, last_shots: List[str], max_run: int = 2) -> str:
    if len(last_shots) >= max_run and all(s == shot for s in last_shots[-max_run:]):
        alternatives = {
            "establishing": ["establishing_wide_landscape", "medium_detail"],
            "medium_detail": ["insert", "hero_prop_close"],
            "insert": ["medium_detail", "hero_prop_close"],
            "archival": ["diagram", "map"],
            "abstract": ["medium_detail", "insert"],
            "map": ["archival", "diagram"],
            "diagram": ["archival", "map"],
            "reenactment": ["medium_detail", "insert"],
        }
        opts = alternatives.get(shot, ["medium_detail"]) or ["medium_detail"]
        return opts[0]
    return shot


# --------- Beat duration splitter (post-plan safety net) ---------
def split_beats_by_duration(
    beats: List[Dict],
    words_per_second: float = 2.2,
    target_range: Tuple[float, float] = (8.0, 12.0),
    hard_min: float = 6.0,
    hard_max: float = 15.0,
) -> List[Dict]:
    """Split long beats into sub-beats so each lands in [hard_min, hard_max].
    Preserves seed_group continuity and style_locks where present.
    """
    tmin, tmax = target_range
    out: List[Dict] = []

    for b in beats:
        st = int(b.get("narration_span", {}).get("start_token", 0))
        et = int(b.get("narration_span", {}).get("end_token", 0))
        token_len = max(0, et - st)

        # Prefer provided duration; fallback to token estimate
        if b.get("end_s") is not None and b.get("start_s") is not None:
            dur = float(b["end_s"]) - float(b["start_s"])
        else:
            dur = token_len / max(1e-6, words_per_second)
        dur = max(hard_min, float(dur))

        if dur <= hard_max:
            # Clamp into range if timings exist
            if b.get("start_s") is not None:
                b["end_s"] = max(float(b["start_s"]) + hard_min, min(float(b["start_s"]) + dur, float(b["start_s"]) + hard_max))
            b["_est_duration_s"] = (b.get("end_s", 0.0) - b.get("start_s", 0.0)) if b.get("end_s") is not None else dur
            out.append(b)
            continue

        # Split based on target_range midpoint
        target = (tmin + tmax) * 0.5
        n_segments = max(2, int(round(dur / max(0.1, target))))

        seg_tokens = max(1, token_len // n_segments) if token_len > 0 else 1
        segs: List[Dict] = []
        cursor_tok = st
        for i in range(n_segments):
            seg_st = cursor_tok
            seg_et = et if i == n_segments - 1 else min(et, seg_st + seg_tokens)
            cursor_tok = seg_et

            # Time slicing anchored to start_s if present; else derive from tokens
            if b.get("start_s") is not None and b.get("end_s") is not None and token_len > 0:
                total_d = float(b["end_s"]) - float(b["start_s"])
                share = (seg_et - seg_st) / token_len
                seg_dur = max(hard_min, min(hard_max, share * total_d))
                seg_start = float(b["start_s"]) if i == 0 else segs[-1]["end_s"]
                seg_end = seg_start + seg_dur
            else:
                est_dur = max(hard_min, min(hard_max, (seg_et - seg_st) / max(1e-6, words_per_second)))
                seg_start = segs[-1]["end_s"] if segs else float(b.get("start_s", 0.0) or 0.0)
                seg_end = seg_start + est_dur

            seg = {
                **{k: v for k, v in b.items() if k not in ("id", "narration_span", "start_s", "end_s")},
                "id": f"{b.get('id','beat')}_p{i+1}",
                "narration_span": {"start_token": int(seg_st), "end_token": int(seg_et)},
                "start_s": float(seg_start),
                "end_s": float(seg_end),
                "_est_duration_s": float(seg_end - seg_start),
                "seed_group": f"{b.get('seed_group', b.get('id','beat'))}_p{i+1}",
            }
            segs.append(seg)

        out.extend(segs)

    # Chain start_s to guarantee continuous timeline (fix numeric drift)
    if out:
        t_cursor = float(out[0].get("start_s", 0.0))
        if out[0].get("end_s") is None or out[0]["end_s"] <= t_cursor:
            out[0]["end_s"] = t_cursor + max(hard_min, (out[0]["narration_span"]["end_token"] - out[0]["narration_span"]["start_token"]) / max(1e-6, words_per_second))
        t_cursor = out[0]["end_s"]
        for i in range(1, len(out)):
            bb = out[i]
            bb["start_s"] = max(t_cursor, float(bb.get("start_s", t_cursor)))
            if bb.get("end_s") is None or bb["end_s"] <= bb["start_s"]:
                bb["end_s"] = bb["start_s"] + max(hard_min, (bb["narration_span"]["end_token"] - bb["narration_span"]["start_token"]) / max(1e-6, words_per_second))
            t_cursor = bb["end_s"]

    return out



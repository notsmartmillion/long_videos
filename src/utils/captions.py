from pathlib import Path
from typing import Optional

STOPWORDS = {"image", "photo", "picture", "statue", "drawing", "painting"}


def normalize_caption(text: str) -> str:
    try:
        return " ".join(w for w in text.lower().split() if w not in STOPWORDS)
    except Exception:
        return text

_blip_loaded = False
_blip_processor = None
_blip_model = None


def _load_blip():  # lazy, optional
    global _blip_loaded, _blip_processor, _blip_model
    if _blip_loaded:
        return _blip_processor, _blip_model
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_loaded = True
    except Exception:
        _blip_loaded = False
        _blip_processor = None
        _blip_model = None
    return _blip_processor, _blip_model


def captioner_mode_from_config(cfg) -> str:
    try:
        if getattr(cfg, 'alignment', None) and getattr(cfg.alignment, 'captioner', 'stub') == 'blip':
            try:
                import transformers  # type: ignore
                return 'blip'
            except Exception:
                return 'stub'
    except Exception:
        pass
    return 'stub'


def caption_image(path: str, mode: str = "stub") -> str:
    # stub fallback
    if mode == "stub":
        return Path(path).stem.replace('_', ' ')
    if mode == "blip":
        proc, model = _load_blip()
        if proc is None or model is None:
            return Path(path).stem.replace('_', ' ')
        try:
            from PIL import Image  # type: ignore
            image = Image.open(path).convert('RGB')
            inputs = proc(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=30)
            caption = proc.decode(out[0], skip_special_tokens=True)
            return normalize_caption(caption.strip())
        except Exception:
            return Path(path).stem.replace('_', ' ')
    # future: llava/gemini
    return Path(path).stem.replace('_', ' ')



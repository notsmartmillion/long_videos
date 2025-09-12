from .alignment import force_align


def align_text_audio(text: str, audio_path: str, mode: str = "heuristic"):
    """Wrapper for alignment providers. Returns same shape as force_align().
    Currently delegates to heuristic; aeneas integration can be added later.
    """
    if mode == "aeneas":
        # TODO: integrate aeneas to return identical keys
        # Placeholder: fall back to heuristic for now
        pass
    return force_align(audio_path, text)



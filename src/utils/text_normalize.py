import re
from typing import Iterable

# Common non-name tokens that might be Titlecased at sentence start
# and shouldn’t be touched even if they end with 's's (rare, but safe).
_STOP_TOKENS = {
    "This", "As", "Its", "His", "Hers", "Ours", "Yours", "Theirs",
    "Is", "Was", "Has", "Does"
}

_APOSTROPHES = ("’", "‘", "＇")  # common unicode apostrophes


def _to_straight_quotes(text: str) -> str:
    return text.translate(str.maketrans({"’": "'", "‘": "'", "＇": "'"}))


def normalize_name_possessives(text: str) -> str:
    """
    Convert possessives for singular proper names ending in s:
        James's  -> James'
        Achilles’s -> Achilles'
        Socrates's -> Socrates'

    Heuristics:
    - Only convert single tokens that are Titlecased and end with 's's or ’s.
    - Skip ALL-CAPS tokens (ACRONYMS), tokens containing digits/punctuation (ids, codes).
    - Do NOT touch plural possessives already written as "Joneses'".
    - Curly quotes normalized to straight quotes first.
    """
    if not text:
        return text

    text = _to_straight_quotes(text)

    # TitlecaseName's -> TitlecaseName' (straight apostrophe)
    pattern = re.compile(r"\b([A-Z][a-zA-Z]+)'s\b")

    def ok_name_token(tok: str) -> bool:
        if tok in _STOP_TOKENS:
            return False
        if not tok[0].isupper():
            return False
        if tok[-1].lower() != "s":
            return False
        if tok.isupper():
            return False
        if any(ch.isdigit() for ch in tok):
            return False
        if any(ch in "-_.:/\\" for ch in tok):
            return False
        return True

    def repl(m: re.Match) -> str:
        token = m.group(1)
        return f"{token}'" if ok_name_token(token) else m.group(0)

    text = pattern.sub(repl, text)

    # Defensive pass for any remaining fancy quotes patterns
    for fancy in _APOSTROPHES:
        pat = re.compile(rf"\b([A-Z][a-zA-Z]+){re.escape(fancy)}s\b")
        text = pat.sub(lambda m: f"{m.group(1)}'", text)

    return text



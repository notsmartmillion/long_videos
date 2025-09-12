import os
from dotenv import load_dotenv

# Load .env.local once at import
try:
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env.local"))
except Exception:
    pass

from openai import OpenAI

_client = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing (set it in .env.local)")
        _client = OpenAI(api_key=api_key)
    return _client


def choose_model(kind: str) -> str:
    # kind in {"planner","gen","img"}
    if kind == "planner":
        return os.getenv("PLANNER_OPENAI_MODEL", "gpt-4o-mini")
    if kind == "img":
        return os.getenv("IMG_OPENAI_MODEL", "gpt-4o-mini")
    return os.getenv("GEN_OPENAI_MODEL", "gpt-4o-mini")



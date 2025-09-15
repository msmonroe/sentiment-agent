# app/guardrails.py
"""Safety guardrails for input text processing."""
from fastapi import HTTPException
from langdetect import detect, LangDetectException
from typing import Any, Dict

from .config import get_cfg

try:
    from detoxify import Detoxify
except ImportError:
    Detoxify = None

# Global cache for toxicity model
_tox: Detoxify | None = None

def get_tox() -> Detoxify | None:
    """Lazy-loads the Detoxify model for toxicity detection."""
    global _tox
    if _tox is None and Detoxify is not None:
        _tox = Detoxify('original-small')
    return _tox

def guardrails_pre(text: str) -> Dict[str, Any]:
    """Applies pre-inference safety guardrails to the input text."""
    cfg = get_cfg()
    reasons = []
    safety = "ok"
    tox_val = 0.0

    if len(text) > int(cfg["guardrails"]["max_chars"]):
        raise HTTPException(status_code=400, detail=f"Input too long (>{cfg['guardrails']['max_chars']} chars).")

    allow = cfg["guardrails"]["allow_langs"] or []
    try:
        lang = detect(text) if text.strip() else "unknown"
    except LangDetectException:
        lang = "unknown"
    
    if allow and lang not in allow:
        reasons.append(f"lang={lang}")
        safety = "abstain"

    if cfg["guardrails"]["toxicity_gate"]:
        tox_model = get_tox()
        if tox_model is not None:
            tox_val = float(tox_model.predict(text)["toxicity"])
            if tox_val >= float(cfg["guardrails"]["toxicity_threshold"]):
                safety = "blocked"
                reasons.append("toxicity")

    return {"safety": safety, "reasons": reasons, "toxicity": tox_val}
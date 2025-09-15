# app/models.py
"""Handles model loading and prediction logic for sentiment analysis."""
import os
import re
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Any, Dict, Tuple

from .config import get_cfg

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Global cache for models
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None

def get_tokenizer_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Lazy-loads the Hugging Face tokenizer and model."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        cfg = get_cfg()
        model_id = cfg["model"]["id"]
        device = 0 if torch.cuda.is_available() else -1
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForSequenceClassification.from_pretrained(model_id)
        if device == 0:
            _model.cuda()
    return _tokenizer, _model

def softmax(x: np.ndarray) -> np.ndarray:
    """Computes the softmax function for a given array."""
    e = np.exp(x - np.max(x))
    return e / e.sum()

def redact_pii(text: str) -> str:
    """Redacts Personally Identifiable Information (PII) from text."""
    patterns = [
        (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
        (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"), "<PHONE>"),
    ]
    for pat, repl in patterns:
        text = pat.sub(repl, text)
    return text

def predict_with_controls(text: str) -> Dict[str, Any]:
    """Performs sentiment analysis using the local Hugging Face model."""
    cfg = get_cfg()
    tokenizer, model = get_tokenizer_model()

    text_proc = redact_pii(text) if cfg["guardrails"]["pii_redaction"] else text

    inputs = tokenizer(text_proc, return_tensors="pt", truncation=True, max_length=512)
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().detach().cpu().numpy()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    norm = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
    labels = [norm.get(id2label[i].lower(), id2label[i]) for i in range(len(logits))]

    temp = float(cfg["model"]["temperature"])
    logits = logits / max(temp, 1e-6)

    probs = softmax(logits)
    label_to_prob = dict(zip(labels, probs))

    pos = label_to_prob.get("positive", 0.0)
    neg = label_to_prob.get("negative", 0.0)
    neu = label_to_prob.get("neutral", 0.0)
    top_label = max(label_to_prob, key=label_to_prob.get)
    top_p = float(label_to_prob[top_label])

    conf_threshold = float(cfg["model"]["conf_threshold"])
    neutral_band = float(cfg["model"]["neutral_band"])

    if top_p < conf_threshold:
        return {"label": "abstain", "score": top_p, "probs": label_to_prob}
    if abs(pos - neg) < neutral_band:
        return {"label": "neutral", "score": float(max(neu, (pos + neg) / 2)), "probs": label_to_prob}

    return {"label": top_label, "score": top_p, "probs": label_to_prob}

def predict_with_openai(text: str) -> Dict[str, Any]:
    """Performs sentiment analysis using the OpenAI API."""
    cfg = get_cfg()
    if not OpenAI:
        raise RuntimeError("OpenAI SDK not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(
        api_key=api_key,
        organization=os.environ.get("OPENAI_ORG_ID"),
        project=os.environ.get("OPENAI_PROJECT_ID"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )
    model_name = os.environ.get("OPENAI_MODEL") or cfg.get("openai", {}).get("model", "gpt-4o-mini")

    system = "You are a precise sentiment classifier. Return strict JSON with keys: label (positive|neutral|negative) and score (0..1)."
    user = f"Classify sentiment for this text:\n{text}\n\nRespond as JSON only."

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    
    data = json.loads(resp.choices[0].message.content)
    label = str(data.get("label", "neutral")).lower()
    if label not in {"positive", "neutral", "negative"}:
        label = "neutral"
    score = float(data.get("score", 0.5))
    return {"label": label, "score": score, "probs": {label: score}}
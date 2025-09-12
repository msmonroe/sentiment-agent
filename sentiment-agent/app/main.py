# app/main.py
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, time, os, tomllib, threading, logging, hashlib, re
import numpy as np
from langdetect import detect, LangDetectException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
load_dotenv()  # load .env early

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
    
try:
    from detoxify import Detoxify
except Exception:  # optional at runtime
    Detoxify = None

# ---------- Models & Schemas ----------
class AnalyzeRequest(BaseModel):
    text: str
    lang_hint: Optional[str] = None

class AnalyzeBatchRequest(BaseModel):
    texts: List[str]
    lang_hint: Optional[str] = None

class SentimentResult(BaseModel):
    label: str
    score: float

class PolicyBlock(BaseModel):
    safety: str           # "ok" | "abstain" | "blocked"
    reasons: List[str] = []
    toxicity: float = 0.0

class AnalyzeResponse(BaseModel):
    input: str
    result: SentimentResult
    policy: PolicyBlock

# ---------- Config Handling ----------
DEFAULT_CONFIG = {
    "model": {
        "id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "temperature": 1.0,
        "conf_threshold": 0.55,
        "neutral_band": 0.10,
    },
    "guardrails": {
        "max_chars": 2000,
        "allow_langs": ["en"],
        "toxicity_gate": True,
        "toxicity_threshold": 0.80,
        "pii_redaction": False,
        "rate_limit": "10/minute",
        "audit_log": True,
    },
    "server": {
        "reload_config_seconds": 10,
        "cors_allow_origins": ["*"],
    },
}

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.toml")
_config_lock = threading.Lock()
_config: Dict[str, Any] = DEFAULT_CONFIG.copy()

# Fallback no-op decorator
def _rate_limit_decorator(fn):
    return fn

def _load_config():
    global _config
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "rb") as f:
                user_cfg = tomllib.load(f)
            # shallow merge
            merged = DEFAULT_CONFIG.copy()
            for k, v in user_cfg.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
            with _config_lock:
                _config = merged
        else:
            with _config_lock:
                _config = DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"[config] failed to load: {e}")

def get_cfg():
    with _config_lock:
        return _config

def start_config_reloader():
    def loop():
        while True:
            _load_config()
            time.sleep(get_cfg()["server"]["reload_config_seconds"])
    t = threading.Thread(target=loop, daemon=True)
    t.start()

# ---------- App Setup ----------
app = FastAPI(title="Sentiment Agent", version="1.1.0")
_load_config()
start_config_reloader()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cfg()["server"]["cors_allow_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")


# Logging
logging.basicConfig(level=logging.INFO)
audit_log = logging.getLogger("audit")

def audit_event(kind: str, payload: dict):
    if not get_cfg()["guardrails"]["audit_log"]:
        return
    payload = dict(payload)
    if "text" in payload:
        payload["text_sha256"] = hashlib.sha256(payload["text"].encode()).hexdigest()
        payload.pop("text")
    payload["ts"] = int(time.time())
    audit_log.info({"event": kind, **payload})

# ---------- Model Pipeline ----------
MODEL_ID = get_cfg()["model"]["id"]
_tokenizer = None
_model = None
_tox = None

def get_tokenizer_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        device = 0 if torch.cuda.is_available() else -1
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        if device == 0:
            _model.cuda()
    return _tokenizer, _model

def get_tox():
    global _tox
    if _tox is None and Detoxify is not None:
        _tox = Detoxify('original-small')
    return _tox

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

def redact_pii(text: str) -> str:
    patterns = [
        (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"), "<EMAIL>"),
        (re.compile(r"\\b(?:\\+?\\d{1,3}[-.\\s]?)?(?:\\(?\\d{3}\\)?[-.\\s]?){2}\\d{4}\\b"), "<PHONE>"),
    ]
    for pat, repl in patterns:
        text = pat.sub(repl, text)
    return text

def predict_with_controls(text: str) -> Dict[str, Any]:
    cfg = get_cfg()
    tokenizer, model = get_tokenizer_model()

    if cfg["guardrails"]["pii_redaction"]:
        text_proc = redact_pii(text)
    else:
        text_proc = text

    inputs = tokenizer(text_proc, return_tensors="pt", truncation=True, max_length=512)
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().detach().cpu().numpy()

    # labels
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    norm = {"label_0":"negative","label_1":"neutral","label_2":"positive",
            "negative":"negative","neutral":"neutral","positive":"positive"}
    labels = [norm.get(id2label[i].lower(), id2label[i]) for i in range(len(logits))]

    # temperature
    temp = float(cfg["model"]["temperature"])
    logits = logits / max(temp, 1e-6)

    probs = softmax(logits)
    label_to_prob = dict(zip(labels, probs))

    pos = label_to_prob.get("positive", 0.0)
    neg = label_to_prob.get("negative", 0.0)
    neu = label_to_prob.get("neutral", 0.0)
    top_label = max(label_to_prob, key=label_to_prob.get)
    top_p = float(label_to_prob[top_label])

    # thresholds
    conf_threshold = float(cfg["model"]["conf_threshold"])
    neutral_band = float(cfg["model"]["neutral_band"])

    if top_p < conf_threshold:
        return {"label": "abstain", "score": top_p, "probs": label_to_prob}

    if abs(pos - neg) < neutral_band:
        return {"label": "neutral", "score": float(max(neu, (pos+neg)/2)), "probs": label_to_prob}

    return {"label": top_label, "score": top_p, "probs": label_to_prob}

def predict_with_openai(text: str) -> Dict[str, Any]:
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
        base_url=os.environ.get("OPENAI_API_BASE")  # optional
    )
    model_name = os.environ.get("OPENAI_MODEL") or cfg.get("openai", {}).get("model", "gpt-4o-mini")

    # Ask for structured JSON to avoid brittle parsing.
    system = "You are a precise sentiment classifier. Return strict JSON with keys: label (positive|neutral|negative) and score (0..1)."
    user = f"Classify sentiment for this text:\n{text}\n\nRespond as JSON only."

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type": "json_object"},
    )
    import json
    data = json.loads(resp.choices[0].message.content)
    label = str(data.get("label","neutral")).lower()
    if label not in {"positive","neutral","negative"}:
        label = "neutral"
    score = float(data.get("score", 0.5))
    return {"label": label, "score": score, "probs": {label: score}}

# ---------- Guardrails ----------
def guardrails_pre(text: str) -> Dict[str, Any]:
    cfg = get_cfg()
    reasons = []
    safety = "ok"
    tox_val = 0.0

    # length
    if len(text) > int(cfg["guardrails"]["max_chars"]):
        raise HTTPException(status_code=400, detail=f"Input too long (>{cfg['guardrails']['max_chars']} chars).")

    # language
    allow = cfg["guardrails"]["allow_langs"] or []
    try:
        lang = detect(text) if text.strip() else "unknown"
    except LangDetectException:
        lang = "unknown"
    if allow and lang not in allow:
        reasons.append(f"lang={lang}")
        safety = "abstain"

    # toxicity
    if cfg["guardrails"]["toxicity_gate"]:
        tox_model = get_tox()
        if tox_model is None:
            # If Detoxify not installed, skip gate
            tox_val = 0.0
        else:
            tox_val = float(tox_model.predict(text)["toxicity"])
            if tox_val >= float(cfg["guardrails"]["toxicity_threshold"]):
                safety = "blocked"
                reasons.append("toxicity")

    return {"safety": safety, "reasons": reasons, "toxicity": tox_val}

# ---------- Rate limiting (optional) ----------
# Lightweight: we use SlowAPI if present in requirements.
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from fastapi.responses import JSONResponse

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    def ratelimit_handler(request: Request, exc):
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})
    _rate_limit_decorator = limiter.limit(get_cfg()["guardrails"]["rate_limit"])
except Exception:
    # Fallback no-op decorator
    def _rate_limit_decorator(rule):
        def deco(fn): return fn
        return deco

# ---------- Endpoints ----------

@app.get("/")
def root():
    return RedirectResponse(url="/ui") 

@app.get("/health")
def health():
    get_tokenizer_model()
    return {"status": "ok"}

@app.post("/analyze")
@_rate_limit_decorator
async def analyze(req: AnalyzeRequest, request: Request):
    g = guardrails_pre(req.text)
    if g["safety"] == "blocked":
        audit_event("blocked", {"text": req.text, "reasons": g["reasons"]})
        raise HTTPException(status_code=400, detail="Request blocked by safety policy")

    use_openai = bool(get_cfg().get("openai", {}).get("enabled"))
    pred = predict_with_openai(req.text) if use_openai else predict_with_controls(req.text)

    safety = g["safety"]
    reasons = list(g["reasons"])
    if pred["label"] == "abstain" and safety == "ok":
        safety = "abstain"
        reasons.append("low_confidence")

    resp = AnalyzeResponse(
        input=req.text,
        result=SentimentResult(label=pred["label"], score=float(pred["score"])),
        policy=PolicyBlock(safety=safety, reasons=reasons, toxicity=float(g["toxicity"])),
    )
    audit_event("analyze", {"text": req.text, "label": resp.result.label, "score": resp.result.score, "safety": safety})
    return resp

@app.post("/analyze/batch")
@_rate_limit_decorator
async def analyze_batch(req: AnalyzeBatchRequest, request: Request):
    results = []
    for t in req.texts:
        try:
            g = guardrails_pre(t)
            if g["safety"] == "blocked":
                results.append(AnalyzeResponse(
                    input=t,
                    result=SentimentResult(label="abstain", score=0.0),
                    policy=PolicyBlock(safety="blocked", reasons=g["reasons"], toxicity=float(g["toxicity"]))
                ))
                continue
            pred = predict_with_openai(t) if use_openai else predict_with_controls(t)
            safety = g["safety"]
            reasons = list(g["reasons"])
            if pred["label"] == "abstain" and safety == "ok":
                safety = "abstain"; reasons.append("low_confidence")
            results.append(AnalyzeResponse(
                input=t,
                result=SentimentResult(label=pred["label"], score=float(pred["score"])),
                policy=PolicyBlock(safety=safety, reasons=reasons, toxicity=float(g["toxicity"]))
            ))
        except HTTPException as e:
            results.append(AnalyzeResponse(
                input=t,
                result=SentimentResult(label="abstain", score=0.0),
                policy=PolicyBlock(safety="blocked", reasons=[str(e.detail)], toxicity=0.0)
            ))
    return {"results": [r.dict() for r in results]}




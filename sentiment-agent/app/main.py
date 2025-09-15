# app/main.py
"""
Core FastAPI application, including middleware, endpoints, and audit logging.
"""
import logging
import hashlib
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from dotenv import load_dotenv

# Local module imports
from . import config, models, guardrails, schemas

load_dotenv()

# --- App Setup ---
app = FastAPI(title="Sentiment Agent", version="1.1.0")
config.start_config_reloader()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_cfg()["server"]["cors_allow_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate Limiting (optional) ---
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    def ratelimit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})

    def _rate_limit_decorator(fn):
        return limiter.limit(config.get_cfg()["guardrails"]["rate_limit"])(fn)

except ImportError:
    def _rate_limit_decorator(fn):
        return fn

# --- Logging ---
logging.basicConfig(level=logging.INFO)
audit_log = logging.getLogger("audit")

def audit_event(kind: str, payload: dict):
    """Logs an audit event if enabled."""
    if not config.get_cfg()["guardrails"]["audit_log"]:
        return
    payload = dict(payload)
    if "text" in payload:
        payload["text_sha256"] = hashlib.sha256(payload["text"].encode()).hexdigest()
        del payload["text"]
    payload["ts"] = int(time.time())
    audit_log.info({"event": kind, **payload})

# --- Endpoints ---
app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")

@app.get("/")
def root():
    """Redirects the root path to the static UI."""
    return RedirectResponse(url="/ui")

@app.get("/health")
def health():
    """Health check endpoint. Pre-loads model on first call."""
    if not bool(config.get_cfg().get("openai", {}).get("enabled")):
        models.get_tokenizer_model()
    return {"status": "ok"}

@app.post("/analyze", response_model=schemas.AnalyzeResponse)
@_rate_limit_decorator
async def analyze(req: schemas.AnalyzeRequest, request: Request):
    """Analyzes the sentiment of a single piece of text."""
    g = guardrails.guardrails_pre(req.text)
    if g["safety"] == "blocked":
        audit_event("blocked", {"text": req.text, "reasons": g["reasons"]})
        raise HTTPException(status_code=400, detail="Request blocked by safety policy")

    cfg = config.get_cfg()
    use_openai = bool(cfg.get("openai", {}).get("enabled"))
    pred = models.predict_with_openai(req.text) if use_openai else models.predict_with_controls(req.text)

    safety = g["safety"]
    reasons = list(g["reasons"])
    if pred["label"] == "abstain" and safety == "ok":
        safety = "abstain"
        reasons.append("low_confidence")

    resp = schemas.AnalyzeResponse(
        input=req.text,
        result=schemas.SentimentResult(label=pred["label"], score=float(pred["score"])),
        policy=schemas.PolicyBlock(safety=safety, reasons=reasons, toxicity=float(g["toxicity"])),
    )
    audit_event("analyze", {"text": req.text, "label": resp.result.label, "score": resp.result.score, "safety": safety})
    return resp

@app.post("/analyze/batch")
@_rate_limit_decorator
async def analyze_batch(req: schemas.AnalyzeBatchRequest, request: Request):
    """Analyzes the sentiment of a batch of texts."""
    cfg = config.get_cfg()
    use_openai = bool(cfg.get("openai", {}).get("enabled"))
    results = []
    for t in req.texts:
        try:
            g = guardrails.guardrails_pre(t)
            if g["safety"] == "blocked":
                results.append(schemas.AnalyzeResponse(
                    input=t,
                    result=schemas.SentimentResult(label="abstain", score=0.0),
                    policy=schemas.PolicyBlock(safety="blocked", reasons=g["reasons"], toxicity=float(g["toxicity"]))
                ))
                continue

            pred = models.predict_with_openai(t) if use_openai else models.predict_with_controls(t)
            safety = g["safety"]
            reasons = list(g["reasons"])
            if pred["label"] == "abstain" and safety == "ok":
                safety = "abstain"
                reasons.append("low_confidence")
            
            results.append(schemas.AnalyzeResponse(
                input=t,
                result=schemas.SentimentResult(label=pred["label"], score=float(pred["score"])),
                policy=schemas.PolicyBlock(safety=safety, reasons=reasons, toxicity=float(g["toxicity"]))
            ))
        except HTTPException as e:
            results.append(schemas.AnalyzeResponse(
                input=t,
                result=schemas.SentimentResult(label="abstain", score=0.0),
                policy=schemas.PolicyBlock(safety="blocked", reasons=[str(e.detail)], toxicity=0.0)
            ))
    return {"results": [r.dict() for r in results]}
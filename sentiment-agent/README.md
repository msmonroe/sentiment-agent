# Sentiment Agent (FastAPI + Transformers)

A production-ready **Python web sentiment analysis agent** with configurable thresholds and **guardrails** (toxicity gate, language allow‑list, rate limiting, PII redaction, audit logging).

## Features
- FastAPI JSON API (`/health`, `/analyze`, `/analyze/batch`)
- RoBERTa sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Controls:** temperature, confidence threshold, neutrality band, abstain logic
- **Guardrails:** max input size, language filter, *optional* toxicity gate (Detoxify), *optional* PII redaction, rate limiting, audit log
- Lightweight HTML/JS frontend (`web/index.html`)
- Dockerfile + `config.toml` for runtime tuning

## Quickstart (Local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open web/index.html in your browser
```

## Docker
```bash
docker build -t sentiment-agent .
docker run --rm -p 8000:8000 -v $(pwd)/config.toml:/app/config.toml:ro sentiment-agent
# open web/index.html
```

## Configuration
Edit `config.toml` (hot‑reloaded every 10s by default):

```toml
[model]
id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
temperature = 1.0            # <1.0 sharper, >1.0 softer
conf_threshold = 0.55         # below => 'abstain'
neutral_band = 0.10           # |pos-neg| < band => 'neutral'

[guardrails]
max_chars = 2000
allow_langs = ["en"]
toxicity_gate = true
toxicity_threshold = 0.80
pii_redaction = false
rate_limit = "10/minute"      # slowapi format
audit_log = true

[server]
reload_config_seconds = 10
cors_allow_origins = ["*"]    # tighten for prod
```

## API
### `GET /health`
Warms model and returns `{ "status": "ok" }`.

### `POST /analyze`
```json
{ "text": "I love this!", "lang_hint": null }
```
Response:
```json
{
  "input": "I love this!",
  "result": {"label":"positive","score":0.997},
  "policy": {"safety":"ok","reasons":[],"toxicity":0.02}
}
```

### `POST /analyze/batch`
```json
{"texts":["bad","great"]}
```

## Notes
- First request downloads the model; allow 10–30s CPU cold start.
- For multilingual, set `model.id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"`.
- To disable toxicity, set `toxicity_gate = false` (no Detoxify weights needed).

## Deploy tips
- Put behind Nginx/Cloudflare; set `cors_allow_origins` to your domain.
- Use `--workers` > 1 on larger machines; pin torch CPU threads with `OMP_NUM_THREADS` if needed.

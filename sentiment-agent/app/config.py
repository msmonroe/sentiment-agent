# app/config.py
"""Handles loading and accessing the application configuration."""
import os
import threading
import time
import tomllib
from typing import Any, Dict

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

def _load_config():
    """Loads configuration from a TOML file and merges it with defaults."""
    global _config
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "rb") as f:
                user_cfg = tomllib.load(f)
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

def get_cfg() -> Dict[str, Any]:
    """Thread-safe access to the global configuration."""
    with _config_lock:
        return _config

def start_config_reloader():
    """Starts a background thread to periodically reload the configuration."""
    def loop():
        while True:
            _load_config()
            time.sleep(get_cfg()["server"]["reload_config_seconds"])
    t = threading.Thread(target=loop, daemon=True)
    t.start()

# Initial load
_load_config()
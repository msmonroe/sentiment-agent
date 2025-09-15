# app/schemas.py
"""Data schemas (Pydantic models) for the API."""
from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    """Request model for a single text analysis."""
    text: str
    lang_hint: Optional[str] = None

class AnalyzeBatchRequest(BaseModel):
    """Request model for batch text analysis."""
    texts: List[str]
    lang_hint: Optional[str] = None

class SentimentResult(BaseModel):
    """Result of a sentiment analysis prediction."""
    label: str
    score: float

class PolicyBlock(BaseModel):
    """Details of the policy decision for a request."""
    safety: str           # "ok" | "abstain" | "blocked"
    reasons: List[str] = []
    toxicity: float = 0.0

class AnalyzeResponse(BaseModel):
    """Full response for a single text analysis."""
    input: str
    result: SentimentResult
    policy: PolicyBlock
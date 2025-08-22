from pydantic import BaseModel,Field 
from typing import List, Optional

class IndicatorSnapshot(BaseModel):
    price: float
    change_pct_1d: float
    volume_zscore: float
    volatility_20d: float
    gap_open_pct: float

class NewsItem(BaseModel):
    title: str
    url: str
    published: Optional[str] = None   
    source: Optional[str] = None
    summary: Optional[str] = None

class Evidence(BaseModel):
    source: Optional[str] = None
    url: Optional[str] = None
    quote: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None

class Analysis(BaseModel):
    title: str
    summary: str
    quote: Optional[str] = None

class RiskItem(BaseModel):
    name: str
    rationale: str
    severity: str  # "low" | "medium" | "high"
    evidences: List[Evidence] = []

class Thesis(BaseModel):
    viewpoint: str            # "bullish" | "bearish" | "neutral"
    reasoning: List[str]      # bullet points
    catalysts: List[str] = []
    risks: List[RiskItem] = []
    confidence_0_1: float = Field(ge=0.0, le=1.0)

class LLMReport(BaseModel):
    ticker: str
    date: str
    indicators: IndicatorSnapshot
    top_news: List[NewsItem]
    thesis: Thesis
    
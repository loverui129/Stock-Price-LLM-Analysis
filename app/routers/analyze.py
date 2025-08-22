
import re
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException

from ..services.market import fetch_price_df, compute_indicators
from ..services.news import fetch_rss_headlines
from ..services.llm import analyze_with_llm
from ..schemas.analysis import LLMReport, IndicatorSnapshot, NewsItem, Thesis
from ..services.rag import index_headlines, search_evidences


router = APIRouter(prefix="/analyze", tags=["analyze"])
logger = logging.getLogger("router.analyze")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------- 简易 10 分钟 TTL 缓存（无需第三方库） ----------
_CACHE_TTL_SEC = 10 * 60
# 缓存结构：{ key: (expire_ts, payload_dict) }
_CACHE: Dict[str, Any] = {}

def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    expire_ts, data = item
    if time.time() > expire_ts:
        _CACHE.pop(key, None)
        return None
    return data

def _cache_set(key: str, data: Dict[str, Any], ttl: int = _CACHE_TTL_SEC):
    _CACHE[key] = (time.time() + ttl, data)

# ---------- 输入校验 ----------
_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{0,9}$")

def _validate_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="Invalid ticker format.")
    return t

# ---------- 工具：容错修正 ----------
def _coerce_thesis(raw: dict) -> dict:
    t = raw or {}
    t.setdefault("viewpoint", "neutral")
    t.setdefault("reasoning", [])
    t.setdefault("catalysts", [])
    t.setdefault("risks", [])
    fixed_risks = []
    for r in (t.get("risks") or []):
        r = r or {}
        # evidences 归一化
        evs = r.get("evidences", []) or []
        fixed_evs = []
        for e in evs:
            e = e or {}
            e.setdefault("source", None)
            e.setdefault("url", None)
            e.setdefault("quote", None)
            e.setdefault("title", None)
            e.setdefault("summary", None)
            fixed_evs.append(e)
        r["evidences"] = fixed_evs
        if r.get("severity") not in {"low", "medium", "high"}:
            r["severity"] = r.get("severity") or "medium"
        fixed_risks.append(r)
    t["risks"] = fixed_risks
    t.setdefault("confidence_0_1", 0.5)
    return t

def _coerce_news_item(n: dict) -> dict:
    n = n or {}
    return {
        "title": n.get("title") or "",
        "url": n.get("url") or "",
        "published": n.get("published") or n.get("published_at") or None,
        "source": n.get("source") or "",
        "summary": n.get("summary") or None,
    }

def _rss_sources_for(ticker: str) -> List[str]:
    return [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=news",
        "https://www.marketwatch.com/feeds/topstories",
    ]

@router.get("/{ticker}", response_model=LLMReport)
def analyze_ticker(ticker: str):
    # 1) 指标
    try:
        df = fetch_price_df(ticker)
        indicators = compute_indicators(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) 新闻
    try:
        headlines = fetch_rss_headlines(_rss_sources_for(ticker), limit=8)
    except Exception:
        headlines = []

    # 3) RAG：把最新 headlines 入库，然后做一次相似度检索拿证据
    try:
        index_headlines(ticker, headlines)
        # 简单把近期“风险相关”关键词放入查询（也可根据 indicators 动态拼接）
        rag_query = f"{ticker} stock risks volatility earnings regulation macro AI rout"
        evidences = search_evidences(ticker, rag_query, k=5)
    except Exception:
        evidences = []

    # 4) LLM
    try:
        raw = analyze_with_llm(ticker, indicators, headlines, evidences)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    # 5) 组装响应（仍然以本地指标 + 我们抓到的新闻为准）
    try:
        raw_news = raw.get("top_news") or headlines
        raw_thesis = _coerce_thesis(raw.get("thesis") or {})

        report = LLMReport(
            ticker=ticker.upper(),
            date=datetime.now(timezone.utc).isoformat(),
            indicators=IndicatorSnapshot(**indicators),
            top_news=[NewsItem(**_coerce_news_item(n)) for n in raw_news],
            thesis=Thesis(**raw_thesis),
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid LLM payload: {e}")
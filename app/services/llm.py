# app/services/llm.py  (LangChain + RAG 证据池)
import os, json
from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..schemas.analysis import Thesis  # Thesis 内含 RiskItem/Evidence 等


class ThesisOnly(BaseModel):
    thesis: Thesis


_PROMPT = """
You are a sell-side style equity analyst. Read the signals and the recent headlines,
and produce an investment thesis JSON strictly matching the schema.

Ticker: {ticker}

Indicators (JSON):
{indicators_json}

Top headlines (one per line as: title | source | published | url):
{headlines_bullets}

Evidence pool (each line is: summary | source | url):
{evidences_bullets}

Rules:
- thesis.viewpoint must be one of: "bullish", "bearish", "neutral".
- thesis.reasoning: 2–4 concise bullet points.
- thesis.catalysts: 1–3 concise items.
- thesis.risks: 1–3 items, each with fields:
  - name (short),
  - rationale (1–2 sentences),
  - severity in ["low","medium","high"],
  - evidences: 0–2 items, each with fields source, url, summary.
- Use ONLY the above evidence pool for evidences; DO NOT fabricate sources or URLs.
- thesis.confidence_0_1: a float in [0,1].

Return JSON ONLY. No commentary.
""".strip()


def _format_headlines(headlines: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for h in headlines[:8]:
        title = h.get("title", "") or ""
        source = h.get("source", "") or ""
        published = h.get("published") or h.get("published_at") or ""
        url = h.get("url", "") or ""
        lines.append(f"- {title} | {source} | {published} | {url}")
    return "\n".join(lines) if lines else "- (no headlines)"


def _format_evidences(evs: Optional[List[Dict[str, Any]]]) -> str:
    if not evs:
        return "- (no evidences)"
    lines = []
    for e in evs[:8]:
        summary = (e.get("summary") or "").replace("\n", " ").strip()
        source = e.get("source", "") or ""
        url = e.get("url", "") or ""
        lines.append(f"- {summary} | {source} | {url}")
    return "\n".join(lines)


def analyze_with_llm(
    ticker: str,
    indicators: Dict[str, Any],
    headlines: List[Dict[str, Any]],
    evidences: Optional[List[Dict[str, Any]]] = None,  # ⬅️ RAG 检索来的证据池
):
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    tmpl = PromptTemplate(
        template=_PROMPT,
        input_variables=["ticker", "indicators_json", "headlines_bullets", "evidences_bullets"],
    )
    chain = tmpl | llm.with_structured_output(ThesisOnly)

    payload = chain.invoke({
        "ticker": ticker,
        "indicators_json": json.dumps(indicators, ensure_ascii=False),
        "headlines_bullets": _format_headlines(headlines),
        "evidences_bullets": _format_evidences(evidences),
    })

    return {"thesis": payload.thesis.model_dump()}

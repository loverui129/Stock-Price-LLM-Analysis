# ui/app.py
import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Stock LLM Dashboard", layout="wide")

# --- Header ---
st.title("📊 Stock Price & LLM Analysis")
st.caption("Backend: FastAPI  •  Frontend: Streamlit  •  Model: OpenAI")

# --- Sidebar / Input ---
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", value="TSLA").strip().upper()
    period = st.selectbox("History period", ["3mo", "6mo", "1y"], index=1)
    submit = st.button("Analyze")

# Small helper to format percentage
def pct(x):
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "-"

def load_analysis(ticker: str):
    url = f"{API_BASE}/analyze/{ticker}"
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    return r.json()

def load_price_history(ticker: str, period: str = "6mo"):
    
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None

 
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns:
        
        if "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        else:
            
            df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in col if c not in (None, "", "None")]).strip("_")
            for col in df.columns
        ]

  
    if "Close" not in df.columns:
        
        cand = next((c for c in df.columns if str(c).lower().startswith("close")), None)
        if cand:
            df = df.rename(columns={cand: "Close"})
        elif "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        else:
           
            return None

    
    out = df[["date", "Close"]].dropna()
    
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out

# Auto run once
if submit or ticker:
    try:
        with st.spinner(f"Calling API … {ticker}"):
            data = load_analysis(ticker)
            time.sleep(0.1)

        # --- Top row: Price chart & Indicators ---
        col_left, col_right = st.columns([2.2, 1.3])

        # Price chart (history)
        with col_left:
            st.subheader(f"📈 {ticker} — Price History ({period})")
            hist = load_price_history(ticker, period=period)
            if hist is None or hist.empty:
                st.info("No history data.")
            else:
                fig = px.line(hist, x="date", y="Close", title=None)
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True)

        # Indicators
        with col_right:
            st.subheader("🔎 Indicators")
            ind = data.get("indicators", {})
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Price", f"{ind.get('price', 0):,.2f}")
                st.metric("Vol Zscore", f"{ind.get('volume_zscore', 0):.2f}")
                st.metric("Gap Open", pct(ind.get("gap_open_pct", 0)))
            with m2:
                st.metric("Change 1D", pct(ind.get("change_pct_1d", 0)))
                st.metric("Volatility 20d", f"{ind.get('volatility_20d', 0):.3f}")
                st.metric("Updated (UTC)", data.get("date", ""))

        st.divider()

        # --- News list ---
        st.subheader("📰 Top News")
        news = data.get("top_news", [])
        if not news:
            st.info("No news fetched.")
        else:
            # show as a dataframe-like table with links
            rows = []
            for n in news:
                rows.append({
                    "Title": f"[{n.get('title','')}]({n.get('url','')})",
                    "Source": n.get("source",""),
                    "Published": n.get("published","") or n.get("published_at",""),
                })
            df_news = pd.DataFrame(rows)
            st.markdown(df_news.to_markdown(index=False), unsafe_allow_html=True)

        st.divider()

        # --- LLM Thesis ---
        st.subheader("🤖 LLM Thesis")
        th = data.get("thesis", {}) or {}
        viewpoint = th.get("viewpoint", "neutral")
        conf = th.get("confidence_0_1", 0.5)

        # viewpoint badge
        vp_color = {"bullish": "green", "bearish": "red", "neutral": "gray"}.get(viewpoint, "blue")
        st.markdown(f"**Viewpoint:** <span style='color:{vp_color}'>{viewpoint.upper()}</span> &nbsp;&nbsp; **Confidence:** {conf:.2f}", unsafe_allow_html=True)

        # reasoning
        reasoning = th.get("reasoning", [])
        if reasoning:
            st.markdown("**Reasoning**")
            for r in reasoning:
                st.write(f"- {r}")

        # catalysts
        catalysts = th.get("catalysts", [])
        if catalysts:
            st.markdown("**Catalysts**")
            for c in catalysts:
                st.write(f"- {c}")

        # risks
        risks = th.get("risks", [])
        if risks:
            st.markdown("**Risks**")
            for r in risks:
                name = r.get("name","")
                sev = r.get("severity","")
                rationale = r.get("rationale","")
                st.write(f"- **{name}** (severity: {sev}) — {rationale}")

                evs = r.get("evidences", [])
                for e in evs:
                    src = e.get("source","")
                    url = e.get("url","")
                    quote = e.get("quote","")
                    bullet = f"    - _{src}_"
                    if url:
                        bullet += f": [{url}]({url})"
                    if quote:
                        bullet += f" — “{quote}”"
                    st.markdown(bullet)

    except requests.HTTPError as http_err:
        st.error(f"API error: {http_err.response.status_code} — {http_err.response.text[:400]}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

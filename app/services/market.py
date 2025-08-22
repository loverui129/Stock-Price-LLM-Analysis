import yfinance as yf
import numpy as np
import pandas as pd

def fetch_price_df(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False)
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    if df is None or df.empty:
        raise ValueError(f"No market data for {ticker}. Try another symbol (e.g., MSFT) or retry later.")

    
    df = df.copy()

    # 情况 A：MultiIndex（('Open','TSLA') 这种）
    if isinstance(df.columns, pd.MultiIndex):
        # 取第 1 层（Open/High/Low/Close/Adj Close/Volume）
        df.columns = [c[0] for c in df.columns]

    # 情况 B：已经被拼成了 'Open_TSLA' 这种
    if any(col.endswith(f"_{ticker}") for col in df.columns):
        newcols = {}
        for col in df.columns:
            if col.endswith(f"_{ticker}"):
                base = col[:-(len(ticker)+1)]  # 去掉 '_TSLA'
                newcols[col] = base
        df = df.rename(columns=newcols)

    # 到这里我们希望有标准列名
    needed = ["Open", "High", "Low", "Close", "Volume"]
    have = set(df.columns)
    # 个别情况下 auto_adjust=True 只给 'Adj Close'，补一份 Close
    if "Close" not in have and "Adj Close" in have:
        df["Close"] = df["Adj Close"]

    # 最终检查
    if not set(["Open", "Close", "Volume"]).issubset(df.columns):
        raise ValueError(f"Incomplete columns for {ticker}: got {list(df.columns)}")

    # 去掉关键列为空的行，并只保留最近 60 根（加速）
    df = df.dropna(subset=["Open", "Close", "Volume"])
    if len(df) > 60:
        df = df.tail(60)

    return df

def compute_indicators(df: pd.DataFrame) -> dict:
    # Ensure we have enough rows
    if df is None or df.empty:
        raise ValueError("Empty market data")

    close = df["Close"]
    vol = df["Volume"].astype(float)

    # 1) daily return
    ret_1d = close.pct_change()

    # 2) 20d volatility (annualized-ish) — handle NaN safely
    ret20_std = ret_1d.rolling(20).std().iloc[-1]
    vol20 = float(ret20_std * np.sqrt(252)) if pd.notna(ret20_std) else 0.0

    # 3) Volume z-score vs last 20d — handle NaN/0 safely
    vol_mean = vol.rolling(20).mean().iloc[-1]
    vol_std = vol.rolling(20).std().iloc[-1]
    if pd.isna(vol_mean) or pd.isna(vol_std) or vol_std == 0:
        vol_z = 0.0
    else:
        vol_z = float((float(vol.iloc[-1]) - float(vol_mean)) / float(vol_std))

    # 4) Gap open %
    gap_pct = 0.0
    if df.shape[0] >= 2:
        prev_close = float(close.iloc[-2])
        today_open = float(df["Open"].iloc[-1])
        if prev_close != 0.0:
            gap_pct = (today_open / prev_close) - 1.0

    return {
        "price": float(close.iloc[-1]),
        "change_pct_1d": float(ret_1d.iloc[-1]) if pd.notna(ret_1d.iloc[-1]) else 0.0,
        "volume_zscore": float(vol_z),
        "volatility_20d": float(vol20),
        "gap_open_pct": float(gap_pct),
    }

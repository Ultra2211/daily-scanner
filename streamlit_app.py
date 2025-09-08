import streamlit as st 
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Scanner — Full-Width Charts (Stocks & FX)", layout="wide")

# ===================== TradingView embed (FULL WIDTH) =====================
def tradingview_widget_full(symbol: str, interval: str = "D", theme: str = "light", height: int = 950):
    """
    Embeds a TradingView chart that stretches to full browser width.
    Uses the lightweight widgetembed iframe with explicit width=100%.
    NOTE: Embedded widgets cannot access your TradingView account (no saved layouts/indicators).
    """
    safe_id = symbol.replace(":", "_").replace("/", "_")
    tv = f"""
    <iframe
        id="tradingview_{safe_id}"
        src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{safe_id}&symbol={symbol}&interval={interval}&theme={theme}&style=1&locale=en&allow_symbol_change=1&withdateranges=1&details=1"
        width="100%"
        height="{height}"
        frameborder="0"
        allowtransparency="true"
        scrolling="no">
    </iframe>
    """
    st.markdown(tv, unsafe_allow_html=True)

def tradingview_direct_link(symbol: str, interval: str = "D") -> str:
    """
    Builds a direct TradingView URL. Opening this in a new tab uses your
    TradingView login session, so your saved indicators/layout appear.
    """
    # Example: https://www.tradingview.com/chart/?symbol=NASDAQ:AAPL&interval=D
    return f"https://www.tradingview.com/chart/?symbol={symbol}&interval={interval}"

# ===================== Sidebar Settings =====================
st.sidebar.title("Scanner Settings")

st.sidebar.subheader("Signal Logic")
lookback_breakout = st.sidebar.slider("Breakout lookback (days)", 20, 100, 55, 5)
ema_short = st.sidebar.slider("EMA short", 5, 50, 20, 1)
ema_mid   = st.sidebar.slider("EMA mid", 10, 100, 50, 1)
ema_long  = st.sidebar.slider("EMA long", 50, 300, 200, 5)
rsi_len   = st.sidebar.slider("RSI length", 7, 30, 14, 1)
rsi_min   = st.sidebar.slider("RSI min", 45, 60, 52, 1)
rsi_max   = st.sidebar.slider("RSI max", 65, 85, 70, 1)
min_atr_pc = st.sidebar.slider("Min ATR%", 0.2, 5.0, 0.6, 0.1)
max_atr_pc = st.sidebar.slider("Max ATR%", 1.0, 15.0, 6.0, 0.5)

st.sidebar.subheader("Output")
top_n = st.sidebar.slider("Top picks per board", 3, 15, 8, 1)
chart_interval = st.sidebar.selectbox("Chart interval", ["D","240","60","30","W"], index=0)
theme = st.sidebar.selectbox("Chart theme", ["light","dark"], index=0)
chart_height_px = st.sidebar.slider("Chart height (px)", 700, 1400, 950, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("TradingView Prefixes")
stocks_prefix = st.sidebar.selectbox("Stocks TV prefix", ["NASDAQ","NYSE"], index=0)
fx_prefix = st.sidebar.selectbox("FX TV prefix", ["FX","OANDA","FOREXCOM"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Universes")
stocks_custom = st.sidebar.text_area("Stocks (Yahoo tickers, comma-separated):", value="")
fx_custom = st.sidebar.text_area("FX pairs (EURUSD, GBPUSD, ... comma-separated):", value="")

# ===================== Universes =====================
DEFAULT_STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM",
    "XOM","UNH","LLY","V","MA","HD","ABBV","PG","COST","PEP","NFLX","KO","MRK"
]
DEFAULT_FX = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD","EURGBP","EURJPY","GBPJPY"]

stocks_universe = [s.strip().upper() for s in (stocks_custom.split(",") if stocks_custom else DEFAULT_STOCKS) if s.strip()]
fx_universe = [s.strip().upper() for s in (fx_custom.split(",") if fx_custom else DEFAULT_FX) if s.strip()]

# ===================== Helpers =====================
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def atr(high, low, close, length=14):
    hl = (high - low).abs()
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def tv_stock_symbol(prefix: str, yahoo_sym: str) -> str:
    # Convert Yahoo tickers like 'BRK-B' -> TradingView 'BRK.B'
    tv_core = yahoo_sym.replace("-", ".")
    return f"{prefix}:{tv_core}"

def tv_fx_symbol(prefix: str, pair: str) -> str:
    return f"{prefix}:{pair}"

@st.cache_data(show_spinner=False)
def fetch_yf(symbols, is_fx=False):
    end = datetime.utcnow()
    start = end - timedelta(days=400)
    if is_fx:
        y_symbols = [f"{s}=X" for s in symbols]
        data = yf.download(y_symbols, start=start.date(), end=end.date(), auto_adjust=False, progress=False, group_by='ticker', threads=True)
        return data
    else:
        data = yf.download(symbols, start=start.date(), end=end.date(), auto_adjust=False, progress=False, group_by='ticker', threads=True)
        return data

def get_df(data, sym, is_fx=False):
    key = f"{sym}=X" if is_fx else sym
    if key not in data: return None
    return data[key].rename(columns=str.title).dropna()

def score_rows(symbols, data, is_fx=False):
    rows = []
    for s in symbols:
        df = get_df(data, s, is_fx)
        if df is None or len(df) < 220:
            continue
        df["EMA_S"] = df["Close"].ewm(span=ema_short, adjust=False).mean()
        df["EMA_M"] = df["Close"].ewm(span=ema_mid, adjust=False).mean()
        df["EMA_L"] = df["Close"].ewm(span=ema_long, adjust=False).mean()
        df["RSI"]   = rsi(df["Close"], rsi_len)
        df["ATR"]   = atr(df["High"], df["Low"], df["Close"], 14)
        df["ATR%"]  = (df["ATR"] / df["Close"]) * 100.0
        df["HH_N"]  = df["High"].rolling(lookback_breakout).max()

        last = df.iloc[-1]
        trend_align = last["EMA_S"] > last["EMA_M"] > last["EMA_L"]
        breakout    = last["Close"] >= last["HH_N"]
        rsi_ok      = (last["RSI"] >= rsi_min) and (last["RSI"] <= rsi_max)
        atr_ok      = (last["ATR%"] >= min_atr_pc) and (last["ATR%"] <= max_atr_pc)

        score = (3 if trend_align else 0) + (2 if breakout else 0) + (2 if rsi_ok else 0) + (1 if atr_ok else 0)

        rows.append({
            "Symbol": s,
            "Score": int(score),
            "TrendAlign": bool(trend_align),
            "Breakout": bool(breakout),
            "RSI": round(float(last["RSI"]), 1),
            "ATR%": round(float(last["ATR%"]), 2)
        })
    if not rows:
        return pd.DataFrame(columns=["Symbol","Score","TrendAlign","Breakout","RSI","ATR%"])
    return pd.DataFrame(rows).sort_values(["Score","ATR%"], ascending=[False, False])

# ===================== Fetch & Score =====================
with st.spinner("Downloading STOCKS data..."):
    stocks_data = fetch_yf(stocks_universe, is_fx=False)
with st.spinner("Downloading FX data..."):
    fx_data = fetch_yf(fx_universe, is_fx=True)

rank_stocks = score_rows(stocks_universe, stocks_data, is_fx=False).head(top_n)
rank_fx     = score_rows(fx_universe, fx_data, is_fx=True).head(top_n)

# ===================== Two Boards (tables in columns) =====================
st.title("📊 Daily Scanner — Stocks & Forex (Full-Width Charts)")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("Top Stocks")
    if rank_stocks.empty:
        st.info("No stocks passed the filters today.")
        selected_stock = None
    else:
        st.dataframe(rank_stocks, use_container_width=True)
        selected_stock = st.selectbox("Choose stock:", rank_stocks["Symbol"].tolist(), index=0, key="stock_sel")

with col_right:
    st.subheader("Top Forex")
    if rank_fx.empty:
        st.info("No FX pairs passed the filters today.")
        selected_fx = None
    else:
        st.dataframe(rank_fx, use_container_width=True)
        selected_fx = st.selectbox("Choose FX:", rank_fx["Symbol"].tolist(), index=0, key="fx_sel")

# ===================== FULL-WIDTH CHARTS (no columns) =====================
st.markdown("---")

if selected_stock:
    tv_sym_stock = tv_stock_symbol(stocks_prefix, selected_stock)
    st.subheader(f"Stock chart — {tv_sym_stock}")
    tradingview_widget_full(
        symbol=tv_sym_stock,
        interval=chart_interval,
        theme=theme,
        height=chart_height_px
    )
    # --- Open in TradingView (loads your login + saved indicators) ---
    url_stock = tradingview_direct_link(tv_sym_stock, chart_interval)
    st.link_button("Open in TradingView (with my indicators)", url_stock)

if selected_fx:
    tv_sym_fx = tv_fx_symbol(fx_prefix, selected_fx)
    st.subheader(f"FX chart — {tv_sym_fx}")
    tradingview_widget_full(
        symbol=tv_sym_fx,
        interval=chart_interval,
        theme=theme,
        height=chart_height_px
    )
    # --- Open in TradingView (loads your login + saved indicators) ---
    url_fx = tradingview_direct_link(tv_sym_fx, chart_interval)
    st.link_button("Open in TradingView (with my indicators)", url_fx)

st.markdown("---")
st.caption(
    "Embedded charts cannot access your TradingView account. "
    "Use the “Open in TradingView” button to load your saved indicators/layout while logged in."
)


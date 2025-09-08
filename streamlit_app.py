import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Scanner — Full Charts", layout="wide")

# ===================== TradingView embed =====================
def tradingview_widget(symbol: str, interval: str = "D", theme: str = "light", height: int = 720):
    """
    Embed a full-width TradingView chart (takes entire Streamlit container).
    """
    tv = f'''
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tradingview_chart_{symbol.replace(':','_')}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "{interval}",
        "timezone": "Etc/UTC",
        "theme": "{theme}",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "withdateranges": true,
        "allow_symbol_change": true,
        "details": true,
        "studies": [],
        "container_id": "tradingview_chart_{symbol.replace(':','_')}"
      }});
      </script>
    </div>
    '''
    st.components.v1.html(tv, height=height, scrolling=False)

# ===================== Sidebar Settings =====================
st.sidebar.title("Scanner Settings")
lookback_breakout = st.sidebar.slider("Breakout lookback (days)", 20, 100, 55, 5)
ema_short = st.sidebar.slider("EMA short", 5, 50, 20, 1)
ema_mid   = st.sidebar.slider("EMA mid", 10, 100, 50, 1)
ema_long  = st.sidebar.slider("EMA long", 50, 300, 200, 5)
rsi_len   = st.sidebar.slider("RSI length", 7, 30, 14, 1)
rsi_min   = st.sidebar.slider("RSI min", 45, 60, 52, 1)
rsi_max   = st.sidebar.slider("RSI max", 65, 85, 70, 1)
min_atr_pc = st.sidebar.slider("Min ATR%", 0.2, 5.0, 0.6, 0.1)
max_atr_pc = st.sidebar.slider("Max ATR%", 1.0, 15.0, 6.0, 0.5)

top_n = st.sidebar.slider("Top picks per board", 3, 15, 8, 1)
chart_interval = st.sidebar.selectbox("Chart interval", ["D","240","60","30","W"], index=0)
theme = st.sidebar.selectbox("Chart theme", ["light","dark"], index=0)

st.sidebar.markdown("---")
stocks_prefix = st.sidebar.selectbox("Stocks prefix", ["NASDAQ","NYSE"], index=0)
fx_prefix = st.sidebar.selectbox("FX prefix", ["FX","OANDA","FOREXCOM"], index=0)

# ===================== Universes =====================
DEFAULT_STOCKS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","JPM","XOM"]
DEFAULT_FX = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"]

def tv_stock_symbol(prefix: str, yahoo_sym: str) -> str:
    return f"{prefix}:{yahoo_sym.replace('-', '.')}"

def tv_fx_symbol(prefix: str, pair: str) -> str:
    return f"{prefix}:{pair}"

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

def fetch(symbols, is_fx=False):
    end = datetime.utcnow()
    start = end - timedelta(days=400)
    if is_fx:
        y_symbols = [f"{s}=X" for s in symbols]
        data = yf.download(y_symbols, start=start.date(), end=end.date(), progress=False, group_by='ticker', threads=True)
        return data, {f"{s}=X": s for s in symbols}
    else:
        data = yf.download(symbols, start=start.date(), end=end.date(), progress=False, group_by='ticker', threads=True)
        return data, {}

def get_df(data, sym, is_fx=False):
    key = f"{sym}=X" if is_fx else sym
    if key not in data: return None
    return data[key].rename(columns=str.title).dropna()

def score_rows(symbols, data, is_fx=False):
    rows = []
    for s in symbols:
        df = get_df(data, s, is_fx)
        if df is None or len(df) < 220: continue
        df["EMA_S"] = df["Close"].ewm(span=ema_short, adjust=False).mean()
        df["EMA_M"] = df["Close"].ewm(span=ema_mid, adjust=False).mean()
        df["EMA_L"] = df["Close"].ewm(span=ema_long, adjust=False).mean()
        df["RSI"] = rsi(df["Close"], rsi_len)
        df["ATR"] = atr(df["High"], df["Low"], df["Close"], 14)
        df["ATR%"] = (df["ATR"] / df["Close"]) * 100.0
        df["HH_N"] = df["High"].rolling(lookback_breakout).max()

        last = df.iloc[-1]
        score = (
            (3 if last["EMA_S"] > last["EMA_M"] > last["EMA_L"] else 0) +
            (2 if last["Close"] >= last["HH_N"] else 0) +
            (2 if rsi_min <= last["RSI"] <= rsi_max else 0) +
            (1 if min_atr_pc <= last["ATR%"] <= max_atr_pc else 0)
        )
        rows.append({"Symbol": s, "Score": int(score), "RSI": round(last["RSI"],1), "ATR%": round(last["ATR%"],2)})
    return pd.DataFrame(rows).sort_values("Score", ascending=False).head(top_n)

# ===================== Data Fetch =====================
with st.spinner("Loading stocks..."):
    stocks_data, _ = fetch(DEFAULT_STOCKS, is_fx=False)
with st.spinner("Loading forex..."):
    fx_data, _ = fetch(DEFAULT_FX, is_fx=True)

rank_stocks = score_rows(DEFAULT_STOCKS, stocks_data, is_fx=False)
rank_fx = score_rows(DEFAULT_FX, fx_data, is_fx=True)

# ===================== Layout =====================
st.title("📊 Daily Scanner — Stocks & Forex with Full Charts")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Stocks")
    st.dataframe(rank_stocks, use_container_width=True)
    selected_stock = st.selectbox("Choose stock:", rank_stocks["Symbol"].tolist(), index=0)
with col2:
    st.subheader("Top Forex")
    st.dataframe(rank_fx, use_container_width=True)
    selected_fx = st.selectbox("Choose FX:", rank_fx["Symbol"].tolist(), index=0)

st.markdown("---")

# Big full-width charts below
if selected_stock:
    st.subheader(f"Stock chart — {stocks_prefix}:{selected_stock}")
    tradingview_widget(tv_stock_symbol(stocks_prefix, selected_stock), interval=chart_interval, theme=theme, height=720)

if selected_fx:
    st.subheader(f"FX chart — {fx_prefix}:{selected_fx}")
    tradingview_widget(tv_fx_symbol(fx_prefix, selected_fx), interval=chart_interval, theme=theme, height=720)

# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Scanner (Stocks & FX) + TradingView", layout="wide")

# ---------------------- TradingView embed ----------------------
def tradingview_widget(symbol: str, interval: str = "D", theme: str = "light", height: int = 610):
    tv = f'''
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
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
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    '''
    st.components.v1.html(tv, height=height, scrolling=False)

# ---------------------- Universe ----------------------
DEFAULT_STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM",
    "XOM","UNH","LLY","V","MA","HD","ABBV","PG","COST","PEP","NFLX","KO","MRK"
]
DEFAULT_FX = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD","EURGBP","EURJPY","GBPJPY"]

st.sidebar.title("Scanner Settings")
mode = st.sidebar.selectbox("Asset Class", ["Stocks (US)", "Forex (FX)"])

custom_list = st.sidebar.text_area(
    "Custom Symbols (comma-separated). Stocks: Yahoo tickers (AAPL). FX: raw pairs (EURUSD). Leave blank for defaults.",
    value=""
)

if mode == "Stocks (US)":
    symbol_list = [s.strip().upper() for s in (custom_list.split(",") if custom_list else DEFAULT_STOCKS) if s.strip()]
    tv_prefix = st.sidebar.selectbox("TradingView prefix for chart", ["NASDAQ", "NYSE"], index=0)
    universe_type = "stocks"
else:
    symbol_list = [s.strip().upper() for s in (custom_list.split(",") if custom_list else DEFAULT_FX) if s.strip()]
    tv_prefix = st.sidebar.selectbox("TradingView FX prefix", ["FX","OANDA","FOREXCOM"], index=0)
    universe_type = "fx"

st.sidebar.markdown("---")
st.sidebar.subheader("Signal Logic")
lookback_breakout = st.sidebar.slider("Breakout lookback (days)", 20, 100, 55, 5)
ema_short = st.sidebar.slider("EMA short", 5, 50, 20, 1)
ema_mid   = st.sidebar.slider("EMA mid", 10, 100, 50, 1)
ema_long  = st.sidebar.slider("EMA long", 50, 300, 200, 5)
rsi_len   = st.sidebar.slider("RSI length", 7, 30, 14, 1)
rsi_min   = st.sidebar.slider("RSI lower bound (long bias)", 45, 60, 52, 1)
rsi_max   = st.sidebar.slider("RSI upper bound (avoid overheating)", 65, 85, 70, 1)

st.sidebar.subheader("Filters")
min_atr_pc = st.sidebar.slider("Min ATR% (volatility floor)", 0.2, 5.0, 0.6, 0.1)
max_atr_pc = st.sidebar.slider("Max ATR% (volatility cap)", 1.0, 15.0, 6.0, 0.5)
min_avg_dollar_vol = st.sidebar.number_input("Min $ volume (stocks only, M USD)", 0.0, 5000.0, 50.0, 10.0)

st.sidebar.subheader("Output")
top_n = st.sidebar.slider("How many top picks?", 3, 15, 8, 1)
chart_interval = st.sidebar.selectbox("Chart interval", ["D","240","60","30","W"], index=0)
theme = st.sidebar.selectbox("Chart theme", ["light","dark"], index=0)

st.title("📈 Daily Scanner — Stocks & FX (with TradingView chart)")

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

@st.cache_data(show_spinner=False)
def fetch(symbols, universetype):
    end = datetime.utcnow()
    start = end - timedelta(days=400)
    # yfinance FX tickers use '=X'
    y_symbols = []
    fx_map = {}
    for s in symbols:
        if universetype == "fx":
            ysym = f"{s}=X"
            y_symbols.append(ysym)
            fx_map[ysym] = s
        else:
            y_symbols.append(s)
    data = yf.download(y_symbols, start=start.date(), end=end.date(), auto_adjust=False, progress=False, group_by='ticker', threads=True)
    return data, fx_map

with st.spinner("Downloading data..."):
    data, fx_map = fetch(symbol_list, universe_type)

def get_df_for(sym):
    if universe_type == "fx":
        ysym = f"{sym}=X"
        if ysym not in data:
            return None
        return data[ysym].rename(columns=str.title)
    else:
        if sym not in data:
            return None
        return data[sym].rename(columns=str.title)

rows = []
for s in symbol_list:
    df = get_df_for(s)
    if df is None or len(df) < 220:
        continue
    df = df.dropna()
    df["EMA_S"] = df["Close"].ewm(span=ema_short, adjust=False).mean()
    df["EMA_M"] = df["Close"].ewm(span=ema_mid, adjust=False).mean()
    df["EMA_L"] = df["Close"].ewm(span=ema_long, adjust=False).mean()
    df["RSI"] = rsi(df["Close"], rsi_len)
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], 14)
    df["ATR%"] = (df["ATR"] / df["Close"]) * 100.0
    df["HH_N"] = df["High"].rolling(lookback_breakout).max()
    last = df.iloc[-1]

    trend_align = int(last["EMA_S"] > last["EMA_M"] > last["EMA_L"])
    breakout = int(last["Close"] >= last["HH_N"])
    rsi_ok = int((last["RSI"] >= rsi_min) & (last["RSI"] <= rsi_max))
    atr_ok = int((last["ATR%"] >= min_atr_pc) & (last["ATR%"] <= max_atr_pc))

    score = (3*trend_align) + (2*breakout) + (2*rsi_ok) + (1*atr_ok)

    liq_ok = True
    avg_dollar_vol = np.nan
    if universe_type == "stocks":
        if "Volume" in df.columns:
            avg_dollar_vol = float((df["Volume"].tail(20) * df["Close"].tail(20)).mean() / 1e6)
            liq_ok = avg_dollar_vol >= min_avg_dollar_vol
        else:
            liq_ok = False

    if liq_ok:
        rows.append({
            "Symbol": s,
            "Score": int(score),
            "TrendAlign": bool(trend_align),
            "Breakout": bool(breakout),
            "RSI": round(float(last["RSI"]), 1),
            "ATR%": round(float(last["ATR%"]), 2),
            "Avg$Vol(M)" : None if np.isnan(avg_dollar_vol) else round(avg_dollar_vol, 1)
        })

rank = pd.DataFrame(rows).sort_values(["Score","ATR%"], ascending=[False, False]).head(top_n)

col1, col2 = st.columns([1,2], gap="large")

with col1:
    st.subheader("Top Picks")
    if rank.empty:
        st.info("No symbols passed the filters today. Loosen filters or change the universe.")
    else:
        st.dataframe(rank, use_container_width=True)
    st.markdown("---")
    st.caption("Scores: 3=trend alignment (EMA S>M>L), 2=breakout over N-day high, 2=RSI in band, 1=ATR% in range.")

    default_symbol = rank["Symbol"].iloc[0] if not rank.empty else (symbol_list[0] if symbol_list else "")
    selected = st.selectbox("Open chart for:", options=symbol_list, index=(symbol_list.index(default_symbol) if default_symbol in symbol_list else 0))

with col2:
    if selected:
        tv_symbol = f"{tv_prefix}:{selected}"
        st.subheader(f"TradingView Chart — {tv_symbol}")
        tradingview_widget(tv_symbol, interval=chart_interval, theme=theme, height=640)

st.markdown("---")
st.caption("Tip: paste your own symbols in the sidebar. For FX use raw pairs (EURUSD). For stocks use Yahoo tickers (AAPL).")

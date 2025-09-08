import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Scanner — Stocks & FX (with TradingView)", layout="wide")

# ===================== TradingView embed =====================
def tradingview_widget(symbol: str, interval: str = "D", theme: str = "light", height: int = 640):
    """
    Embeds a TradingView chart (symbol like 'NASDAQ:AAPL' or 'FX:EURUSD').
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

# Shared signal knobs
st.sidebar.subheader("Signal Logic")
lookback_breakout = st.sidebar.slider("Breakout lookback (days)", 20, 100, 55, 5)
ema_short = st.sidebar.slider("EMA short", 5, 50, 20, 1)
ema_mid   = st.sidebar.slider("EMA mid", 10, 100, 50, 1)
ema_long  = st.sidebar.slider("EMA long", 50, 300, 200, 5)
rsi_len   = st.sidebar.slider("RSI length", 7, 30, 14, 1)
rsi_min   = st.sidebar.slider("RSI lower bound (long bias)", 45, 60, 52, 1)
rsi_max   = st.sidebar.slider("RSI upper bound (avoid overheating)", 65, 85, 70, 1)
min_atr_pc = st.sidebar.slider("Min ATR% (volatility floor)", 0.2, 5.0, 0.6, 0.1)
max_atr_pc = st.sidebar.slider("Max ATR% (volatility cap)", 1.0, 15.0, 6.0, 0.5)

st.sidebar.subheader("Output")
top_n = st.sidebar.slider("Top picks per board", 3, 15, 8, 1)
chart_interval = st.sidebar.selectbox("Chart interval", ["D","240","60","30","W"], index=0)
theme = st.sidebar.selectbox("Chart theme", ["light","dark"], index=0)

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
    """
    Convert Yahoo tickers like 'BRK-B' to TradingView format 'BRK.B' and add prefix e.g. 'NYSE:BRK.B'.
    """
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
        return data, {f"{s}=X": s for s in symbols}
    else:
        data = yf.download(symbols, start=start.date(), end=end.date(), auto_adjust=False, progress=False, group_by='ticker', threads=True)
        return data, {}

def get_df(data, sym, is_fx=False):
    if is_fx:
        key = f"{sym}=X"
        if key not in data: return None
        return data[key].rename(columns=str.title).dropna()
    else:
        if sym not in data: return None
        return data[sym].rename(columns=str.title).dropna()

def score_rows(symbols, data, universe_type: str, min_dollar_vol_m: float = 50.0):
    rows = []
    for s in symbols:
        df = get_df(data, s, is_fx=(universe_type=="fx"))
        if df is None or len(df) < 220:  # need enough history
            continue
        df["EMA_S"] = df["Close"].ewm(span=ema_short, adjust=False).mean()
        df["EMA_M"] = df["Close"].ewm(span=ema_mid, adjust=False).mean()
        df["EMA_L"] = df["Close"].ewm(span=ema_long, adjust=False).mean()
        df["RSI"]   = rsi(df["Close"], rsi_len)
        df["ATR"]   = atr(df["High"], df["Low"], df["Close"], 14)
        df["ATR%"]  = (df["ATR"] / df["Close"]) * 100.0
        df["HH_N"]  = df["High"].rolling(lookback_breakout).max()

        last = df.iloc[-1]
        trend_align = int(last["EMA_S"] > last["EMA_M"] > last["EMA_L"])
        breakout    = int(last["Close"] >= last["HH_N"])
        rsi_ok      = int((last["RSI"] >= rsi_min) & (last["RSI"] <= rsi_max))
        atr_ok      = int((last["ATR%"] >= min_atr_pc) & (last["ATR%"] <= max_atr_pc))
        score       = (3*trend_align) + (2*breakout) + (2*rsi_ok) + (1*atr_ok)

        liq_ok = True
        avg_dollar_vol = np.nan
        if universe_type == "stocks":
            if "Volume" in df.columns:
                avg_dollar_vol = float((df["Volume"].tail(20) * df["Close"].tail(20)).mean() / 1e6)
                liq_ok = avg_dollar_vol >= min_dollar_vol_m
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
    if not rows:
        return pd.DataFrame(columns=["Symbol","Score","TrendAlign","Breakout","RSI","ATR%","Avg$Vol(M)"])
    return pd.DataFrame(rows).sort_values(["Score","ATR%"], ascending=[False, False])

# ===================== Fetch data once per board =====================
with st.spinner("Downloading STOCKS data..."):
    stocks_data, _ = fetch_yf(stocks_universe, is_fx=False)

with st.spinner("Downloading FX data..."):
    fx_data, _ = fetch_yf(fx_universe, is_fx=True)

# ===================== Score / Rank =====================
rank_stocks = score_rows(stocks_universe, stocks_data, "stocks")
rank_fx     = score_rows(fx_universe, fx_data, "fx")

rank_stocks = rank_stocks.head(top_n)
rank_fx     = rank_fx.head(top_n)

# ===================== UI: Two boards =====================
st.title("📈 Daily Scanner — Stocks & Forex (Two Boards + TradingView)")

col_left, col_right = st.columns(2, gap="large")

# ----- Stocks board -----
with col_left:
    st.subheader("Top Stocks")
    if rank_stocks.empty:
        st.info("No stocks passed the filters. Loosen filters or edit the universe.")
        selected_stock = None
    else:
        st.dataframe(rank_stocks, use_container_width=True)
        default_stock = rank_stocks["Symbol"].iloc[0]
        selected_stock = st.selectbox("Stock chart:", options=rank_stocks["Symbol"].tolist(), index=0, key="stock_select")

    # Compact chart inside the column
    if selected_stock:
        tv_sym_stock = tv_stock_symbol(stocks_prefix, selected_stock)
        tradingview_widget(tv_sym_stock, interval=chart_interval, theme=theme, height=520)

    # Optional full-width chart below both columns
    stock_full = st.checkbox("Show stock chart full width", value=False, key="stock_wide")
    if stock_full and selected_stock:
        st.markdown("---")
        st.subheader(f"Full-width — {stocks_prefix}:{selected_stock.replace('-', '.')}")
        tradingview_widget(tv_stock_symbol(stocks_prefix, selected_stock), interval=chart_interval, theme=theme, height=740)

# ----- FX board -----
with col_right:
    st.subheader("Top Forex")
    if rank_fx.empty:
        st.info("No FX pairs passed the filters. Loosen filters or edit the universe.")
        selected_fx = None
    else:
        st.dataframe(rank_fx, use_container_width=True)
        default_fx = rank_fx["Symbol"].iloc[0]
        selected_fx = st.selectbox("FX chart:", options=rank_fx["Symbol"].tolist(), index=0, key="fx_select")

    # Compact chart inside the column
    if selected_fx:
        tv_sym_fx = tv_fx_symbol(fx_prefix, selected_fx)
        tradingview_widget(tv_sym_fx, interval=chart_interval, theme=theme, height=520)

    # Optional full-width chart below both columns
    fx_full = st.checkbox("Show FX chart full width", value=False, key="fx_wide")
    if fx_full and selected_fx:
        st.markdown("---")
        st.subheader(f"Full-width — {fx_prefix}:{selected_fx}")
        tradingview_widget(tv_fx_symbol(fx_prefix, selected_fx), interval=chart_interval, theme=theme, height=740)

st.markdown("---")
st.caption(
    "Scores: 3 = EMA alignment (EMA short > mid > long), 2 = breakout above N-day high, "
    "2 = RSI inside band, 1 = ATR% within range. Stocks include a 20-day average $ volume filter."
)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Scanner — Stocks & FX (Lenient/Strict + Full-Width Charts)", layout="wide")

# ===================== TradingView embed (FULL WIDTH) =====================
def tradingview_widget_full(symbol: str, interval: str = "D", theme: str = "light", height: int = 950):
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

# ===================== Sidebar =====================
st.sidebar.title("Scanner Settings")

# Mode strict/lenient + score mini
st.sidebar.subheader("Mode")
lenient = st.sidebar.toggle("Lenient mode (more setups)", value=True,
                            help="Relaxes breakout by tolerance, widens RSI band, widens ATR range, and accepts softer EMA alignment.")
min_score_to_show = st.sidebar.slider("Minimum score to list", 1, 8, 4, 1)

# Logique de signal
st.sidebar.subheader("Signal Logic")
lookback_breakout = st.sidebar.slider("Breakout lookback (days)", 20, 100, 55, 5)
ema_short = st.sidebar.slider("EMA short", 5, 50, 20, 1)
ema_mid   = st.sidebar.slider("EMA mid", 10, 100, 50, 1)
ema_long  = st.sidebar.slider("EMA long", 50, 300, 200, 5)
rsi_len   = st.sidebar.slider("RSI length", 7, 30, 14, 1)

# Base bands; lenient widens
if lenient:
    rsi_min, rsi_max = 45, 75
    min_atr_pc, max_atr_pc = 0.3, 10.0
    breakout_tol_pc = 0.8   # allow within 0.8% of N-day high/low
    soft_ema = True         # EMA_S > EMA_M and EMA_S > EMA_L (instead of strict S>M>L)
else:
    rsi_min = st.sidebar.slider("RSI min", 45, 60, 52, 1)
    rsi_max = st.sidebar.slider("RSI max", 65, 85, 70, 1)
    min_atr_pc = st.sidebar.slider("Min ATR%", 0.2, 5.0, 0.6, 0.1)
    max_atr_pc = st.sidebar.slider("Max ATR%", 1.0, 15.0, 6.0, 0.5)
    breakout_tol_pc = 0.0
    soft_ema = False

# Entrées / stops / objectifs
st.sidebar.subheader("Entries & Targets")
atr_mult = st.sidebar.slider("Stop ATR multiple", 0.5, 5.0, 2.0, 0.1)
buffer_pc_stocks = st.sidebar.slider("Entry buffer (stocks) %", 0.0, 1.0, 0.20, 0.05)
buffer_pc_fx     = st.sidebar.slider("Entry buffer (FX) %", 0.0, 0.50, 0.05, 0.01)
tp1_R = st.sidebar.slider("TP1 (R multiple)", 0.5, 3.0, 1.5, 0.1)
tp2_R = st.sidebar.slider("TP2 (R multiple)", 1.0, 5.0, 2.0, 0.1)
enable_shorts = st.sidebar.checkbox("Include SHORT setups", value=True)

# Sortie / chart
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
min_dollar_vol_m = st.sidebar.number_input("Min avg $ volume (stocks, M USD)", 0.0, 10000.0, 30.0, 5.0,
                                           help="Lower this if you get no stock setups.")

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
    tv_core = yahoo_sym.replace("-", ".")  # BRK-B -> BRK.B
    return f"{prefix}:{tv_core}"

def tv_fx_symbol(prefix: str, pair: str) -> str:
    return f"{prefix}:{pair}"

@st.cache_data(show_spinner=False)
def fetch_yf(symbols, is_fx=False):
    end = datetime.utcnow()
    start = end - timedelta(days=420)
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

# ===================== Scoring + Levels =====================
def compute_signal_and_levels(df: pd.DataFrame, is_fx: bool):
    d = df.copy()
    d["EMA_S"] = d["Close"].ewm(span=ema_short, adjust=False).mean()
    d["EMA_M"] = d["Close"].ewm(span=ema_mid, adjust=False).mean()
    d["EMA_L"] = d["Close"].ewm(span=ema_long, adjust=False).mean()
    d["RSI"]   = rsi(d["Close"], rsi_len)
    d["ATR"]   = atr(d["High"], d["Low"], d["Close"], 14)
    d["ATR%"]  = (d["ATR"] / d["Close"]) * 100.0
    d["HH_N"]  = d["High"].rolling(lookback_breakout).max()
    d["LL_N"]  = d["Low"].rolling(lookback_breakout).min()
    last = d.iloc[-1]

    # EMA conditions
    if soft_ema:
        ema_up   = bool((last["EMA_S"] > last["EMA_M"]) and (last["EMA_S"] > last["EMA_L"]))
        ema_down = bool((last["EMA_S"] < last["EMA_M"]) and (last["EMA_S"] < last["EMA_L"]))
    else:
        ema_up   = bool(last["EMA_S"] > last["EMA_M"] > last["EMA_L"])
        ema_down = bool(last["EMA_S"] < last["EMA_M"] < last["EMA_L"])

    # Breakout with tolerance
    tol = breakout_tol_pc / 100.0
    bo_up = bool((last["Close"] >= last["HH_N"]) or (abs(last["HH_N"] - last["Close"]) / max(1e-9, last["HH_N"]) <= tol))
    bo_dn = bool((last["Close"] <= last["LL_N"]) or (abs(last["Close"] - last["LL_N"]) / max(1e-9, last["LL_N"]) <= tol))

    # RSI & ATR ranges
    rsi_ok = (last["RSI"] >= rsi_min) and (last["RSI"] <= rsi_max)
    atr_ok = (last["ATR%"] >= min_atr_pc) and (last["ATR%"] <= max_atr_pc)

    # Scores
    score_long  = (3 if ema_up else 0) + (2 if bo_up else 0) + (2 if rsi_ok else 0) + (1 if atr_ok else 0)
    score_short = (3 if ema_down else 0) + (2 if bo_dn else 0) + (2 if rsi_ok else 0) + (1 if atr_ok else 0)

    # Build reasons string
    parts_long = []
    parts_short = []
    parts_long += ["EMA↑" if ema_up else "ema?"]
    parts_short += ["EMA↓" if ema_down else "ema?"]
    parts_long += ["B/O" if bo_up else "bo?"]
    parts_short += ["B/D" if bo_dn else "bd?"]
    parts_long += [f"RSI({round(float(last['RSI']),1)}) ok" if rsi_ok else f"RSI {round(float(last['RSI']),1)}"]
    parts_short += [f"RSI({round(float(last['RSI']),1)}) ok" if rsi_ok else f"RSI {round(float(last['RSI']),1)}"]
    parts_long += [f"ATR%({round(float(last['ATR%']),2)}) ok" if atr_ok else f"ATR% {round(float(last['ATR%']),2)}"]
    parts_short += [f"ATR%({round(float(last['ATR%']),2)}) ok" if atr_ok else f"ATR% {round(float(last['ATR%']),2)}"]
    reasons_long = ", ".join(parts_long)
    reasons_short = ", ".join(parts_short)

    # Entries/Stops/Targets
    buffer_pc = (buffer_pc_fx if is_fx else buffer_pc_stocks) / 100.0
    entry = stop = tp1 = tp2 = np.nan
    bias = "None"
    score = 0
    reasons = ""

    if ema_up and bo_up and rsi_ok and atr_ok:
        bias = "Long"
        score = int(score_long)
        trigger = float(last["High"])
        entry = trigger * (1.0 + buffer_pc)
        stop  = float(last["Close"] - last["ATR"] * atr_mult)
        risk  = max(1e-9, entry - stop)
        tp1   = entry + risk * tp1_R
        tp2   = entry + risk * tp2_R
        reasons = reasons_long

    elif enable_shorts and ema_down and bo_dn and rsi_ok and atr_ok:
        bias = "Short"
        score = int(score_short)
        trigger = float(last["Low"])
        entry = trigger * (1.0 - buffer_pc)
        stop  = float(last["Close"] + last["ATR"] * atr_mult)
        risk  = max(1e-9, stop - entry)
        tp1   = entry - risk * tp1_R
        tp2   = entry - risk * tp2_R
        reasons = reasons_short

    return {
        "Bias": bias,
        "Score": int(score),
        "Entry": None if np.isnan(entry) else round(entry, 6 if is_fx else 4),
        "Stop": None if np.isnan(stop) else round(stop, 6 if is_fx else 4),
        "TP1": None if np.isnan(tp1) else round(tp1, 6 if is_fx else 4),
        "TP2": None if np.isnan(tp2) else round(tp2, 6 if is_fx else 4),
        "RSI": round(float(last["RSI"]), 1),
        "ATR%": round(float(last["ATR%"]), 2),
        "Reasons": reasons
    }

def score_rows_with_levels(symbols, data, universe_type: str):
    rows = []
    is_fx = (universe_type == "fx")
    for s in symbols:
        df = get_df(data, s, is_fx=is_fx)
        if df is None or len(df) < 220:
            continue

        # Liquidity filter for stocks
        if universe_type == "stocks":
            if "Volume" in df.columns:
                avg_dollar_vol = float((df["Volume"].tail(20) * df["Close"].tail(20)).mean() / 1e6)
                if avg_dollar_vol < min_dollar_vol_m:
                    continue

        sig = compute_signal_and_levels(df, is_fx=is_fx)
        if sig["Score"] >= min_score_to_show and sig["Bias"] != "None":
            rows.append({
                "Symbol": s,
                "Bias": sig["Bias"],
                "Score": sig["Score"],
                "Entry": sig["Entry"],
                "Stop": sig["Stop"],
                "TP1": sig["TP1"],
                "TP2": sig["TP2"],
                "RSI": sig["RSI"],
                "ATR%": sig["ATR%"],
                "Reasons": sig["Reasons"]
            })
    if not rows:
        return pd.DataFrame(columns=["Symbol","Bias","Score","Entry","Stop","TP1","TP2","RSI","ATR%","Reasons"])
    return pd.DataFrame(rows).sort_values(["Score","ATR%"], ascending=[False, False])

# ===================== Fetch & Score =====================
with st.spinner("Downloading STOCKS data..."):
    stocks_data = fetch_yf(stocks_universe, is_fx=False)
with st.spinner("Downloading FX data..."):
    fx_data = fetch_yf(fx_universe, is_fx=True)

rank_stocks = score_rows_with_levels(stocks_universe, stocks_data, "stocks").head(top_n)
rank_fx     = score_rows_with_levels(fx_universe, fx_data, "fx").head(top_n)

# ===================== UI =====================
st.title("📊 Daily Scanner — Stocks & Forex (Lenient/Strict, Entries/Stops/Targets)")

# Deux tableaux (gauche = actions, droite = forex)
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("Top Stocks")
    if rank_stocks.empty:
        st.warning("No stock setups with current filters. Try Lenient mode, lower Min score, or lower Min $ volume.")
        selected_stock = None
    else:
        st.dataframe(rank_stocks, use_container_width=True)
        selected_stock = st.selectbox("Choose stock:", rank_stocks["Symbol"].tolist(), index=0, key="stock_sel")

with col_right:
    st.subheader("Top Forex")
    if rank_fx.empty:
        st.warning("No FX setups with current filters. Try Lenient mode or lower Min score.")
        selected_fx = None
    else:
        st.dataframe(rank_fx, use_container_width=True)
        selected_fx = st.selectbox("Choose FX:", rank_fx["Symbol"].tolist(), index=0, key="fx_sel")

st.markdown("---")

# Graphiques TradingView (plein écran)
if selected_stock:
    st.subheader(f"Stock chart — {stocks_prefix}:{selected_stock.replace('-', '.')}")
    tradingview_widget_full(
        symbol=tv_stock_symbol(stocks_prefix, selected_stock),
        interval=chart_interval,
        theme=theme,
        height=chart_height_px
    )

if selected_fx:
    st.subheader(f"FX chart — {fx_prefix}:{selected_fx}")
    tradingview_widget_full(
        symbol=tv_fx_symbol(fx_prefix, selected_fx),
        interval=chart_interval,
        theme=theme,
        height=chart_height_px
    )

st.markdown("---")
st.caption(
    "Score = 3 (EMA trend) + 2 (breakout/breakdown with tolerance if lenient) + 2 (RSI band) + 1 (ATR% in range). "
    "Entries = breakout ± buffer, Stop = ATR × multiple, TP1/TP2 = R multiples."
)

st.subheader("Top Picks")
if rank.empty:
    st.info("No symbols passed the filters today. Loosen filters or change the universe.")
else:
    st.dataframe(rank, use_container_width=True)

st.markdown("---")

# --- Chart full width ---
if not rank.empty:
    default_symbol = rank["Symbol"].iloc[0]
else:
    default_symbol = (symbol_list[0] if symbol_list else "")

selected = st.selectbox("Open chart for:", options=symbol_list,
                        index=(symbol_list.index(default_symbol) if default_symbol in symbol_list else 0))

if selected:
    tv_symbol = f"{tv_prefix}:{selected}"
    st.subheader(f"TradingView Chart — {tv_symbol}")
    tradingview_widget(tv_symbol, interval=chart_interval, theme=theme, height=720)


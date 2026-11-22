"""SignalGap One-Search Stock Outlook — transparent research demonstration.

No API keys, no paid AI calls, no trades. The directional tilt is a simple
uncalibrated heuristic; the old historical test did NOT establish spectral alpha.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from signal_engine import analyze, clean_symbol, get_prices

st.set_page_config(
    page_title="SignalGap | One-Search Stock Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown('''
<style>
  .stApp{background:linear-gradient(140deg,#08121d 0%,#0b1224 55%,#10102a 100%);color:#f1f7ff;}
  .block-container{max-width:1160px;padding-top:2.2rem;padding-bottom:4rem;}
  [data-testid="stAppViewContainer"]{color:#e8f0fb;}
  [data-testid="stHeader"]{background-color:transparent;}
  .eyebrow{color:#6bdedb;letter-spacing:.18em;font-size:12px;font-weight:750;}
  .name{font-weight:850;font-size:clamp(2rem,5vw,3.5rem);letter-spacing:-.045em;line-height:1.1;margin:8px 0;}
  .lead{font-size:1.09rem;line-height:1.7;color:#bac8db;max-width:860px;}
  .hero{padding:30px 32px 24px;border:1px solid #30455f;border-radius:24px;
     background:linear-gradient(112deg,rgba(27,73,86,.46),rgba(53,33,95,.36));margin:0 0 20px;}
  .pill{display:inline-block;margin:9px 7px 0 0;padding:6px 12px;border:1px solid #394f68;
     border-radius:999px;color:#c7d9ea;font-size:.80rem;}
  .feature{padding:22px 24px;border-radius:18px;background:#101e31;border:1px solid #2b3c53;margin:10px 0;}
  .feature h3{font-size:1.18rem;margin:0 0 8px;color:#eef5fc;}
  .feature p{color:#c3d1e3;line-height:1.65;margin:0;}
  .outlook{padding:22px 27px;border-radius:20px;background:linear-gradient(115deg,#152b38,#20213d);
      border:1px solid #42566d;margin-top:18px;margin-bottom:17px;}
  .outlook-big{font-size:2rem;line-height:1.15;font-weight:800;letter-spacing:-.035em;}
  .outlook-sub{color:#c2d1e1;margin-top:9px;line-height:1.55;}
  .fine{color:#aebfd3;font-size:.80rem;line-height:1.6;}
  div[data-testid="stMetric"]{background:#111d30;border:1px solid #2b3a53;border-radius:15px;padding:16px 17px;}
  [data-testid="stMetricLabel"]{color:#bbcbe1!important;}
  [data-testid="stMetricValue"]{color:#f5f9ff!important;}
  label, .stSelectbox label, .stTextInput label{color:#d1deed!important;}
  .stButton > button[kind="primary"],.stFormSubmitButton > button[kind="primary"]{
     background:#4bc4c6;color:#06111b;border:0;font-weight:750;min-height:46px;}
  .stButton > button[kind="primary"]:hover,.stFormSubmitButton > button[kind="primary"]:hover{background:#77e4e1;}
  div[data-testid="stExpander"]{border:1px solid #2a3d55;border-radius:12px;background:#0f1b2d;}
  a{color:#79dfdd!important;}
  @media (max-width:600px){.hero{padding:22px 18px;}.outlook{padding:20px 18px;}}
</style>
''', unsafe_allow_html=True)

st.markdown('''<div class="hero">
<div class="eyebrow">EXPERIMENTAL MARKET INTELLIGENCE</div>
<div class="name">SIGNAL<span style="color:#81e4df">GAP</span></div>
<div class="lead">Where might a stock be heading—and what could the usual price chart be missing?
Enter any supported stock ticker to explore its current market tilt, unusual movement against peers,
and structural market signals.</div>
<span class="pill">One ticker</span><span class="pill">Fresh historical data</span>
<span class="pill">Math + evidence</span><span class="pill">No invented forecasts</span>
</div>''', unsafe_allow_html=True)

# One simple interaction, suitable for a public shareable demo.
if "active_ticker" not in st.session_state:
    initial = st.query_params.get("ticker", "NVDA")
    try:
        st.session_state.active_ticker = clean_symbol(str(initial))
    except ValueError:
        st.session_state.active_ticker = "NVDA"

with st.form("ticker_search"):
    left, middle, right = st.columns([3.2, 1.5, 1.2], vertical_alignment="bottom")
    with left:
        ticker_text = st.text_input("Enter a stock symbol", value=st.session_state.active_ticker,
                                   placeholder="e.g. NVDA, AAPL, MSFT, TSLA", max_chars=15)
    with middle:
        horizon = st.selectbox("Your outlook window", [5, 20],
                               format_func=lambda n: f"Next {n} sessions", index=0)
    with right:
        submitted = st.form_submit_button("Analyze stock", type="primary", use_container_width=True)
    if submitted:
        try:
            st.session_state.active_ticker = clean_symbol(ticker_text)
            st.query_params["ticker"] = st.session_state.active_ticker
        except ValueError as error:
            st.error(str(error))
            st.stop()

@st.cache_data(ttl=1800, show_spinner=False)
def price_snapshot(symbol: str):
    return get_prices(symbol)

active = st.session_state.active_ticker
try:
    with st.spinner(f"Reading latest available closes and peer-market structure for {active}…"):
        price_table = price_snapshot(active)
        result = analyze(price_table, active, horizon)
except Exception as error:
    st.error(f"We couldn't retrieve sufficient price history for {html.escape(active)} right now.")
    st.info("The public data service may be rate-limited or the symbol may be unavailable. Try another ticker or refresh after a while. No outlook will be fabricated when data is missing.")
    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")
    st.stop()

# Topline first. Avoid "buy/sell" or probabilities that have not been calibrated.
color = "#7ce4bd" if result.tilt == "UPWARD TILT" else "#ffb48d" if result.tilt == "DOWNWARD TILT" else "#e1ca85"
st.markdown(f'''<div class="outlook">
<div class="eyebrow">{html.escape(active)} · {horizon}-SESSION EXPERIMENTAL OUTLOOK · DATA THROUGH {result.as_of}</div>
<div class="outlook-big" style="color:{color};margin-top:12px">{result.tilt}</div>
<div class="outlook-sub">{html.escape(result.assessment)}</div>
<div class="fine" style="margin-top:10px">Heuristic assessment, not a statistically validated probability or investment recommendation.</div>
</div>''', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Latest adjusted close", f"${result.adjusted_close:,.2f}", help="Last available daily adjusted close, not a live trading quote.")
m2.metric("20-session strength vs SPY",
          f"{result.recent_vs_spy:+.1%}" if result.recent_vs_spy is not None else "N/A",
          help="Stock's trailing 20-session return minus SPY's trailing 20-session return; descriptive, not forward-looking.")
m3.metric("Peer divergence (5 sessions)",
          f"{result.divergence_z:+.2f}σ" if result.divergence_z is not None else "N/A",
          help="Unusual recent movement relative to an illustrative AI/technology peer basket; not a p-value.")

st.markdown('### What might be easy to miss?')
st.markdown(f'<div class="feature"><h3>◈ Under-the-surface signal</h3><p>{html.escape(result.reason)}</p></div>',unsafe_allow_html=True)

c1, c2 = st.columns([1.2,.8],gap="large")
with c1:
    st.markdown("### Price strength vs the market")
    comparison = pd.concat({active: result.index_close, "SPY": result.benchmark_close}, axis=1).dropna()
    if len(comparison) > 1:
        relative = comparison.div(comparison.iloc[0]).sub(1) * 100
        f = go.Figure()
        f.add_trace(go.Scatter(x=relative.index, y=relative[active], name=active,
                               mode="lines", line=dict(color="#71e5df",width=3)))
        f.add_trace(go.Scatter(x=relative.index, y=relative["SPY"], name="SPY",
                               mode="lines", line=dict(color="#adacf2",width=2)))
        f.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c8d5e6"), margin=dict(l=10,r=10,t=14,b=14),
                        legend=dict(orientation="h",y=1.13),xaxis=dict(showgrid=False),
                        yaxis=dict(title="Historical change (%)",gridcolor="#28364c"))
        st.plotly_chart(f,use_container_width=True)
        st.caption("Historical adjusted closes indexed to 0% at the beginning of the displayed period. This chart is not a forecast.")
    else:
        st.info("Not enough shared price observations to chart stock versus SPY.")
with c2:
    st.markdown("### A key risk")
    st.markdown(f'<div class="feature"><h3>What could reverse this?</h3><p>{html.escape(result.counter_case)}</p></div>',
                unsafe_allow_html=True)
    if result.market_mode_share is not None:
        st.metric("Shared-market movement",f"{result.market_mode_share:.0%}",
                  help="Leading eigenvalue share of the stock/peer return correlation matrix. This is not an expected return.")
        if result.mode_change is not None:
            st.caption(f"Change in shared movement across overlapping historical windows: {result.mode_change:+.1%}.")
    else:
        st.caption("Market-structure reading unavailable for the current ticker and peer sample.")

st.markdown("### How far could the stock move?")
if result.volatility_scale is not None and result.lower_reference is not None:
    s1, s2, s3 = st.columns(3)
    s1.metric("Lower movement reference",f"${max(0,result.lower_reference):,.2f}")
    s2.metric("Latest adjusted close",f"${result.adjusted_close:,.2f}")
    s3.metric("Upper movement reference",f"${result.upper_reference:,.2f}")
    st.caption(f"Illustrative historical-volatility movement scale (±${result.volatility_scale:,.2f} over {horizon} sessions), centered on the latest adjusted close. NOT a predicted price target, confidence interval, or guarantee.")
else:
    st.info("There is not enough recent history to calculate even an illustrative volatility-based movement scale.")

if result.diagnostics:
    with st.expander("Data quality notes"):
        for note in result.diagnostics:
            st.write("• " + note)

with st.expander("What powers the outlook? And how reliable is it?"):
    st.markdown('''**1. Relative strength.** Compare trailing stock returns against SPY, plus the stock's own 50-session trend. This sets the displayed *tilt*; it is not a calibrated forecast.\n\n
**2. Hidden divergence.** Compare the stock's recent returns against a basket of technology/AI-related stocks. Flag unusually large residual moves for investigation.\n\n
**3. Spectral mathematics.** Compute eigenvalues of the peer correlation matrix to gauge whether stocks are becoming more synchronized or independent. This is random-matrix-inspired market-structure research—not the Riemann Hypothesis predicting stocks.\n\n
**4. Independent backtesting.** The earlier five-stock historical study showed that adding spectral features made stock-direction probability scores **worse** than the price-only baseline. Consequently, this public version does *not* turn eigenvalues into bullish probabilities.\n\n
**5. What isn't connected.** Your fine-tuned Inkling, the filing-event extractor, current news, analyst expectations, and investment consensus are **not running** in this public build. Future versions can add those with validated timestamps and secure, rate-limited services.\n\n
**Data source:** Yahoo Finance through the unofficial yfinance library, refreshed on a cache. Market data may be delayed or temporarily unavailable and may be subject to provider usage restrictions.''')
    try:
        summary_path = Path(__file__).parent / "data" / "validation_summary.json"
        if summary_path.exists():
            values=json.loads(summary_path.read_text(encoding="utf-8"))
            h5=values.get("horizons",{}).get("5",{}).get("direction",{})
            h20=values.get("horizons",{}).get("20",{}).get("direction",{})
            if h5 and h20:
                st.caption(f"Prior research, 5-day Brier: price-only {h5['baseline_brier']:.4f}, spectral {h5['spectral_brier']:.4f}; 20-day Brier: price-only {h20['baseline_brier']:.4f}, spectral {h20['spectral_brier']:.4f}. Lower is better. Not a validation of this simplified UI heuristic.")
    except Exception:
        pass

st.markdown("---")
st.caption(f"SIGNALGAP · Experimental independent market research · Last usable daily close: {result.as_of}. Not investment advice. No automatic trading or future-information access.")

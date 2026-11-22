"""SignalGap | Public, read-only market intelligence research showcase.

No model calls, live stock APIs, credentials, trade execution or data uploads.
The historical evaluation and spectral reference series were provided by the
project author. The 6-asset spectral reference is intentionally *not* described
as the 5-stock feature matrix used by the backtest.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from analytics import (read_snapshot, validate_against_summary, model_scorecard,
                       by_asset, calibration_table, confidence_slice, regime_mae)

st.set_page_config(
    page_title='SignalGap | Quantitative AI Research',
    page_icon='◈',
    layout='wide',
    initial_sidebar_state='expanded',
)

P = {'bg':'#08101d','card':'#101d30','line':'#26354b','text':'#ecf5ff',
     'muted':'#a0adc1','cyan':'#47d9db','violet':'#b4a3ff','amber':'#ffce76',
     'coral':'#ff8e91','green':'#76e5b0'}

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0B1424;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F1F5F9;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #F1F5F9;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_data():
    summary,pred,regimes,spectral=read_snapshot()
    validate_against_summary(summary,pred,regimes)
    return summary,pred,regimes,spectral

try:
    summary, predictions, regimes, spectral = get_data()
except Exception as exc:
    st.error('The bundled snapshot could not be verified. Please redeploy with the complete data/ folder.')
    st.exception(exc)
    st.stop()

st.sidebar.markdown('### ◈ SIGNALGAP')
st.sidebar.caption('Independent research · static November 2026 snapshot')
st.sidebar.markdown('---')
horizon = st.sidebar.radio('Forward horizon', [5, 20], horizontal=True, format_func=lambda v:f'{v} days')
all_tickers = sorted(predictions[horizon]['ticker'].unique())
selected = st.sidebar.multiselect('Assets to inspect', all_tickers, default=all_tickers)
st.sidebar.markdown('---')
st.sidebar.markdown('**Validation principles**')
st.sidebar.caption('✓ Point-in-time features  \n✓ Walk-forward evaluation  \n✓ Baseline comparison  \n✓ No invented trading probabilities')
st.sidebar.markdown('---')
st.sidebar.caption('Published build: Research snapshot v1.0')
st.sidebar.caption('This page does not call Inkling, Tinker, or live market APIs.')

st.markdown('''<section class="hero">
<div class="eyebrow">Experimental Market Intelligence · AI + Spectral Mathematics</div>
<div class="brand">SIGNAL<span class="gradient">GAP</span></div>
<div class="sub">Can eigenvalue-based market structure reveal changes that conventional price models miss? A reproducible research lab combining Riemann-inspired random matrix methods, AI event analysis, and honest walk-forward evaluation.</div>
<span class="pill">Historical research</span><span class="pill">5 AI-related equities</span><span class="pill">SPY benchmark</span><span class="pill">Not investment advice</span>
</section>''', unsafe_allow_html=True)

st.markdown('''<div class="callout"><strong>Research finding:</strong> The added spectral indicators did not improve benchmark-relative stock-direction probability forecasts on this sample. The application deliberately shows unsuccessful results, uncertainty, and limitations rather than generating a bullish or bearish stock signal.</div>''',unsafe_allow_html=True)

subset = predictions[horizon][predictions[horizon]['ticker'].isin(selected)].copy()
if subset.empty:
    st.warning('Select at least one stock in the sidebar to view the research results.')
    st.stop()

scores=model_scorecard(subset)
base=scores.iloc[0]; spec=scores.iloc[1]
a,b,c,d = st.columns(4)
a.metric('Walk-forward predictions',f'{len(subset):,}',help='Correlated observations: multiple assets per date and overlapping forecast horizons')
b.metric('Baseline Brier',f'{base.brier:.4f}',help='Lower is better; computed on filtered predictions')
c.metric('Spectral Brier',f'{spec.brier:.4f}',delta=f'{(base.brier-spec.brier):+.4f} improvement',delta_color='normal',help='Positive delta would mean spectral improvement')
d.metric('Decision dates',f'{subset.date.nunique():,}',help='Number of distinct feature dates, not statistically independent observations')

tab1,tab2,tab3,tab4=st.tabs(['◉ Overview','⌁ Backtest Laboratory','◈ Spectral Radar','✧ Research Method'])


def plot_layout(fig, height=365, ytitle='', legend=True):
    fig.update_layout(height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
      font=dict(family='Arial',color='#bdc9d9',size=12),margin=dict(l=12,r=12,t=38,b=12),
      legend=dict(orientation='h',y=1.13,x=0),showlegend=legend,
      xaxis=dict(showgrid=False,zeroline=False),yaxis=dict(gridcolor='#253147',zeroline=False,title=ytitle))
    return fig

with tab1:
    st.markdown('### Do spectral features add predictive value?')
    left,right=st.columns([1.15,.85])
    with left:
        comp=pd.DataFrame({'Model':['Price-only baseline','+ Spectral features'],
                 'Brier score':[base.brier,spec.brier],
                 'Color':[P['cyan'],P['violet']]})
        fig=go.Figure(go.Bar(x=comp['Model'], y=comp['Brier score'], marker_color=comp['Color'],
            text=[f'{n:.4f}' for n in comp['Brier score']],textposition='outside'))
        fig.update_yaxes(range=[0, max(.31,float(comp['Brier score'].max())*1.12)])
        st.plotly_chart(plot_layout(fig,390,'Brier score (lower is better)',False),use_container_width=True)
    with right:
        st.markdown('#### What the numbers say')
        st.markdown(f'''<div class="note">
        <b>{horizon}-day horizon:</b> {spec.brier:.4f} with spectral signals versus {base.brier:.4f} without them.<br><br>
        <b>Direction accuracy:</b> {spec.accuracy:.1%} spectral versus {base.accuracy:.1%} baseline.<br><br>
        <b>Conclusion:</b> There is no demonstrated directional edge in this snapshot. Calibration and a larger, independently held-out universe are needed before making market claims.
        </div>''',unsafe_allow_html=True)
        st.caption('Brier score penalizes inaccurate probability forecasts. All displayed metrics reflect your selection in the sidebar.')
    st.markdown('#### Where the spectral model helps—or hurts')
    by=by_asset(subset)
    fig=go.Figure()
    fig.add_bar(name='Price-only baseline', x=by['Ticker'],y=by['Base Brier'],marker_color=P['cyan'])
    fig.add_bar(name='Spectral enhanced',x=by['Ticker'],y=by['Spectral Brier'],marker_color=P['violet'])
    fig.update_layout(barmode='group')
    st.plotly_chart(plot_layout(fig,365,'Brier score'),use_container_width=True)
    st.caption('Per-asset probabilities are tested out of sample by decision date, but labels overlap across horizons and assets. This is not a transaction-level simulation.')

with tab2:
    st.markdown('### Probability reliability and confidence')
    c1,c2=st.columns([.57,.43])
    with c1:
        calibration_model=st.selectbox('Model for calibration view',['Spectral enhanced','Price-only baseline'])
        model_col='spectral_prob' if calibration_model=='Spectral enhanced' else 'baseline_prob'
        calib=calibration_table(subset, model_col)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color='#8997ac',dash='dash'),name='Perfect calibration'))
        fig.add_trace(go.Scatter(x=calib['mean_prob'],y=calib['observed'],mode='lines+markers',
         line=dict(color=P['violet'] if model_col=='spectral_prob' else P['cyan'],width=3),
         marker=dict(size=10),name='Observed',customdata=calib['n'],
         hovertemplate='Mean predicted: %{x:.1%}<br>Observed: %{y:.1%}<br>Cases: %{customdata}<extra></extra>'))
        fig.update_xaxes(range=[0,1],title='Mean predicted probability')
        fig.update_yaxes(range=[0,1],title='Observed positive-return frequency')
        st.plotly_chart(plot_layout(fig,405),use_container_width=True)
    with c2:
        threshold=st.slider('Minimum absolute direction confidence',.50,.99,.90,.01,
                            help='Confidence = max(p, 1-p). Counts both confident positive and negative forecasts.')
        sl=confidence_slice(subset,model_col,threshold)
        st.metric('High-confidence forecasts',f"{sl['n']}")
        st.metric('Observed directional accuracy', f"{sl['accuracy']:.1%}" if sl['n'] else 'N/A')
        st.metric('Mean forecast confidence',f"{sl['avg_confidence']:.1%}" if sl['n'] else 'N/A')
        st.caption('When forecast confidence exceeds actual accuracy, the model may be overconfident. Small groups are noisy; confidence is not an investment recommendation.')
        st.markdown('''<div class="note"><b>Why this matters:</b> A convincing-looking confidence score means little without empirical calibration. This diagnostic exposes the model’s failure modes instead of hiding them.</div>''',unsafe_allow_html=True)
    st.markdown('### Inspect historical prediction records')
    cols=['date','ticker','entry_date','outcome_end','baseline_prob','spectral_prob','positive_excess','excess_log_return']
    show=subset[cols].sort_values(['date','ticker'],ascending=[False,True]).head(250).copy()
    st.dataframe(show,hide_index=True,use_container_width=True,height=360)
    st.download_button('Export filtered prediction records', data=subset.to_csv(index=False).encode('utf-8'),
                       file_name=f'signalgap_{horizon}d_filtered_research.csv',mime='text/csv',type='secondary')

with tab3:
    st.markdown('### Risk and regime validation')
    st.caption('These measurements address future volatility and dispersion—not stock price direction.')
    rg=regime_mae(regimes[horizon])
    st.dataframe(rg,hide_index=True,use_container_width=True)
    st.markdown('#### Spectral structure through time')
    st.info('The reference series below includes six assets (five stocks plus SPY). It is a separate exploratory basket, not the five-stock spectral dataset used in the prediction backtest. The model validation numbers above come from the five-stock experiment.')
    f=go.Figure()
    f.add_trace(go.Scatter(x=spectral['date'],y=spectral['market_mode_share']*100,
                           mode='lines',line=dict(color=P['cyan'],width=2),name='Common-factor share'))
    f.update_yaxes(title='Variance share (%)',range=[0,100])
    st.plotly_chart(plot_layout(f,390),use_container_width=True)
    left,right=st.columns(2)
    with left:
        latest=spectral.iloc[-1]
        st.metric('Last reference market-mode share',f"{latest['market_mode_share']:.1%}")
    with right:
        st.metric('Last reference eigenvalue / MP edge',f"{latest['lambda1_over_mp']:.2f}×")
    st.caption('Random Matrix Theory helps characterize correlation structure, but it does not prove that eigenvalue shifts anticipate positive stock returns. The idealized Marchenko–Pastur edge is not a validated financial forecasting threshold.')

with tab4:
    st.markdown('### The research architecture')
    st.markdown('''<div class="note"><b>01 · Riemann-inspired spectral mathematics</b><br>Rolling return correlations, eigenvalues and common-factor concentration. This uses random matrix concepts; it does not use the unproven Riemann Hypothesis to predict stocks.<br><br>
    <b>02 · Inkling 2 Mini fine-tuning</b><br>A six-example, one-optimizer-step LoRA smoke test completed. General financial Q&A did not establish a measurable improvement. Its outputs are <b>not included</b> in the directional backtest shown here.<br><br>
    <b>03 · Evidence and expectation-gap prototypes</b><br>Source-grounded event extraction and expectation comparisons are separate experimental workflows. Current backtesting does <b>not</b> use their scores.<br><br>
    <b>04 · Walk-forward validation</b><br>5- and 20-session benchmark-relative return labels, next-session-close entry assumptions, and comparison against a price-only baseline. Outputs are historic research observations, not live forecasts.</div>''',unsafe_allow_html=True)
    st.markdown('### Scope and limitations')
    st.markdown('''- **Snapshot, not a live feed:** the results were frozen as of **November 20, 2026**. No automatic stock, SEC, or social-media updates occur.
- **Five-stock selection:** AMZN, GOOGL, MSFT, NVDA, ORCL; SPY is the external benchmark. This is a small, fixed, potentially survivorship-biased universe.
- **Overlapping observations:** 5- and 20-session outcomes overlap. Counts are predictions, not independent experiments.
- **Historical execution simplification:** adjusted closing prices and next-session-close entry; excludes spreads, taxes, slippage, liquidity and costs.
- **No demonstrated alpha:** probability quality worsened when spectral features were added, despite some tiny 20-day risk-metric gains.
- **Not a stock-trading product:** no live signals, price targets, position sizing, automatic trades, or investment advice.''')
    st.markdown('### Reproducibility')
    st.caption('The repository contains the exact historical prediction and risk results used in this display plus validation_summary.json. No employer information, Tinker key, model weights, or licensed raw stock prices are included.')
    with st.expander('Show summarized source snapshot'):
        st.json(summary, expanded=False)

st.markdown('---')
st.caption('SIGNALGAP · Independent AI and quantitative research prototype · Research data frozen 2026-11-20 · Historical evaluation, not investment advice.')

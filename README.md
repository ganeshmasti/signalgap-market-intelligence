# SignalGap — One-Search Stock Intelligence (public MVP)

A **simplified public Streamlit demonstration**. Visitors enter a ticker such as `NVDA`, choose 5 or 20 trading sessions, and receive:

- A plain-language **upward / downward / mixed** experimental *tilt* based on observed relative strength and trend.
- An **unusual peer-divergence** check based on recent stock-versus-peer residual returns.
- A **market correlation structure** measure based on the leading eigenvalue of a rolling return correlation matrix.
- A stock-versus-SPY historical performance chart.
- An illustrative **volatility movement scale** (not a forecast interval or price target).
- Clear counterarguments and source limitations.

The app intentionally **does not** produce calibrated return probabilities, price targets, automatic trades, or unsupported claims of alpha. Earlier SignalGap testing of spectral features *did not* establish an advantage over baseline price-only models. The current directional tilt is **not** the trained model from those tests.

## Deploy over existing Streamlit GitHub repository

Replace the **root** `app.py` and `requirements.txt` and add `signal_engine.py` beside them. Leave existing `data/` and `analytics.py` alone. GitHub/Streamlit will redeploy from `app.py` on the `main` branch. No new folders need to be uploaded.

Required file tree:

```
app.py
signal_engine.py
requirements.txt
# other files from earlier SignalGap version may remain
```

No Tinker key, payment method, or local Ubuntu process is required in the deployed app. This version does not run Inkling; it runs price-based heuristics and spectral diagnostics locally in the Streamlit process.

## Local check

```bash
uv add streamlit plotly pandas numpy yfinance
uv run python signal_engine.py
uv run streamlit run app.py
```

The Python self-test is offline. The Streamlit app needs network access to retrieve adjusted daily closes via yfinance and may be rate-limited by Yahoo. Cache TTL is 30 minutes. Data may be delayed, affected by corporate actions, or unavailable. Usage rights must be verified for commercial deployment.

## What the UI actually means

- **UPWARD TILT / DOWNWARD TILT / MIXED:** uncalibrated rule, not a prediction or estimated probability. Signals are relative 20-/60-day price momentum vs SPY and deviation from the 50-session moving average.
- **Peer divergence:** sum of the last five daily stock-minus-equal-weight-peer returns divided by their historical standard deviation scaled by sqrt(5). |z| >= 1.5 is a deliberately exploratory threshold, not a statistical discovery or unpriced news.
- **Market-mode share:** leading eigenvalue / number of securities of a rolling return correlation matrix. Regime change compares two highly overlapping 90-day windows separated by 10 days. Its magnitude does not imply price direction.
- **Movement reference:** last adjusted close +/- last 60-session daily volatility times sqrt(horizon). This is not a calibrated probability interval.
- **Recent data:** adjusted daily closes, NOT a tick-by-tick real-time quote.

If a newly listed ticker has fewer than 65 shared sessions with SPY, the app displays **INSUFFICIENT HISTORY**, not a fabricated directional signal.

## Future build (not implemented here)

A stronger **forward-looking research** product would connect timestamped filings, consensus-revision datasets, independently evaluated Inkling extraction, and paper-tracked prospective forecasts. This public release cannot claim to identify material information before the market has priced it in.

Independent research demonstration. Not investment advice.

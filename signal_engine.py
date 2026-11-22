"""SignalGap public demonstration engine.

Uses recent *observed* adjusted-price behavior, peer divergence, and rolling
correlation spectra. It is NOT an empirically validated return predictor.
No Inkling/Tinker calls and no financial-article sentiment analysis occur here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# A fixed, illustrative AI-and-technology basket. A point-in-time investable
# universe would need historical membership and corporate-action controls.
PEERS = (
    "NVDA", "MSFT", "GOOGL", "AMZN", "ORCL", "META", "AMD", "AVGO",
    "CRM", "ADBE", "MU", "AMAT", "NOW", "TSM", "INTC",
)
BENCHMARK = "SPY"


@dataclass
class SignalResult:
    ticker: str
    as_of: str
    adjusted_close: float
    horizon: int
    tilt: str
    assessment: str
    reason: str
    counter_case: str
    recent_vs_spy: float | None
    medium_vs_spy: float | None
    trend_vs_ma50: float | None
    divergence_z: float | None
    market_mode_share: float | None
    mode_change: float | None
    peers_used: int
    volatility_scale: float | None
    lower_reference: float | None
    upper_reference: float | None
    index_close: pd.Series
    benchmark_close: pd.Series
    diagnostics: list[str]
    methodology_notes: list[str]


def clean_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", symbol):
        raise ValueError("Enter a stock ticker such as NVDA, MSFT, AAPL or TSLA.")
    return symbol


def _safe_last_return(price: pd.Series, sessions: int) -> float | None:
    price = price.dropna()
    if len(price) <= sessions:
        return None
    beginning = float(price.iloc[-sessions - 1])
    ending = float(price.iloc[-1])
    if beginning <= 0:
        return None
    return ending / beginning - 1.0


def normalize_yfinance_closes(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Accept yfinance download output with field-first or ticker-first columns."""
    if frame is None or frame.empty:
        raise ValueError("The price provider returned no data.")
    out = pd.DataFrame(index=pd.to_datetime(frame.index).tz_localize(None)
                       if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None
                       else pd.to_datetime(frame.index))
    if isinstance(frame.columns, pd.MultiIndex):
        names0 = list(frame.columns.get_level_values(0))
        names1 = list(frame.columns.get_level_values(1))
        if "Close" in names0:
            closed = frame.xs("Close", axis=1, level=0)
        elif "Close" in names1:
            closed = frame.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Downloaded prices do not contain Close data.")
        for sym in symbols:
            if sym in closed.columns:
                out[sym] = pd.to_numeric(closed[sym], errors="coerce").to_numpy()
    else:
        if "Close" in frame.columns and len(symbols) == 1:
            out[symbols[0]] = pd.to_numeric(frame["Close"], errors="coerce").to_numpy()
        else:
            raise ValueError("Unexpected provider format; expected adjusted Close fields.")
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    out[out <= 0] = np.nan
    return out


def get_prices(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Network boundary: only invoked by the Streamlit app after user submit."""
    import yfinance as yf

    symbol = clean_symbol(ticker)
    symbols = list(dict.fromkeys([symbol, BENCHMARK, *PEERS]))
    raw = yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        multi_level_index=True,
        threads=True,
        progress=False,
        timeout=15,
    )
    return normalize_yfinance_closes(raw, symbols)


def _market_mode(returns: pd.DataFrame) -> tuple[float, float] | None:
    """Return market-mode concentration and idealized MP ratio for a window."""
    sample = returns.dropna(how="any")
    if len(sample) < 45 or sample.shape[1] < 4:
        return None
    # Correlation may not be well-defined for near-constant columns.
    nonconstant = sample.loc[:, sample.std(ddof=1) > 1e-8]
    n = nonconstant.shape[1]
    if n < 4:
        return None
    eig = np.linalg.eigvalsh(nonconstant.corr().to_numpy(dtype=float))
    leading = float(np.max(eig))
    noise_edge = (1.0 + np.sqrt(n / len(nonconstant))) ** 2
    return leading / n, leading / noise_edge


def _spectral_features(prices: pd.DataFrame, ticker: str) -> tuple[float | None, float | None, int, str]:
    # SPY excluded. Include the target when it has sufficient history.
    candidates = [sym for sym in [ticker, *PEERS] if sym != BENCHMARK and sym in prices]
    candidates = list(dict.fromkeys(candidates))
    history = prices[candidates].tail(120).copy()
    # Requiring sufficient common, recent observations reduces accidental
    # forward-fill of newly listed assets or stale securities.
    eligible = [c for c in history if history[c].notna().sum() >= 108 and
                bool(history[c].tail(5).notna().all())]
    if len(eligible) < 5:
        return None, None, len(eligible), "Too few stocks with sufficient recent data for spectral analysis."
    history = history[eligible].dropna(how="any")
    returns = history.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 105:
        return None, None, len(eligible), "Not enough shared trading sessions for robust rolling spectra."
    now = _market_mode(returns.tail(90))
    prev = _market_mode(returns.iloc[-100:-10])
    if now is None or prev is None:
        return None, None, len(eligible), "Correlation spectrum unavailable for this stock basket."
    return now[0], now[0] - prev[0], len(eligible), ""


def analyze(prices: pd.DataFrame, ticker: str, horizon: int = 5) -> SignalResult:
    """Deterministic *hypothesis generator*, not a calibrated stock-return model."""
    symbol = clean_symbol(ticker)
    if horizon not in (5, 20):
        raise ValueError("Supported horizons are 5 and 20 trading sessions.")
    if symbol not in prices or prices[symbol].dropna().empty:
        raise ValueError(f"No usable price history found for {symbol}.")
    if BENCHMARK not in prices or prices[BENCHMARK].dropna().empty:
        raise ValueError("Benchmark SPY was not available from the data provider.")

    # Alignment avoids comparing ticker and benchmark on different dates.
    aligned = prices[[symbol, BENCHMARK]].dropna().sort_index()
    if len(aligned) < 65:
        # Show the observable price history for a newly-listed security, but
        # refuse to manufacture a directional forecast from insufficient data.
        latest = aligned[symbol]
        last = float(latest.iloc[-1])
        as_of_short = pd.Timestamp(aligned.index[-1]).strftime("%Y-%m-%d")
        return SignalResult(
            ticker=symbol, as_of=as_of_short, adjusted_close=last,
            horizon=horizon, tilt="INSUFFICIENT HISTORY",
            assessment=(f"Only {len(aligned)} shared trading sessions are available. "
                        "A reliable 50-day trend and 60-day relative-strength comparison cannot yet be calculated."),
            reason="A newly listed stock has not established a long enough observable price history for spectral divergence analysis.",
            counter_case="IPO-related price changes and limited liquidity can overwhelm short historical patterns.",
            recent_vs_spy=None, medium_vs_spy=None, trend_vs_ma50=None,
            divergence_z=None, market_mode_share=None, mode_change=None,
            peers_used=0, volatility_scale=None, lower_reference=None,
            upper_reference=None, index_close=latest.tail(126),
            benchmark_close=aligned[BENCHMARK].tail(126),
            diagnostics=["Insufficient price history: only observed market data is displayed."],
            methodology_notes=["No directional price signal is issued without sufficient historical observations.",
                               "No live news or Tinker model inference is included in this public build."],
        )
    stock, spy = aligned[symbol], aligned[BENCHMARK]
    close = float(stock.iloc[-1])
    as_of = pd.Timestamp(aligned.index[-1]).strftime("%Y-%m-%d")
    diagnostics: list[str] = []
    notes: list[str] = []

    rel20_stock, rel20_spy = _safe_last_return(stock, 20), _safe_last_return(spy, 20)
    rel60_stock, rel60_spy = _safe_last_return(stock, 60), _safe_last_return(spy, 60)
    excess20 = (rel20_stock - rel20_spy) if None not in (rel20_stock, rel20_spy) else None
    excess60 = (rel60_stock - rel60_spy) if None not in (rel60_stock, rel60_spy) else None
    ma50 = float(stock.tail(50).mean())
    deviation = (close / ma50 - 1.0) if ma50 > 0 else None

    # Direction is a *simple transparent heuristic* based on trailing trends,
    # never a calibrated stock-return probability. Spectral features are not
    # included in the directional score: prior tests found no added alpha.
    votes: list[int] = []
    if excess20 is not None:
        votes.append(1 if excess20 > 0.02 else -1 if excess20 < -0.02 else 0)
    if excess60 is not None:
        votes.append(1 if excess60 > 0.035 else -1 if excess60 < -0.035 else 0)
    if deviation is not None:
        votes.append(1 if deviation > 0.01 else -1 if deviation < -0.01 else 0)
    vote = sum(votes)
    if vote >= 2:
        tilt = "UPWARD TILT"
        assessment = "Recent relative performance and trend are constructive. This is a momentum hypothesis, not a price forecast."
    elif vote <= -2:
        tilt = "DOWNWARD TILT"
        assessment = "Recent relative performance and trend are weak. This is a momentum hypothesis, not a price forecast."
    else:
        tilt = "MIXED / UNCLEAR"
        assessment = "The evidence does not support a clear directional tilt."

    # Target vs equal-weight peer basket over shared sessions.
    peer_cols = [x for x in PEERS if x != symbol and x in prices and prices[x].tail(90).notna().sum() >= 80]
    divergence_z: float | None = None
    peer_mean = None
    if len(peer_cols) >= 4:
        common = prices[[symbol, *peer_cols]].dropna().tail(125)
        if len(common) >= 65:
            ret = common.pct_change(fill_method=None).dropna()
            peer_ret = ret[peer_cols].mean(axis=1)
            residual = ret[symbol] - peer_ret
            if len(residual) >= 45:
                vol = float(residual.std(ddof=1))
                if vol > 1e-9:
                    # Last 5-day residual versus normal 5-day residual movement.
                    divergence_z = float(residual.tail(5).sum() / (vol * np.sqrt(5)))
                    peer_mean = float(peer_ret.tail(5).sum())

    share, share_delta, n_peers, spectral_issue = _spectral_features(prices, symbol)
    if spectral_issue:
        diagnostics.append(spectral_issue)

    # The "unusual" label is explicitly defined by an arbitrary exploratory
    # threshold. It is not a p-value nor a claim of nonpublic information.
    if divergence_z is not None and abs(divergence_z) >= 1.5:
        direction = "outperforming" if divergence_z > 0 else "underperforming"
        reason = (f"Unusual peer divergence: {symbol} has been {direction} its peer basket over the "
                  f"last 5 sessions (standardized divergence {divergence_z:+.2f}). "
                  "This merits investigation but does not show the change is unpriced.")
    elif share_delta is not None and abs(share_delta) >= .06:
        direction = "stronger" if share_delta > 0 else "weaker"
        reason = (f"Correlation structure is shifting: common movement has become {direction} "
                  f"(change of {share_delta:+.1%} in market-mode share over the comparison windows). "
                  "The direction of the next stock return is not implied.")
    elif divergence_z is not None:
        reason = "No unusually large five-session divergence from the peer basket was detected."
    else:
        reason = "Insufficient comparable peer history to assess unusual relative movement."

    if share is not None and share >= .68:
        counter = "Strong shared movement may dominate stock-specific developments; a company story may not translate into price outperformance."
    elif divergence_z is not None and abs(divergence_z) >= 1.5:
        counter = "Peer divergence can quickly reverse. Check verified company disclosures before treating it as a new fundamental signal."
    else:
        counter = "A new filing, earnings surprise, market-wide selloff, or reversal in recent momentum could invalidate this tilt."

    daily = stock.pct_change(fill_method=None).dropna().tail(60)
    dvol = float(daily.std(ddof=1)) if len(daily) >= 25 else None
    movement = float(close * dvol * np.sqrt(horizon)) if dvol is not None else None

    # Stale-data rule is not tied to local market timezone; weekends are okay.
    try:
        age_days = (datetime.now(timezone.utc).date() - pd.Timestamp(aligned.index[-1]).date()).days
        if age_days > 7:
            diagnostics.append(f"Most recent usable close is {as_of}; the data may be stale.")
    except Exception:
        pass

    notes.extend([
        "Observed price history only; no live news, financial consensus, or Inkling calls.",
        "Outlook is an uncalibrated heuristic based on relative momentum and trend.",
        "Spectral change is a descriptive market-structure diagnostic, not a directional predictor.",
        "The earlier five-stock walk-forward study did not demonstrate spectral alpha.",
        "The displayed movement scale is historical volatility × sqrt(horizon), not a confidence or prediction interval.",
    ])
    return SignalResult(
        ticker=symbol, as_of=as_of, adjusted_close=close, horizon=horizon,
        tilt=tilt, assessment=assessment, reason=reason, counter_case=counter,
        recent_vs_spy=excess20, medium_vs_spy=excess60, trend_vs_ma50=deviation,
        divergence_z=divergence_z, market_mode_share=share,
        mode_change=share_delta, peers_used=n_peers,
        volatility_scale=movement,
        lower_reference=close-movement if movement is not None else None,
        upper_reference=close+movement if movement is not None else None,
        index_close=stock.tail(126), benchmark_close=spy.tail(126),
        diagnostics=diagnostics, methodology_notes=notes,
    )


def self_test() -> None:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2025-01-01", periods=250)
    market = rng.normal(0.0006, 0.009, len(idx))
    data = {}
    for i, name in enumerate(["NVDA", "MSFT", "GOOGL", "AMZN", "ORCL", "META", "AMD", "SPY"]):
        noise = rng.normal(0, 0.005 + i/20000.0, len(idx))
        returns = market + noise
        data[name] = 100 * np.exp(np.cumsum(returns))
    prices = pd.DataFrame(data, index=idx)
    result = analyze(prices, "NVDA", 20)
    assert result.as_of == "2025-12-16"
    assert result.tilt in ("UPWARD TILT", "DOWNWARD TILT", "MIXED / UNCLEAR")
    assert 0 <= (result.market_mode_share or 0) <= 1
    assert result.volatility_scale is not None and result.volatility_scale >= 0
    assert len(result.index_close) <= 126
    assert clean_symbol(" msft ") == "MSFT"
    try:
        clean_symbol("../../etc")
        raise AssertionError("Invalid symbol unexpectedly accepted")
    except ValueError:
        pass
    # Two common yfinance MultiIndex layouts.
    closes = pd.DataFrame({("Close", "NVDA"): [100, 110], ("Close", "SPY"): [200, 190]},
                          index=pd.bdate_range("2025-01-01", periods=2))
    normalized = normalize_yfinance_closes(closes, ["NVDA", "SPY"])
    assert list(normalized.columns) == ["NVDA", "SPY"]
    assert float(normalized.iloc[-1, 0]) == 110
    reversed_cols = closes.copy()
    reversed_cols.columns = pd.MultiIndex.from_tuples([(b, a) for a, b in closes.columns])
    normalized2 = normalize_yfinance_closes(reversed_cols, ["NVDA", "SPY"])
    assert float(normalized2.iloc[-1, 0]) == 110
    print("PASS: validation, spectral calculations, momentum tilt, provider formats, and synthetic fixture")
    print(f"Example: {result.ticker} | {result.tilt} | correlation share {result.market_mode_share:.1%}")
    print("No API requests or model calls executed.")


if __name__ == "__main__":
    self_test()

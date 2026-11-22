"""Pure, testable calculations for the published SignalGap research snapshot.

Only historical, user-supplied aggregate and prediction data are read. There are
no live price requests, model invocations, or trading signals in this module.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def read_snapshot():
    summary = json.loads((DATA / 'validation_summary.json').read_text(encoding='utf-8'))
    predictions = {}
    regimes = {}
    for h in (5, 20):
        preds = pd.read_csv(DATA / f'direction_predictions_{h}d.csv', parse_dates=['date', 'entry_date', 'outcome_end'])
        reg = pd.read_csv(DATA / f'regime_predictions_{h}d.csv', parse_dates=['date','entry_date','outcome_end'])
        required = {'date','ticker','positive_excess','baseline_prob','spectral_prob','entry_date','outcome_end'}
        if not required.issubset(preds.columns):
            raise ValueError(f'Missing prediction columns: {required-set(preds.columns)}')
        for name in ('baseline_prob', 'spectral_prob'):
            if not preds[name].between(0, 1).all():
                raise ValueError(f'Probability outside 0–1 for {name}')
        if not preds.positive_excess.isin([0,1]).all():
            raise ValueError('Invalid targets')
        if not (preds['date'] < preds['entry_date']).all() or not (preds['entry_date'] < preds['outcome_end']).all():
            raise ValueError('Invalid event order')
        predictions[h] = preds
        regimes[h] = reg
    spectral = pd.read_csv(DATA / 'spectral_features_six_asset_reference.csv', parse_dates=['date'])
    return summary, predictions, regimes, spectral


def brier(df, column):
    if df.empty:
        return float('nan')
    return float(((df[column] - df['positive_excess'])**2).mean())


def accuracy(df, column):
    if df.empty:
        return float('nan')
    return float(((df[column] >= 0.5).astype(int) == df['positive_excess']).mean())


def model_scorecard(df):
    out=[]
    for name, col in (('Baseline', 'baseline_prob'),('Spectral', 'spectral_prob')):
        out.append({
            'model':name,
            'brier':brier(df,col),
            'accuracy':accuracy(df,col),
            'mean_confidence':float(np.maximum(df[col],1-df[col]).mean()) if len(df) else float('nan'),
            'count':len(df),
        })
    return pd.DataFrame(out)


def by_asset(df):
    rows=[]
    for ticker,g in df.groupby('ticker'):
        rows.append({
            'Ticker':ticker,
            'Predictions':len(g),
            'Base Brier':brier(g,'baseline_prob'),
            'Spectral Brier':brier(g,'spectral_prob'),
            'Delta (spectral−base)':brier(g,'spectral_prob')-brier(g,'baseline_prob'),
        })
    return pd.DataFrame(rows).sort_values('Ticker')


def confidence_slice(df, model='spectral_prob', threshold=.9):
    if df.empty:
        return {'n':0,'accuracy':float('nan'),'avg_confidence':float('nan')}
    conf = np.maximum(df[model], 1-df[model])
    d = df.loc[conf>=threshold]
    if d.empty:
        return {'n':0,'accuracy':float('nan'),'avg_confidence':float('nan')}
    return {
        'n':len(d),
        'accuracy':accuracy(d, model),
        'avg_confidence':float(conf.loc[d.index].mean()),
    }


def calibration_table(df, model='spectral_prob', bins=8):
    if df.empty:
        return pd.DataFrame(columns=['band','mean_prob','observed','n'])
    data = df.copy()
    data['bucket'] = pd.cut(data[model], bins=np.linspace(0,1,bins+1), include_lowest=True)
    grouped=data.groupby('bucket', observed=True).agg(mean_prob=(model,'mean'),observed=('positive_excess','mean'),n=('positive_excess','size')).reset_index()
    grouped['band']=grouped['bucket'].astype(str)
    return grouped[['band','mean_prob','observed','n']]


def regime_mae(df):
    result=[]
    for task,g in df.groupby('task'):
        base=np.abs(g['actual']-g['baseline_prediction']).mean()
        spec=np.abs(g['actual']-g['spectral_prediction']).mean()
        result.append({'Target': 'SPY realized volatility' if task=='future_spy_vol' else 'Cross-stock dispersion',
                       'Baseline MAE': float(base), 'Spectral MAE':float(spec),
                       'Change %':float((base-spec)/base*100) if base else float('nan'), 'Windows':len(g)})
    return pd.DataFrame(result)


def validate_against_summary(summary,predictions,regimes,atol=1e-10):
    for horizon in (5,20):
        expected=summary['horizons'][str(horizon)]['direction']
        df=predictions[horizon]
        if len(df) != expected['n_predictions']:
            raise AssertionError(f'{horizon}d row-count mismatch')
        if abs(brier(df,'baseline_prob')-expected['baseline_brier'])>atol:
            raise AssertionError('Baseline score mismatch')
        if abs(brier(df,'spectral_prob')-expected['spectral_brier'])>atol:
            raise AssertionError('Spectral score mismatch')
        if len(df.date.unique())!=expected['n_decision_dates']:
            raise AssertionError('Decision-date mismatch')
        for row in regime_mae(regimes[horizon]).to_dict('records'):
            key='future_spy_vol' if row['Target']=='SPY realized volatility' else 'future_dispersion'
            expected_reg=summary['horizons'][str(horizon)]['regime'][key]
            assert abs(row['Baseline MAE']-expected_reg['baseline_mae'])<atol
            assert abs(row['Spectral MAE']-expected_reg['spectral_mae'])<atol
    return True

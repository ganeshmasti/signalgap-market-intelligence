import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analytics import read_snapshot, validate_against_summary, by_asset, confidence_slice


def test_snapshot():
    summary, pred, regimes, spectral = read_snapshot()
    assert validate_against_summary(summary,pred,regimes)
    assert len(pred[5])==890 and len(pred[20])==860
    assert set(by_asset(pred[5])['Ticker']) == {'AMZN','GOOGL','MSFT','NVDA','ORCL'}
    assert len(spectral)==228
    assert confidence_slice(pred[20])['n']>0

if __name__=='__main__':
    test_snapshot()
    print('PASS: 5d and 20d scores match uploaded validation summary; source files consistent')

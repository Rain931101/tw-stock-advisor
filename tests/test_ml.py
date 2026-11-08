"""機器學習：還原權值、三欄式標記的交易規則防禦、Purged walk-forward。"""
import numpy as np
import pandas as pd

from app.ml import features as MF
from app.ml.labels import COST, triple_barrier
from app.ml.model import purged_walk_forward


def _frame(closes, opens=None, highs=None, lows=None, atr=1.0):
    n = len(closes)
    idx = pd.bdate_range("2026-01-05", periods=n)
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else c, float)
    h = np.array(highs if highs is not None else np.maximum(o, c), float)
    l = np.array(lows if lows is not None else np.minimum(o, c), float)
    f = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
    for k in ("open", "high", "low", "close"):
        f[f"adj_{k}"] = f[k]
    f["atr14"] = atr
    f["ex_day"] = False
    f["limit_up_close"] = False
    return f


def test_adjust_prices_removes_dividend_gap():
    idx = pd.bdate_range("2026-01-05", periods=4)
    df = pd.DataFrame({"open": [100, 100, 95, 95], "high": [100, 100, 95, 95], "low": [100, 100, 95, 95],
                       "close": [100, 100, 95, 95]}, index=idx, dtype=float)
    divs = pd.DataFrame({"date": [idx[2].date().isoformat()], "prev_close": [100.0], "ref_price": [95.0]})
    a = MF.adjust_prices(df, divs)
    assert np.allclose(a["adj_close"], [95, 95, 95, 95])   # 除息 5 元不再是 −5% 的跌幅
    assert a["ex_day"].tolist() == [False, False, True, False]


def test_take_profit_and_stop_loss():
    # t=0 訊號，t=1 開盤 100 進場；ATR=1 → 停利 102、停損 98.5
    f = _frame([100, 100, 101, 103, 103, 103], opens=[100, 100, 100, 101, 103, 103], highs=[100, 100, 101, 103, 103, 103])
    r = triple_barrier(f, up=2, down=1.5, horizon=4)
    assert r["y"].iloc[0] == 1 and r["barrier"].iloc[0] == "up"
    assert abs(r["ret"].iloc[0] - (102 / 100 - 1 - COST)) < 1e-9
    f = _frame([100, 100, 99, 98, 98, 98], lows=[100, 100, 99, 98, 98, 98])
    r = triple_barrier(f, up=2, down=1.5, horizon=4)
    assert r["y"].iloc[0] == 0 and r["barrier"].iloc[0] == "down"


def test_gap_down_fills_at_open_not_stop():
    # 第 3 天跳空開在 95（停損 98.5）→ 以 95 成交，不是理想的 98.5
    f = _frame([100, 100, 100, 95, 95, 95], opens=[100, 100, 100, 95, 95, 95])
    r = triple_barrier(f, up=2, down=1.5, horizon=4)
    assert abs(r["ret"].iloc[0] - (95 / 100 - 1 - COST)) < 1e-9


def test_limit_up_signal_and_locked_entry_not_filled():
    f = _frame([100, 110, 121, 121, 121, 121, 121], opens=[100, 110, 121, 121, 121, 121, 121],
               lows=[100, 110, 121, 121, 121, 121, 121])
    f["limit_up_close"] = [False, True, True, False, False, False, False]
    r = triple_barrier(f, up=2, down=1.5, horizon=3)
    assert r["no_fill"].iloc[0]                 # 進場日（第 2 天）一字漲停：買不到
    assert r["no_fill"].iloc[1]                 # 訊號日收盤鎖漲停：不計
    assert np.isnan(r["y"].iloc[0]) and np.isnan(r["y"].iloc[1])


def test_locked_limit_down_defers_exit():
    # 進場 100、停損 98.5；第 3 天一字跌停 90（賣不掉）→ 第 4 天開盤 85 才賣出
    c = [100, 100, 100, 90, 85, 85, 85]
    f = _frame(c, opens=[100, 100, 100, 90, 85, 85, 85], highs=[100, 100, 100, 90, 86, 86, 86],
               lows=[100, 100, 100, 90, 84, 84, 84])
    r = triple_barrier(f, up=2, down=1.5, horizon=4)
    assert r["y"].iloc[0] == 0
    assert abs(r["ret"].iloc[0] - (85 / 100 - 1 - COST)) < 1e-9


def test_purged_walk_forward_no_overlap():
    dates = pd.bdate_range("2026-01-01", periods=200).to_numpy()
    folds = purged_walk_forward(dates, 5, purge=11)
    assert len(folds) == 5
    u = np.sort(np.unique(dates))
    for tr, te in folds:
        assert tr.max() < te.min()
        gap = np.searchsorted(u, te.min()) - np.searchsorted(u, tr.max())
        assert gap > 11                          # 訓練最後一天距測試第一天超過 horizon+1 個交易日

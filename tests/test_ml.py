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


def test_split_detected_and_adjusted():
    idx = pd.bdate_range("2026-01-05", periods=4)
    # 1 拆 4：546 → 開盤 136.5、收 143
    df = pd.DataFrame({"open": [540, 546, 136.5, 143], "high": [550, 546, 145, 145], "low": [540, 540, 136, 140],
                       "close": [546, 546, 143, 143]}, index=idx, dtype=float)
    a = MF.adjust_prices(df, pd.DataFrame(columns=["date", "prev_close", "ref_price"]))
    assert abs(a["adj_close"].iloc[1] - 136.5) < 1e-6      # 分割前價格依比例縮小
    assert a["ex_day"].tolist() == [False, False, True, False]


def test_exit_policy_breakeven_and_trailing():
    from app.ml import exits as XT
    # 進場 100、ATR 2：初始停損 max(97, 95)=97；漲到 104（+2 ATR... 這裡 be_atr=1 → 102）後保本 100.5；
    # 最高 110 → 移動停利 110 − 2×2 = 106；之後跌到 105 → 以 106 出場
    c = [100, 100, 103, 108, 110, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105]
    n = len(c)
    o = np.array([100.0] + c[:-1]); o[1] = 100
    h = np.maximum(o, c).astype(float); l = np.minimum(o, c).astype(float); c = np.array(c, float)
    atr = np.full(n, 2.0)
    p = XT.Policy(sl_atr=1.5, sl_cap=0.05, be_atr=1.0, trail_atr=2.0, ma5_mode="none", horizon=10)
    r = XT.simulate(o, h, l, c, c, atr, c, h, l, np.zeros(n, bool), np.array([0]), p)
    assert r["reason"][0] == "stop" and abs(r["ret"][0] - (106 / 100 - 1 - XT.COST)) < 1e-9


def test_exit_policy_loss_capped():
    from app.ml import exits as XT
    # ATR 很大（10）：1.5 ATR = 15%，但 sl_cap 5% → 停損 95，小賠出場
    c = np.array([100, 100, 99, 94, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], float)
    o = np.r_[100, c[:-1]]; h = np.maximum(o, c); l = np.minimum(o, c)
    p = XT.Policy(sl_atr=1.5, sl_cap=0.05, be_atr=None, trail_atr=None, horizon=5)
    r = XT.simulate(o, h, l, c, c, np.full(len(c), 10.0), c, h, l, np.zeros(len(c), bool), np.array([0]), p)
    assert abs(r["ret"][0] - (95 / 100 - 1 - XT.COST)) < 1e-9

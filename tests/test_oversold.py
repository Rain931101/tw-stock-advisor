import numpy as np
import pandas as pd

from app.services import oversold as O


def test_stage_waits_for_share_to_fall():
    share = pd.Series([0.05, 0.20, 0.40, 0.30, 0.35, 0.10], index=pd.date_range("2026-01-01", periods=6))
    assert list(O.stage(share)) == ["none", "watch", "watch", "entry", "watch", "none"]


def _market(n_stocks=60, days=60, crash_day=40, crash_frac=0.5, rebound=0.10):
    idx = pd.bdate_range("2026-01-01", periods=days)
    closes = pd.DataFrame(100.0, index=idx, columns=[f"S{i}" for i in range(n_stocks)])
    hit = closes.columns[: int(n_stocks * crash_frac)]
    closes.loc[idx[crash_day]:, hit] = 80.0                         # 一半股票同一天急跌 20%
    closes.loc[idx[crash_day + 2]:, hit] = 80.0 * (1 + rebound)     # 兩天後反彈
    opens = closes.shift(1).fillna(100.0)
    return closes, opens


def test_breadth_counts_share_below_ma20():
    closes, _ = _market()
    bias, share = O.breadth(closes)
    day = closes.index[40]
    assert share.loc[day] == 0.5
    assert bias.loc[day, "S0"] < O.BIAS and bias.loc[day, "S59"] == 0


def test_backtest_separates_first_day_and_entry():
    closes, opens = _market()
    # 第二天超跌比例下降：讓一部分股票先反彈
    closes.loc[closes.index[41]:, ["S0", "S1", "S2", "S3", "S4"]] = 95.0
    out = O.backtest(closes, opens)
    assert out["first"]["days"] == 1 and out["first"]["n"] == 30
    assert out["entry"]["days"] >= 1
    assert out["episodes"][0]["start"] == str(closes.index[40].date())
    assert np.isfinite(out["entry"]["avg"])

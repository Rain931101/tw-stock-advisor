"""明日大盤預測：準確率計分與區間（不連網）。"""
import numpy as np
import pandas as pd

from app.services import forecast as FC


def test_score_compares_with_always_up_baseline():
    idx = pd.bdate_range("2026-01-01", periods=10)
    oos = pd.DataFrame({"y": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], "prob": [0.7] * 6 + [0.3] * 4}, index=idx)
    s = FC._score(oos)
    assert s["accuracy"] == 1.0 and s["baseline"] == 0.6 and s["up_when_high"] == 1.0 and s["up_when_low"] == 0.0


def test_band_follows_night_move():
    rng = np.random.default_rng(1)
    n = 400
    night = rng.normal(0, 1, n)
    nxt = 0.9 * night + rng.normal(0, 0.3, n)
    close, atr = 20000.0, 200.0
    d = pd.DataFrame({"night_ret": night, "next_ret": nxt, "next_open": 0.8 * night, "close": close, "atr": atr,
                      "next_hi_atr": (nxt / 100 * close) / atr + 0.5, "next_lo_atr": (nxt / 100 * close) / atr - 0.5},
                     index=pd.bdate_range("2024-01-01", periods=n))
    b = FC._band(d, ["night_ret"])
    assert 0.8 < b["w"][0] < 1.0                       # 明天漲跌大致跟著夜盤
    assert abs(b["w_open"][0] - 0.8) < 0.01 and b["open_dir_acc"] > 0.95
    assert abs(b["hi50"] - 0.5) < 0.2 and abs(b["lo50"] + 0.5) < 0.2


def test_gapfade_intraday_check(monkeypatch):
    from datetime import datetime
    from app.services import gapfade as G
    monkeypatch.setattr(G, "stats", lambda: {"intraday": {"checks": {"09:15": {"fade_below": 0.73, "flat_below": 0.39,
                                                                               "fade_above": 0.22, "flat_above": 0.1, "n_below": 44}}}})
    monkeypatch.setattr(G, "premarket_today", lambda: None)
    q = {"open": 20200.0, "prev_close": 20000.0, "price": 20150.0}          # 開高 1%，現在跌破開盤價
    r = G.intraday_check(q, datetime(2026, 1, 5, 9, 20))
    assert r and r["gap"] == 1.0 and r["check"] == "09:15" and r["stats"]["fade_below"] == 0.73
    assert G.intraday_check({**q, "price": 20250.0}, datetime(2026, 1, 5, 9, 20)) is None   # 沒跌破開盤價
    assert G.intraday_check({**q, "open": 20050.0, "price": 20040.0}, datetime(2026, 1, 5, 9, 20)) is None  # 開高不到 0.5%
    assert G.intraday_check(q, datetime(2026, 1, 5, 11, 0)) is None          # 只在 9:05～10:05 判斷


def test_us_alignment_uses_sessions_before_taiwan_open():
    """台股日 D 開盤前已知的美股 = 日期 < D 的最近一個美股收盤；今晚美股 = 下一個台股日開盤前的漲跌。"""
    us = pd.DataFrame({"SOX": [100.0, 110.0, 99.0, 120.0]},
                      index=pd.to_datetime(["2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"]))
    tw = pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"])
    at = FC._us_before(us, pd.DatetimeIndex(tw))
    # 11/06 開盤前 → 美股 11/05；11/09（週一）→ 美股 11/06（週五）；11/10 → 11/09
    assert at["SOX"].tolist() == [100.0, 110.0, 99.0, 120.0]
    nxt = (at["SOX"].shift(-1) / at["SOX"] - 1) * 100       # 台股 11/09 收盤後到 11/10 開盤前 = 美股 11/09 那一晚
    assert round(nxt.iloc[1], 2) == -10.0

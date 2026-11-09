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
    d = pd.DataFrame({"night_ret": night, "next_ret": nxt, "close": close, "atr": atr,
                      "next_hi_atr": (nxt / 100 * close) / atr + 0.5, "next_lo_atr": (nxt / 100 * close) / atr - 0.5},
                     index=pd.bdate_range("2024-01-01", periods=n))
    b = FC._band(d, "night_ret")
    assert 0.8 < b["slope"] < 1.0                      # 明天漲跌大致跟著夜盤
    assert abs(b["hi50"] - 0.5) < 0.2 and abs(b["lo50"] + 0.5) < 0.2

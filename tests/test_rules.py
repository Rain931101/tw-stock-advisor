from datetime import datetime, time

import numpy as np
import pandas as pd
import pytest

from app import db
from app.rules import fundamental, indicators as ind, risk, technical


def make_df(closes, vols=None, start="2025-01-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    vols = np.full(len(closes), 2000.0) if vols is None else np.asarray(vols, dtype=float)
    return pd.DataFrame({
        "open": closes * 0.995, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": vols, "value": closes * vols * 1000,
    }, index=idx)


# ---------- 風控：講義的華通例子（100 萬、單筆最大虧損 1 萬、每筆 10 萬 → 10%） ----------
def test_risk_max_entry_and_sizing():
    s = risk.size_position(entry=240, stop=223.5, per_trade_capital=100_000, max_loss=10_000)
    assert s.ok
    assert s.max_entry == pytest.approx(223.5 / 0.9, rel=1e-3)
    assert s.loss_at_stop <= 10_000
    bad = risk.size_position(entry=260, stop=223.5, per_trade_capital=100_000, max_loss=10_000)
    assert not bad.ok and "不進場" in bad.reason


def test_round_tick():
    assert risk.round_tick(123.27) == 123.5
    assert risk.round_tick(48.33) == 48.35
    assert risk.round_tick(1012) == 1010


# ---------- 量比 ----------
def test_volume_ratio_estimates_full_day():
    # 10:00 累積 3900 張，約為全日 39% → 估計 10000 張；昨量 5000 → 量比 2
    assert ind.volume_ratio(3900, 5000, time(10, 0)) == pytest.approx(2.0, rel=0.01)


# ---------- 角度 ----------
def test_angle_regimes():
    flat = make_df(100 + np.sin(np.arange(60)) * 0.5)
    trend = make_df(100 * 1.006 ** np.arange(60))
    rocket = make_df(100 * 1.03 ** np.arange(60))
    assert ind.angle_regime(ind.trend_angle(flat)) == "震盪盤"
    assert ind.angle_regime(ind.trend_angle(trend)) == "趨勢盤"
    assert ind.angle_regime(ind.trend_angle(rocket)) == "急拉盤"


# ---------- 波段目標價 ----------
def test_wave_target_1to1():
    rally = np.linspace(100, 150, 30)
    box = 140 + np.sin(np.arange(40) / 2) * 5  # 135~145 震盪
    df = make_df(np.concatenate([np.full(30, 100), rally, box]))
    w = ind.wave_target(df)
    assert w is not None
    height = w.resistance - w.rally_start
    assert w.target == pytest.approx(w.box_bottom + height)
    assert w.healthy_pullback == pytest.approx(w.target - (w.resistance - w.box_bottom))


# ---------- 格局 ----------
def test_structure_up_and_down():
    up = make_df(100 + np.arange(120) * 0.5 + np.sin(np.arange(120) / 3) * 4)
    down = make_df(200 - np.arange(120) * 0.5 + np.sin(np.arange(120) / 3) * 4)
    assert ind.structure(up) in ("uptrend", "turning_up")
    assert ind.structure(down) in ("downtrend", "turning_down")


# ---------- 型態：量增突破 ----------
def test_breakout_triggers_after_long_consolidation():
    base = 100 + np.sin(np.arange(260) / 4) * 3  # 一年整理（97~103）
    closes = np.concatenate([base, [108]])
    vols = np.concatenate([np.full(255, 1500), [2200, 2300, 2500, 2600, 2800], [6000]])
    df = make_df(closes, vols)
    res = technical.evaluate(df)
    sig = next(s for s in res["signals"] if s.setup == "breakout")
    assert sig.triggered, sig.reasons
    assert sig.stop < 108 and sig.targets[0]["price"] > 108


def test_no_entry_in_first_five_minutes():
    base = 100 + np.sin(np.arange(260) / 4) * 3
    closes = np.concatenate([base, [108]])
    vols = np.concatenate([np.full(255, 1500), [2200, 2300, 2500, 2600, 2800], [6000]])
    res = technical.evaluate(make_df(closes, vols), now=datetime(2026, 11, 6, 9, 3), intraday=True)
    assert not any(s.triggered for s in res["signals"])


# ---------- 型態：季線站回 ----------
def test_ma60_reclaim():
    up = 100 * 1.004 ** np.arange(300)
    dip = up[-1] * np.array([0.97, 0.95, 0.93, 0.92, 0.93])
    reclaim = [up[-1] * 1.0]
    df = make_df(np.concatenate([up, dip, reclaim]))
    res = technical.evaluate(df)
    sig = next(s for s in res["signals"] if s.setup == "ma60_reclaim")
    assert any("季線" in r for r in sig.reasons)
    assert sig.stop < df["close"].iloc[-1]
    assert any(r["type"] == "ma_below" and r["action"] == "sell_half" for r in sig.exit_rules)


# ---------- 基本面 ----------
def _q(dates, **series):
    rows = []
    for k, vals in series.items():
        for d, v in zip(dates, vals):
            rows.append({"date": d, "type": k, "value": v})
    return rows


QD = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def test_fundamental_good_company():
    income = _q(QD, Revenue=[100, 110, 120, 130, 140, 150, 165, 180], GrossProfit=[40, 45, 50, 55, 60, 66, 73, 80],
                IncomeAfterTaxes=[10, 11, 12, 13, 15, 17, 19, 21], EPS=[1, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.1])
    balance = _q(QD, CurrentAssets=[300] * 8, CurrentLiabilities=[100] * 8, Inventories=[30] * 8,
                 EquityAttributableToOwnersOfParent=[200] * 8, TotalAssets=[400] * 8, Liabilities=[200] * 8,
                 Equity=[200] * 8, AccountsReceivableNet=[50] * 8)
    # 現金流為年初累計數
    cash = _q(QD, CashFlowsFromOperatingActivities=[12, 25, 39, 55, 16, 34, 54, 76],
              CashProvidedByInvestingActivities=[-5, -10, -15, -20, -6, -12, -18, -24],
              CashFlowsProvidedFromFinancingActivities=[-2, -4, -6, -8, -2, -4, -6, -8])
    revenue = [{"revenue_year": y, "revenue_month": m, "revenue": (120 if y == 2025 else 100) * 1e6}
               for y in (2024, 2025) for m in range(1, 13)]
    rep = fundamental.analyze(income, balance, cash, revenue)
    assert rep["pass"], rep["checks"]
    assert rep["grade"] in ("A", "B")
    assert any(c["name"] == "財務穩定的好公司型態" for c in rep["checks"])


def test_fundamental_dangerous_cashflow_is_fatal():
    income = _q(QD, Revenue=[100] * 8, GrossProfit=[20] * 8, IncomeAfterTaxes=[-1] * 8, EPS=[-0.1] * 8)
    cash = _q(QD, CashFlowsFromOperatingActivities=[-1, -2, -3, -4, -1, -2, -3, -4],
              CashProvidedByInvestingActivities=[5, 10, 15, 20, 5, 10, 15, 20],
              CashFlowsProvidedFromFinancingActivities=[5, 10, 15, 20, 5, 10, 15, 20])
    rep = fundamental.analyze(income, [], cash, [])
    assert not rep["pass"] and rep["fatal"]


# ---------- 持股出場規則 ----------
def test_exit_rules_stop_and_ma5():
    from app.services import portfolio

    db.reset_for_tests()
    closes = 100 * 1.003 ** np.arange(100)
    df = make_df(np.concatenate([closes, [closes[-1] * 0.9]]))
    pos = {"id": 1, "buy_price": closes[-1], "buy_date": "2025-05-01", "remaining": 1000, "plan": {
        "exit_rules": [
            {"type": "price_below", "level": closes[-1] * 0.93, "action": "sell_all", "label": "停損"},
            {"type": "close_below_ma", "ma": 5, "action": "sell_all", "label": "破五日線", "intraday_after": "12:30"},
        ]}}
    hits = portfolio.evaluate_exits(pos, df, None, intraday=False)
    labels = {h["label"] for h in hits}
    assert "停損" in labels and "破五日線" in labels
    # 盤中 11:00 跌破五日線還不出場（12:30 後才判斷）
    hits_am = portfolio.evaluate_exits(pos, df, datetime(2026, 11, 6, 11, 0), intraday=True)
    assert not any(h["type"] == "close_below_ma" for h in hits_am)


# ---------- 停損由系統管理：只上不下 ----------
def test_trail_stop_breakeven_at_2r():
    from app.services import portfolio

    closes = np.concatenate([np.full(30, 100.0), np.linspace(100, 125, 10)])
    df = make_df(closes)
    pos = {"buy_price": 100.0, "setup": "ma60_reclaim",
           "plan": {"stop": 90.0, "initial_stop": 90.0, "state": {}, "exit_rules": []}}
    new, reason = portfolio.trail_stop(pos, df)
    assert new == 100.0 and "保本" in reason
    # 已經在成本之上就不會再往下調
    pos["plan"]["stop"] = 110.0
    assert portfolio.trail_stop(pos, df) is None


def test_confirm_buy_keeps_system_stop_and_warns_over_limit():
    from app.services import portfolio

    db.reset_for_tests()
    db.execute("INSERT INTO stocks VALUES ('9999','測試','twse','半導體業')")
    plan = {"stop": 90.0, "stop_reason": "關鍵低點", "setup_name": "季線站回", "exit_rules": []}
    db.execute("INSERT INTO recommendations(created_at,trade_date,stock_id,name,setup,score,status,plan) VALUES (?,?,?,?,?,?,?,?)",
               ("x", "2026-11-06", "9999", "測試", "ma60_reclaim", 80, "active", db.dumps(plan)))
    r = portfolio.confirm_buy(1, "9999", price=100.0, shares=2000)
    assert r["stop"] == 90.0  # 成交價不影響停損
    assert r["warnings"] and "超過單筆可承受虧損" in r["warnings"][0]  # 2000 股 × 10 元 = 2 萬 > 1 萬


# ---------- 融資（槓桿） ----------
def test_margin_call_price_and_sizing():
    # 上市 6 成：追繳價 = 1.3 × 0.6 × 買進價 = 78%
    assert risk.margin_call_price(100, 0.6) == pytest.approx(78.0)
    m = risk.size_margin(100, 94, per_trade_capital=100_000, max_loss=10_000, ratio=0.6, annual_rate=0.065)
    assert m.ok and m.shares == 1000 and m.leverage == 2.5
    assert m.loss_at_stop <= 10_000  # 單筆最大虧損不因槓桿放大
    # 停損在追繳價之下 → 不適合融資
    bad = risk.size_margin(100, 70, 100_000, 10_000, 0.6, 0.065)
    assert not bad.ok and "追繳" in bad.reason


def test_margin_position_interest_and_maintenance():
    from app.services import portfolio

    db.reset_for_tests()
    db.execute("INSERT INTO stocks VALUES ('9998','測試融資','twse','半導體業')")
    closes = np.full(80, 100.0)
    db.upsert_prices([("9998", d.date().isoformat(), c, c, c, c, 2000, 0) for d, c in zip(pd.bdate_range("2026-01-01", periods=80), closes)])
    r = portfolio.confirm_buy(None, "9998", price=100.0, shares=1000, buy_date="2026-01-01", mode="margin")
    assert r["mode"] == "margin"
    h = portfolio.health_check()
    m = h["positions"][0]["margin"]
    assert m["loan"] == 60_000 and m["maintenance_ratio"] == pytest.approx(166.7, abs=0.1)
    assert m["interest"] > 0 and h["summary"]["margin_loan"] == 60_000
    with pytest.raises(ValueError):
        portfolio.confirm_buy(None, "9998", price=100.0, shares=500, mode="margin")  # 融資不能零股


def test_max_loss_above_position_size_does_not_block_everything():
    s = risk.size_position(entry=163.5, stop=154, per_trade_capital=5_000, max_loss=10_000)
    assert s.ok and s.max_entry > 154 and s.shares > 0

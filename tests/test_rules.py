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
    # 上升趨勢中急跌破季線、隔日強勢站回（舊測試的跌幅其實沒有跌破季線，型態根本不成立）
    up = 100 * 1.004 ** np.arange(300)
    dip = up[-1] * np.array([0.95, 0.92, 0.89, 0.87, 0.86])
    reclaim = [up[-1] * 0.93]
    df = make_df(np.concatenate([up, dip, reclaim]))
    res = technical.evaluate(df)
    sig = next(s for s in res["signals"] if s.setup == "ma60_reclaim")
    assert sig.triggered, (sig.reasons, sig.warnings)
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


# ---------- 設定存進資料庫 ----------
def test_settings_persist_any_value_and_reject_invalid(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    db.reset_for_tests()
    monkeypatch.setattr(main.scheduler, "start", lambda: None)
    client = TestClient(main.app)
    r = client.post("/api/settings", json={"capital": 123456, "fee_discount": 0.28, "max_loss_per_trade": 1234})
    assert r.status_code == 200
    got = client.get("/api/settings").json()  # 模擬重新整理
    assert got["capital"] == 123456 and got["fee_discount"] == 0.28 and got["max_loss_per_trade"] == 1234
    bad = client.post("/api/settings", json={"fee_discount": 3, "capital": 999})
    assert bad.status_code == 400 and "手續費折數" in bad.json()["detail"]
    assert client.get("/api/settings").json()["capital"] == 123456  # 有錯就整批不寫


def test_reset_settings_and_undo(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    db.reset_for_tests()
    monkeypatch.setattr(main.scheduler, "start", lambda: None)
    client = TestClient(main.app)
    client.post("/api/settings", json={"capital": 50000, "fee_discount": 0.28})
    r = client.post("/api/settings/reset").json()
    assert r["capital"] == 1_000_000 and r["fee_discount"] == 0.3 and r["can_undo"]
    back = client.post("/api/settings/undo-reset").json()
    assert back["capital"] == 50000 and back["fee_discount"] == 0.28
    assert client.post("/api/settings/undo-reset").status_code == 404  # 只能復原一次


def test_reset_portfolio_and_undo(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import main
    from app.config import get_settings

    db.reset_for_tests()
    monkeypatch.setattr(get_settings(), "db_path", tmp_path / "advisor.db")  # 備份寫到暫存資料夾
    monkeypatch.setattr(main.scheduler, "start", lambda: None)
    db.execute("INSERT INTO stocks VALUES ('9997','測試','twse','半導體業')")
    client = TestClient(main.app)
    client.post("/api/positions", json={"stock_id": "9997", "price": 100, "shares": 1000})
    client.post("/api/positions/1/sell", json={"price": 105, "shares": 500})
    r = client.post("/api/portfolio/reset").json()
    assert r["cleared"] == 1 and r["trades"] == 2 and r["can_undo"]
    assert db.query("SELECT * FROM positions") == [] and db.query("SELECT * FROM trades") == []
    assert (tmp_path / "backups").exists()
    back = client.post("/api/portfolio/undo-reset").json()
    assert back == {"restored": 1, "trades": 2}
    assert db.one("SELECT remaining FROM positions WHERE id=1")["remaining"] == 500


def test_finmind_bucket_reserves_quota_for_interactive():
    import threading as th
    import time as tm

    from app.data import finmind

    b = finmind._Bucket(20)
    with finmind.background():
        for _ in range(17):  # 85% 額度
            b.take()
    done = th.Event()

    def bg():
        with finmind.background():
            b.take()
        done.set()

    th.Thread(target=bg, daemon=True).start()
    tm.sleep(0.3)
    assert not done.is_set()  # 背景已用完自己的份額，必須等待
    t0 = tm.time()
    b.take()  # 互動查詢仍可立即進行
    assert tm.time() - t0 < 0.5


# ---------- 成本、損益平衡點（對齊元大帳上） ----------
def test_unrealized_matches_yuanta_0050():
    from app.rules import costs

    # 元大：0050 350 股、成本 32,274（成本均價 92.21 含手續費）、現價 92.5 → 未實現 +23、0.07%
    u = costs.unrealized("0050", 32274, 92.5, 350)
    assert u == {"market_value": 32375, "sell_fee": 46, "sell_tax": 32, "pnl": 23, "pnl_pct": 0.07}
    be = costs.break_even("0050", 32274, 350)
    assert be["tick_price"] == 92.45
    assert costs.sell_proceeds("0050", 92.45, 350)["net"] >= 32274 > costs.sell_proceeds("0050", 92.40, 350)["net"]


def test_costs_tax_and_min_fee():
    from app.rules import costs

    assert costs.tax_rate("2330") == 0.003 and costs.tax_rate("0050") == 0.001
    assert costs.fee(1000, 0.3, shares=1000) == 20  # 整股最低 20 元
    assert costs.fee(1000, 0.3, shares=10) == 1  # 零股最低 1 元
    assert costs.buy_cost(100, 1000, 1.0) == 100_142


def test_untriggered_setup_has_no_stop():
    flat = make_df(100 + np.sin(np.arange(300) / 5) * 2)
    for sig in technical.evaluate(flat)["signals"]:
        if not sig.triggered:
            assert sig.stop is None and sig.targets == [] and sig.exit_rules == []


# ---------- 個股買進與否 ----------
def test_stock_verdict_avoids_bad_fundamentals_and_waits_without_setup():
    from app.services import recommender

    db.reset_for_tests()
    flat = make_df(100 + np.sin(np.arange(300) / 5) * 2)
    tech = technical.evaluate(flat)
    stock = {"stock_id": "9996", "name": "測試", "market": "twse", "industry": "半導體業"}
    bad = {"pass": False, "grade": "D", "score": 40, "checks": [{"name": "近四季沒有在賺錢", "detail": "虧損", "status": "bad"}]}
    v = recommender.stock_verdict(stock, tech, bad, None, 100.0, "中性", [])
    assert v["decision"] == "avoid" and "基本面" in v["summary"]
    good = {"pass": True, "grade": "A", "score": 90, "checks": [], "metrics": {}}
    v = recommender.stock_verdict(stock, tech, good, None, 100.0, "中性", [])
    assert v["decision"] in ("wait", "avoid")
    if v["decision"] == "wait" and v["closest"]:
        assert v["closest"]["missing"]  # 告訴使用者還差哪些條件


# ---------- 歷史勝率 ----------
def test_label_trades_target_stop_and_expiry():
    from app.services import winrate as W

    up = make_df(np.linspace(100, 130, 40))  # 一路漲 → 先到停利
    lab = W.label_trades(up, hold=20, target=0.08, stop=0.05)
    assert lab["y"].iloc[0] == 1 and lab["ret"].iloc[0] > 0.07
    down = make_df(np.linspace(100, 70, 40))  # 一路跌 → 先到停損
    lab = W.label_trades(down, hold=20, target=0.08, stop=0.05)
    assert lab["y"].iloc[0] == 0 and lab["ret"].iloc[0] < -0.05
    assert lab["y"].iloc[-1] != lab["y"].iloc[-1]  # 最後幾天未來資料不足 → NaN


def test_mine_rules_requires_both_periods_and_assess():
    from app.rules import features as F
    from app.services import winrate as W

    rng = np.random.default_rng(0)
    n = 6000
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    data = pd.DataFrame({k: rng.random(n) < 0.3 for k in F.FEATURES})
    data["date"], data["sid"] = dates, rng.integers(0, 300, n).astype(str)
    good = data["high_60"] & data["vol_up"]
    data["y"] = np.where(good, rng.random(n) < 0.9, rng.random(n) < 0.3).astype(float)
    data["ret"] = np.where(data["y"] == 1, 0.08, -0.05)  # 期望：好組合 ≈ +6.7%、其他 ≈ −1.1%
    p = {"hold_days": 20, "target": 0.08, "stop": 0.05, "buy_exp": 1.0, "min_rate": 0.5, "avoid_exp": 0.0}
    m = W.mine_rules(data, p)
    buy = [r for r in m["rules"] if r["reliable_buy"]]
    assert buy and {"high_60", "vol_up"} <= set(buy[0]["conds"])  # 期望報酬最高的規則一定包含真正有效的條件
    cond = {k: False for k in F.FEATURES} | {"high_60": True, "vol_up": True}
    assert W.assess(cond, m)["decision"] == "buy"
    assert W.assess({k: False for k in F.FEATURES}, m)["decision"] == "avoid"  # 期望報酬為負 → 不推薦



def test_low_winrate_but_positive_expectancy_is_not_avoided():
    from app.rules import features as F
    from app.services import winrate as W

    rng = np.random.default_rng(1)
    n = 6000
    data = pd.DataFrame({k: rng.random(n) < 0.3 for k in F.FEATURES})
    data["date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    data["sid"] = rng.integers(0, 300, n).astype(str)
    data["y"] = (rng.random(n) < 0.45).astype(float)  # 勝率 45%
    data["ret"] = np.where(data["y"] == 1, 0.08, -0.05)  # 期望 ≈ +0.85% > 0
    m = W.mine_rules(data, {"hold_days": 20, "target": 0.08, "stop": 0.05, "buy_exp": 1.0, "min_rate": 0.5, "avoid_exp": 0.0})
    assert W.assess({k: False for k in F.FEATURES}, m)["decision"] == "neutral"  # 勝率不到五成但有賺 → 不會被擋


def test_winrate_all_csv_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main
    from app.services import winrate as W

    monkeypatch.setattr(main.scheduler, "start", lambda: None)
    fake = {"trained": True, "date": "2026-11-06", "rows": [
        {"stock_id": "5371", "name": "中光電", "industry": "光電業", "close": 100, "change_pct": 1.2, "exp": 2.43,
         "rate": 0.6, "decision": "buy", "n_rules": 3, "rule": "均線靠攏", "rule_n": 122, "avg_volume": 5000, "liquid": True}]}
    monkeypatch.setattr(W, "all_stocks", lambda: fake)
    r = TestClient(main.app).get("/api/winrate/all.csv")
    assert r.status_code == 200 and "中光電" in r.text and "推薦" in r.text and "2.43" in r.text


def test_synthetic_bar_estimates_full_day_volume():
    """盤中累計量依經過時間推估全日量（10:00 約只過了 1/4.5），量比不重複推估。"""
    from datetime import datetime
    import pandas as pd
    from app.rules import indicators as ind, technical
    idx = pd.bdate_range("2026-01-05", periods=3)
    df = pd.DataFrame({"open": [10, 10, 10], "high": [10, 10, 10], "low": [10, 10, 10], "close": [10, 10, 10],
                       "volume": [1000, 1000, 1000], "value": [0, 0, 0]}, index=idx, dtype=float)
    now = datetime(2026, 1, 8, 10, 0)
    out = technical.synthetic_bar(df, {"price": 10.5, "open": 10, "high": 10.6, "low": 10, "volume": 300}, "2026-01-08", now=now)
    assert abs(out["volume"].iloc[-1] - ind.estimated_volume(300, now)) < 1e-9
    assert out["volume"].iloc[-1] > 300 and out["vol_est"].iloc[-1] == 1.0

"""個股基本面與籌碼資料（含快取）。"""
from datetime import date, timedelta

from app import db
from app.data import finmind, twse
from app.rules import fundamental


def fundamental_report(stock_id: str, max_age_hours: float = 24 * 7) -> dict:
    key = f"fundamental:{stock_id}"
    hit = db.cache_get(key, max_age_hours)
    if hit is not None:
        return hit
    start_fin = (date.today() - timedelta(days=365 * 3)).isoformat()
    start_rev = (date.today() - timedelta(days=500)).isoformat()
    income = finmind.fetch("TaiwanStockFinancialStatements", stock_id, start_fin)
    balance = finmind.fetch("TaiwanStockBalanceSheet", stock_id, start_fin)
    cash = finmind.fetch("TaiwanStockCashFlowsStatement", stock_id, start_fin)
    revenue = finmind.fetch("TaiwanStockMonthRevenue", stock_id, start_rev)
    val = valuation_all().get(stock_id)
    per = [val] if val else finmind.fetch("TaiwanStockPER", stock_id, (date.today() - timedelta(days=10)).isoformat())
    rep = fundamental.analyze(income, balance, cash, revenue, per)
    rep = db.jsonable(rep)
    rep["updated"] = db.now()
    db.cache_set(key, rep)
    return rep


def valuation_all() -> dict[str, dict]:
    """全市場估值（每日快取），省下每檔一次 FinMind 查詢。"""
    hit = db.cache_get("valuation_all", 12)
    if hit:
        return hit
    data = twse.fetch_valuation_all()
    if data:
        db.cache_set("valuation_all", data)
    return data or (db.cache_get("valuation_all") or {})


def chips(stock_id: str, days: int = 20, max_age_hours: float = 10) -> dict:
    """三大法人近 N 日買賣超（張）與融資變化。講義：主要要判斷控制價格的法人或主力是誰。"""
    key = f"chips:{stock_id}:{days}"
    hit = db.cache_get(key, max_age_hours)
    if hit is not None:
        return hit
    start = (date.today() - timedelta(days=days * 2)).isoformat()
    inst = finmind.fetch("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start)
    margin = finmind.fetch("TaiwanStockMarginPurchaseShortSale", stock_id, start)
    by_day: dict[str, dict] = {}
    for r in inst:
        name = r["name"]
        group = "foreign" if name.startswith("Foreign") else "trust" if name == "Investment_Trust" else "dealer"
        d = by_day.setdefault(r["date"], {"foreign": 0, "trust": 0, "dealer": 0})
        d[group] += (r["buy"] - r["sell"]) / 1000
    dates = sorted(by_day)[-days:]
    series = [{"date": d, **{k: round(v) for k, v in by_day[d].items()}} for d in dates]
    tot = lambda k, n: round(sum(x[k] for x in series[-n:]))  # noqa: E731
    summary = {k: {"5d": tot(k, 5), "20d": tot(k, 20)} for k in ("foreign", "trust", "dealer")}
    # 主導法人
    lead = max(("foreign", "trust"), key=lambda k: abs(summary[k]["5d"]))
    notes = []
    f5, t5 = summary["foreign"]["5d"], summary["trust"]["5d"]
    if f5 > 0 and t5 > 0:
        notes.append("外資投信同買：走勢容易很衝")
    elif f5 > 0 and t5 <= 0:
        notes.append("外資為主要買盤：若持續噴出、外資是唯一控盤法人，通常會一路買上去")
    elif t5 > 0 and f5 < 0:
        notes.append("投信買、外資賣：價格容易震盪")
    elif f5 < 0 and t5 < 0:
        notes.append("外資投信同賣：籌碼面偏弱")
    m = sorted(margin, key=lambda x: x["date"])
    if len(m) >= 2:
        last = m[-1]
        chg = last["MarginPurchaseTodayBalance"] - last["MarginPurchaseYesterdayBalance"]
        avg_chg = sum(abs(x["MarginPurchaseTodayBalance"] - x["MarginPurchaseYesterdayBalance"]) for x in m[-20:]) / max(len(m[-20:]), 1)
        summary["margin_change"] = chg
        if avg_chg and chg > 3 * avg_chg and chg > 0:
            notes.append("單日融資爆增：若在回檔盤整或底部，可能是聰明融資進場（他們不發動股價，要有耐心）")
    out = {"series": series, "summary": summary, "lead": lead, "notes": notes}
    db.cache_set(key, out)
    return out

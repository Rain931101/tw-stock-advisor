"""推薦引擎：由上而下（大盤 → 族群金流 → 公司體質）＋技術面進場型態＋風控。

流程（對應講義〈選股篇〉總結）：
  1. 檢查體質：淘汰沒在賺錢、體質爛的公司（fundamental.pass）
  2. 金流：族群有資金進駐（強勢族群）、法人籌碼同向
  3. 技術面：符合進場型態，並計算停損（進場要素消失）、目標價、減碼規則
  4. 風控：進場價到停損的虧損不得超過單筆可承受虧損，否則不進場
"""
import logging
from datetime import datetime, timedelta

import pandas as pd

from app import db
from app.config import get_settings
from app.data.finmind import QuotaExceeded
from app.rules import risk, technical
from app.services import company, market, notifier

log = logging.getLogger(__name__)


def risk_settings() -> dict:
    s = get_settings()
    return {
        "capital": db.get_setting("capital", s.capital),
        "per_trade_capital": db.get_setting("per_trade_capital", s.per_trade_capital),
        "max_loss_per_trade": db.get_setting("max_loss_per_trade", s.max_loss_per_trade),
        "fee_discount": db.get_setting("fee_discount", s.fee_discount),
        "min_avg_volume": db.get_setting("min_avg_volume", s.min_avg_volume),
        "min_price": db.get_setting("min_price", s.min_price),
    }


def universe(min_avg_volume: float, min_price: float) -> list[dict]:
    d = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    if not d:
        return []
    since = (datetime.fromisoformat(d) - timedelta(days=35)).date().isoformat()
    return db.query(
        """SELECT p.stock_id, s.name, s.market, s.industry, AVG(p.volume) AS avg_vol, SUM(p.value) AS sum_value,
                  (SELECT close FROM prices WHERE stock_id=p.stock_id ORDER BY date DESC LIMIT 1) AS last_close
           FROM prices p JOIN stocks s USING(stock_id)
           WHERE p.date>=? GROUP BY p.stock_id
           HAVING avg_vol>=? AND last_close>=? ORDER BY sum_value DESC""",
        (since, min_avg_volume, min_price),
    )


def _fundamental_or_none(stock_id: str, fetch: bool) -> dict | None:
    if fetch:
        try:
            return company.fundamental_report(stock_id)
        except QuotaExceeded:
            raise
        except Exception as e:
            log.warning("fundamental %s failed: %s", stock_id, e)
            return None
    return db.cache_get(f"fundamental:{stock_id}")


def build_plan(stock: dict, sig: technical.Signal, context: dict, fund: dict | None, chips: dict | None,
               sectors: list[dict], stance: str, rs: dict, price: float) -> dict | None:
    """把訊號整理成完整操作計畫；不符合風控回傳 None。"""
    if sig.stop is None or price <= sig.stop:
        return None
    sizing = risk.size_position(price, sig.stop, rs["per_trade_capital"], rs["max_loss_per_trade"], rs["fee_discount"])
    entry_high = min(sig.entry_high or price, sizing.max_entry)
    if (sig.entry_low or price) > sizing.max_entry:
        return None  # 整個進場區都超過可承受虧損：不進場
    if not sizing.ok:
        # 現價已超出可承受範圍：改為「等回到 max_entry 以下再進」
        sizing = risk.size_position(entry_high, sig.stop, rs["per_trade_capital"], rs["max_loss_per_trade"], rs["fee_discount"])
    score = sig.score
    reasons = list(sig.reasons)
    warnings = list(sig.warnings)
    tags = []
    if fund:
        score += (fund["score"] - 60) / 2
        tags.append(f"基本面 {fund['grade']}")
    top_secs = [s["industry"] for s in sectors[:5]]
    if stock.get("industry") in top_secs:
        score += 8
        reasons.append(f"所屬族群「{stock['industry']}」為今日強勢族群（有金流）")
        tags.append("強勢族群")
    if chips:
        f5, t5 = chips["summary"]["foreign"]["5d"], chips["summary"]["trust"]["5d"]
        if f5 > 0 and t5 > 0:
            score += 8
            reasons.append(f"外資投信同買（5 日外資 {f5:+,} 張、投信 {t5:+,} 張）")
        elif f5 < 0 and t5 < 0:
            score -= 8
            warnings.append(f"外資投信同賣（5 日外資 {f5:+,} 張、投信 {t5:+,} 張）")
    if stance == "防守":
        score -= 5
        warnings.append("今日大盤判斷偏防守：降低部位或等更好的成本")
    return {
        "stock_id": stock["stock_id"], "name": stock.get("name"), "industry": stock.get("industry"),
        "market": stock.get("market"), "setup": sig.setup, "setup_name": sig.name,
        "score": round(score, 1), "price": price,
        "entry_low": sig.entry_low, "entry_high": risk.round_tick(entry_high),
        "stop": sig.stop, "stop_reason": sig.stop_reason,
        "targets": sig.targets, "healthy_pullback": sig.healthy_pullback,
        "exit_rules": sig.exit_rules, "reasons": reasons, "warnings": warnings, "tags": tags,
        "sizing": sizing.to_dict(), "context": context,
        "fundamental": None if not fund else {
            "score": fund["score"], "grade": fund["grade"], "metrics": fund["metrics"],
            "highlights": [c for c in fund["checks"] if c["status"] in ("good", "bad", "warn")][:8],
        },
        "chips": None if not chips else {"summary": chips["summary"], "notes": chips["notes"]},
    }


def save_recommendation(plan: dict, trade_date: str, intraday: bool) -> int | None:
    exists = db.one("SELECT id FROM recommendations WHERE trade_date=? AND stock_id=? AND setup=?",
                    (trade_date, plan["stock_id"], plan["setup"]))
    if exists:
        return None
    rid = db.execute(
        "INSERT INTO recommendations(created_at,trade_date,stock_id,name,setup,score,status,plan) VALUES (?,?,?,?,?,?,?,?)",
        (db.now(), trade_date, plan["stock_id"], plan["name"], plan["setup"], plan["score"], "active", db.dumps(plan)),
    )
    sz = plan["sizing"]
    tgt = "、".join(f"{t['price']}（{t['label']}）" for t in plan["targets"][:2]) or "—"
    when = "盤中" if intraday else "收盤"
    notifier.alert(
        "buy",
        f"{when}符合進場：{plan['stock_id']} {plan['name']}｜{plan['setup_name']}",
        f"現價 {plan['price']}，進場區 {plan['entry_low']}~{plan['entry_high']}\n"
        f"停損 {plan['stop']}（{plan['stop_reason']}）\n目標 {tgt}\n"
        f"建議 {sz['shares']:,} 股，停損虧損約 {sz['loss_at_stop']:,} 元\n買進後請到網站按「我已買進」以啟動持股監控",
        dedup_key=f"rec:{trade_date}:{plan['stock_id']}:{plan['setup']}",
        stock_id=plan["stock_id"], level="high", ref_type="recommendation", ref_id=rid,
    )
    return rid


def expire_old(days: int = 3) -> None:
    dates = db.query("SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT ?", (days,))
    if len(dates) < days:
        return
    cutoff = dates[-1]["date"]
    db.execute("UPDATE recommendations SET status='expired' WHERE status='active' AND trade_date<?", (cutoff,))


def daily_scan(fetch_fundamentals: bool = True, max_fundamental_fetch: int = 60) -> dict:
    """收盤後全市場掃描。"""
    rs = risk_settings()
    uni = universe(rs["min_avg_volume"], rs["min_price"])
    if not uni:
        return {"candidates": 0, "recommended": 0, "message": "尚無價格資料，請先完成回補"}
    trade_date = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    since = (datetime.fromisoformat(trade_date) - timedelta(days=420)).date().isoformat()
    allp = db.load_all_prices(since)
    allp["date"] = pd.to_datetime(allp["date"])
    groups = {sid: g.set_index("date").drop(columns="stock_id") for sid, g in allp.groupby("stock_id")}
    try:
        mk = market.overview()
    except Exception:
        mk = {"sectors": [], "verdict": {"stance": "中性"}}
    sectors, stance = mk.get("sectors") or [], mk["verdict"]["stance"]
    db.cache_set("sectors", sectors)

    candidates = []
    watch = []
    for st in uni:
        df = groups.get(st["stock_id"])
        if df is None or len(df) < 70:
            continue
        res = technical.evaluate(df)
        ctx = res["context"]
        if not ctx:
            continue
        if ctx["level"] != "弱位階" and ctx["structure"] != "downtrend":
            watch.append(st["stock_id"])
        for sig in res["signals"]:
            if sig.triggered:
                candidates.append((st, sig, ctx))
    candidates.sort(key=lambda x: x[1].score, reverse=True)

    fetched, saved = 0, 0
    for st, sig, ctx in candidates:
        fund = db.cache_get(f"fundamental:{st['stock_id']}", 24 * 7)
        if fund is None and fetch_fundamentals and fetched < max_fundamental_fetch:
            fetched += 1
            try:
                fund = _fundamental_or_none(st["stock_id"], True)
            except QuotaExceeded:
                fetch_fundamentals = False
        if fund is None:
            fund = db.cache_get(f"fundamental:{st['stock_id']}")  # 過期的也先用
        if fund is None or not fund.get("pass"):
            continue  # 基本面不合格或未知：不推薦（兩種武器缺一不可）
        try:
            chips = company.chips(st["stock_id"]) if fetch_fundamentals else db.cache_get(f"chips:{st['stock_id']}:20")
        except Exception:
            chips = None
        plan = build_plan(st, sig, ctx, fund, chips, sectors, stance, rs, ctx["close"])
        if plan and save_recommendation(plan, trade_date, intraday=False):
            saved += 1
    db.cache_set("watchlist", watch[:300])
    expire_old()
    notifier.flush()
    return {"trade_date": trade_date, "universe": len(uni), "candidates": len(candidates), "recommended": saved,
            "fundamental_fetched": fetched}


def refresh_fundamentals(limit: int = 80) -> int:
    """背景慢慢更新母體的基本面（FinMind 有額度限制）。"""
    rs = risk_settings()
    n = 0
    for st in universe(rs["min_avg_volume"], rs["min_price"]):
        if db.cache_get(f"fundamental:{st['stock_id']}", 24 * 7) is not None:
            continue
        try:
            company.fundamental_report(st["stock_id"])
        except QuotaExceeded:
            break
        except Exception as e:
            log.warning("fundamental %s: %s", st["stock_id"], e)
        n += 1
        if n >= limit:
            break
    return n

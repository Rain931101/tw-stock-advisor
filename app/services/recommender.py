"""推薦引擎：由上而下（大盤 → 族群金流 → 公司體質）＋技術面進場型態＋風控。

流程（對應講義〈選股篇〉總結）：
  1. 檢查體質：淘汰沒在賺錢、體質爛的公司（fundamental.pass）
  2. 金流：族群有資金進駐（強勢族群）、法人籌碼同向
  3. 技術面：符合進場型態，並計算停損（進場要素消失）、目標價、減碼規則
  4. 風控：進場價到停損的虧損不得超過單筆可承受虧損，否則不進場
"""
import logging
import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from app import db
from app.config import get_settings
from app.data import finmind
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
        "fee_monthly_rebate": db.get_setting("fee_monthly_rebate", s.fee_monthly_rebate),
        "min_avg_volume": db.get_setting("min_avg_volume", s.min_avg_volume),
        "min_price": db.get_setting("min_price", s.min_price),
        "margin_enabled": db.get_setting("margin_enabled", s.margin_enabled),
        "margin_ratio_twse": db.get_setting("margin_ratio_twse", s.margin_ratio_twse),
        "margin_ratio_tpex": db.get_setting("margin_ratio_tpex", s.margin_ratio_tpex),
        "margin_rate": db.get_setting("margin_rate", s.margin_rate),
        "margin_maintenance": db.get_setting("margin_maintenance", s.margin_maintenance),
    }


def margin_ratio(market: str | None, rs: dict) -> float:
    return rs["margin_ratio_tpex"] if market == "tpex" else rs["margin_ratio_twse"]


# 延伸規則（非講義）：講義沒有教何時該用融資，以下是保守的使用條件
MARGIN_SETUPS = {"breakout", "ma60_reclaim", "ma5_pullback", "ma20_ride"}  # 波段型態；短打與猜底不建議用槓桿


def margin_option(stock: dict, sig_setup: str, price: float, stop: float, fund: dict | None, stance: str,
                  cash_shares: int, rs: dict) -> dict:
    ratio = margin_ratio(stock.get("market"), rs)
    m = risk.size_margin(price, stop, rs["per_trade_capital"], rs["max_loss_per_trade"], ratio,
                         rs["margin_rate"], rs["margin_maintenance"], fee_discount=rs["fee_discount"])
    why_not = []
    if not rs["margin_enabled"]:
        why_not.append("設定中未開啟融資（需有信用帳戶）")
    if not m.ok:
        why_not.append(m.reason)
    if not fund or fund.get("grade") not in ("A", "B"):
        why_not.append("基本面未達 A／B：槓桿只用在體質最好的公司")
    if stance == "防守":
        why_not.append("大盤判斷偏防守：不加槓桿")
    if sig_setup not in MARGIN_SETUPS:
        why_not.append("短打／猜底型態不適合加槓桿")
    if m.ok and m.shares <= cash_shares:
        why_not.append("受單筆最大虧損限制，融資也買不到更多股數，用現股即可")
    return {**m.to_dict(), "recommended": not why_not, "why_not": why_not,
            "note": "延伸規則（非講義）：融資虧損以整個部位計算，單筆最大虧損不變；另需確認該股可融資（非全額交割、處置、停資）"}


def universe(min_avg_volume: float, min_price: float) -> list[dict]:
    d = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    if not d:
        return []
    since = (datetime.fromisoformat(d) - timedelta(days=35)).date().isoformat()
    return db.query(
        """SELECT p.stock_id, s.name, s.market, s.industry, AVG(p.volume) AS avg_vol, SUM(p.value) AS sum_value,
                  (SELECT close FROM prices WHERE stock_id=p.stock_id ORDER BY date DESC LIMIT 1) AS last_close
           FROM prices p JOIN stocks s USING(stock_id)
           WHERE p.date>=? AND p.stock_id NOT LIKE '00%' GROUP BY p.stock_id
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
    # 講義：進單一定要思考能操作的空間——最遠目標的報酬／風險至少 1.5
    risk_ps = price - sig.stop
    rr = [(t["price"] - price) / risk_ps for t in sig.targets]
    if not rr or max(rr) < 1.5:
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
        "score": round(score, 1), "price": price, "reward_risk": round(max(rr), 2),
        "entry_low": sig.entry_low, "entry_high": risk.round_tick(entry_high),
        "stop": sig.stop, "stop_reason": sig.stop_reason,
        "targets": sig.targets, "healthy_pullback": sig.healthy_pullback,
        "exit_rules": sig.exit_rules, "reasons": reasons, "warnings": warnings, "tags": tags,
        "sizing": sizing.to_dict(), "context": context,
        "margin": margin_option(stock, sig.setup, price, sig.stop, fund, stance, sizing.shares, rs),
        "fundamental": None if not fund else {
            "score": fund["score"], "grade": fund["grade"], "metrics": fund["metrics"],
            "highlights": [c for c in fund["checks"] if c["status"] in ("good", "bad", "warn")][:8],
        },
        "chips": None if not chips else {"summary": chips["summary"], "notes": chips["notes"]},
    }


def save_recommendation(plan: dict, trade_date: str, intraday: bool, notify: bool = True) -> int | None:
    exists = db.one("SELECT id FROM recommendations WHERE trade_date=? AND stock_id=? AND setup=?",
                    (trade_date, plan["stock_id"], plan["setup"]))
    if exists:
        return None
    rid = db.execute(
        "INSERT INTO recommendations(created_at,trade_date,stock_id,name,setup,score,status,plan) VALUES (?,?,?,?,?,?,?,?)",
        (db.now(), trade_date, plan["stock_id"], plan["name"], plan["setup"], plan["score"], "active", db.dumps(plan)),
    )
    if not notify:
        return rid
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


def remind_unconfirmed(day: str) -> int:
    """收盤後提醒：今天的推薦你買了嗎？買了請到網站確認，系統才會開始監控停損減碼。"""
    rows = db.query("SELECT id, stock_id, name FROM recommendations WHERE status='active' AND trade_date=?", (day,))
    if not rows:
        return 0
    names = "、".join(f"{r['stock_id']} {r['name']}" for r in rows[:10])
    notifier.alert("info", f"今天有 {len(rows)} 檔推薦尚未確認", f"{names}\n若已買進，請到網站「推薦」按「我已買進」；沒買可按「不買」。",
                   dedup_key=f"remind:{day}", level="info")
    return len(rows)


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


fund_status = {"running": False, "done": 0, "total": 0, "message": ""}
_fund_lock = threading.Lock()
FRESH_HOURS = 24 * 7


def all_active_stocks() -> list[str]:
    """近 10 個交易日有成交的上市櫃普通股（排除 ETF），成交金額大的先更新。"""
    dates = db.query("SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT 10")
    if not dates:
        return []
    rows = db.query(
        """SELECT p.stock_id, SUM(p.value) AS v FROM prices p JOIN stocks s USING(stock_id)
           WHERE p.date>=? AND p.stock_id NOT LIKE '00%' AND s.market IN ('twse','tpex')
           GROUP BY p.stock_id ORDER BY v DESC""", (dates[-1]["date"],))
    return [r["stock_id"] for r in rows]


def refresh_all_fundamentals(force: bool = False) -> dict:
    """更新全部股票的基本面。FinMind 免費額度有限：額度用完會等待後自動接續；
    中斷（例如重啟）後再次執行會跳過 7 天內已更新的股票，從中斷處繼續。"""
    if not _fund_lock.acquire(blocking=False):
        return fund_status
    try:
        db.set_setting("fundamentals_full_pending", True)
        ids = all_active_stocks()
        fresh = lambda sid: db.cache_get(f"fundamental:{sid}", FRESH_HOURS) is not None  # noqa: E731
        todo = ids if force else [s for s in ids if not fresh(s)]
        fund_status.update(running=True, total=len(ids), done=len(ids) - len(todo), message="")
        for sid in todo:
            for attempt in range(1000):
                try:
                    with finmind.background():
                        company.fundamental_report(sid, max_age_hours=0 if force else FRESH_HOURS)
                    break
                except QuotaExceeded:
                    resume = (datetime.now() + timedelta(minutes=15)).strftime("%H:%M")
                    fund_status["message"] = f"FinMind 額度用完，{resume} 自動繼續"
                    time.sleep(15 * 60)
                except Exception as e:
                    log.warning("fundamental %s: %s", sid, e)
                    if attempt >= 2:
                        break
                    time.sleep(10)
            fund_status["done"] += 1
            left = fund_status["total"] - fund_status["done"]
            per_hour = 600 if get_settings().finmind_token else 300
            fund_status["message"] = f"約剩 {left * 4 / per_hour:.1f} 小時" if left else ""
        fund_status["message"] = "全部完成 ✓"
        db.set_setting("fundamentals_full_pending", False)
        return dict(fund_status)
    finally:
        fund_status["running"] = False
        _fund_lock.release()


def refresh_fundamentals(limit: int = 80) -> int:
    """（相容舊排程）改為呼叫全市場更新。"""
    refresh_all_fundamentals()
    return fund_status["done"]


# ======================= 個股：買進與否 =======================
def _plan_block_reason(sig: technical.Signal, price: float, rs: dict) -> str:
    if sig.stop is None or price <= sig.stop:
        return "現價已在停損價之下"
    rr = [(t["price"] - price) / (price - sig.stop) for t in sig.targets]
    if not rr or max(rr) < 1.5:
        return f"操作空間不足：到目標的報酬／風險只有 {max(rr) if rr else 0:.1f}（至少要 1.5）"
    sz = risk.size_position(price, sig.stop, rs["per_trade_capital"], rs["max_loss_per_trade"], rs["fee_discount"])
    if (sig.entry_low or price) > sz.max_entry:
        return f"停損距離太遠：在可承受虧損內最高只能買到 {sz.max_entry}，整個進場區都超過"
    return "未通過風控"


def stock_verdict(stock: dict, tech: dict, fund: dict | None, chips: dict | None, price: float,
                  stance: str, sectors: list[dict]) -> dict:
    """依兩份講義給出：建議買進／觀望／不建議買進，並附理由與操作計畫。"""
    rs = risk_settings()
    ctx = tech.get("context") or {}
    sigs: list[technical.Signal] = tech.get("signals") or []
    is_etf = stock["stock_id"].startswith("00")
    out = {"decision": "wait", "label": "觀望", "summary": "", "reasons": [], "cautions": [], "plan": None, "closest": None}

    # 1) 基本面：好公司（兩種武器缺一不可）
    if not is_etf:
        if fund is None:
            out["cautions"].append("基本面資料尚未取得（全市場更新中），暫時無法判斷公司體質")
        elif not fund.get("pass"):
            bad = [f"{c['name']}：{c['detail']}" for c in fund["checks"] if c["status"] in ("bad", "warn")][:4]
            return {**out, "decision": "avoid", "label": "不建議買進",
                    "summary": f"基本面健檢未通過（{fund['grade']}，{fund['score']} 分）：講義第一步就是淘汰體質不好的公司",
                    "reasons": bad}
        else:
            out["reasons"].append(f"基本面 {fund['grade']}（{fund['score']} 分），體質合格")
    else:
        out["cautions"].append("ETF 不做公司基本面健檢，只依技術面判斷")

    # 2) 技術面位階與格局
    if ctx.get("level") == "弱位階":
        return {**out, "decision": "avoid", "label": "不建議買進",
                "summary": "弱位階（年線之下、季線在年線下）：便宜≠風險小，越弱的位階資金越不愛",
                "reasons": out["reasons"] + [f"位階 {ctx.get('level')}、格局 {ctx.get('structure')}"]}
    if ctx.get("structure") == "downtrend":
        return {**out, "decision": "avoid", "label": "不建議買進",
                "summary": "格局低過低、高不過高，屬空方格局，等格局轉換（先看到高過高）再說",
                "reasons": out["reasons"]}

    # 3) 進場型態＋風控
    triggered = [s for s in sigs if s.triggered]
    if (fund and fund.get("pass")) or is_etf or fund is None:
        plans, blocked = [], []
        for s in triggered:
            p = build_plan(stock, s, ctx, fund, chips, sectors, stance, rs, price)
            if p:
                plans.append(p)
            else:
                blocked.append(f"{s.name}：{_plan_block_reason(s, price, rs)}")
        if plans and fund is not None:
            best = max(plans, key=lambda p: p["score"])
            day = datetime.now().date().isoformat()
            rid = save_recommendation(best, day, intraday=False, notify=False)
            if rid is None:
                row = db.one("SELECT id FROM recommendations WHERE trade_date=? AND stock_id=? AND setup=?",
                             (day, best["stock_id"], best["setup"]))
                rid = row["id"] if row else None
            out["rec_id"] = rid
            out.update(decision="buy", label="建議買進", plan=best,
                       summary=f"符合「{best['setup_name']}」，進場區 {best['entry_low']}～{best['entry_high']}，"
                               f"停損 {best['stop']}，報酬／風險 {best['reward_risk']}")
            out["reasons"] += best["reasons"]
            out["cautions"] += best["warnings"]
            return out
        if plans and fund is None:
            out["plan"] = max(plans, key=lambda p: p["score"])
            out["summary"] = "技術面已符合進場型態，但基本面尚未確認，先觀望（講義：兩種武器缺一不可）"
            return out
        if blocked:
            out["summary"] = "技術面有型態成立，但不符合風控或操作空間，暫不進場"
            out["reasons"] += blocked
            return out

    # 4) 沒有成立的型態：找最接近的，告訴使用者在等什麼
    cands = [s for s in sigs if s.reasons or s.missing]
    if cands:
        near = max(cands, key=lambda s: len(s.reasons) / max(len(s.reasons) + len(s.missing), 1))
        out["closest"] = {"name": near.name, "met": near.reasons, "missing": near.missing}
        out["summary"] = f"目前沒有成立的進場型態。最接近的是「{near.name}」，還差 {len(near.missing)} 個條件"
    else:
        out["summary"] = "目前沒有成立的進場型態"
    if stance == "防守":
        out["cautions"].append("今日大盤判斷偏防守")
    return out

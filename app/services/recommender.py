"""推薦引擎：由上而下（大盤 → 族群金流 → 公司體質）＋技術面進場型態＋風控。

流程（對應講義〈選股篇〉總結）：
  1. 檢查體質：淘汰沒在賺錢、體質爛的公司（fundamental.pass）
  2. 金流：族群有資金進駐（強勢族群）、法人籌碼同向
  3. 技術面：符合進場型態，並計算停損（進場要素消失）、目標價、減碼規則
  4. 風控：進場價到停損的虧損不得超過單筆可承受虧損，否則不進場
"""
import json
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


def trade_day() -> str:
    """推薦日期一律用最新交易日（週末、盤前開個股頁也不會多出一筆重複推薦）。"""
    return db.one("SELECT MAX(date) AS d FROM prices")["d"] or datetime.now().date().isoformat()


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


def revalidate_active() -> int:
    """風控設定改變後，重新檢查有效推薦；停損距離超出新的可承受範圍就標為失效。"""
    rs = risk_settings()
    n = 0
    for r in db.query("SELECT id, plan FROM recommendations WHERE status='active'"):
        p = json.loads(r["plan"])
        if not p.get("stop"):
            continue
        s = risk.size_position(p["price"], p["stop"], rs["per_trade_capital"], rs["max_loss_per_trade"], rs["fee_discount"])
        if (p.get("entry_low") or p["price"]) > s.max_entry:
            db.execute("UPDATE recommendations SET status='invalid' WHERE id=?", (r["id"],))
            n += 1
    return n


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
        wr = winrate_for(st["stock_id"])
        if plan and wr and wr.get("trained") and wr["decision"] == "avoid":
            continue  # 歷史勝率低於不推門檻
        if plan and wr and wr.get("trained"):
            plan["winrate"] = {"rate": wr["rate"], "exp": wr["exp"]}
            plan["reasons"].append(f"歷史勝率 {wr['rate']:.0%}、每筆期望報酬 {wr['exp']:+.2f}%")
        if plan and save_recommendation(plan, trade_date, intraday=False):
            saved += 1
    # 勝率規則推薦：不必符合講義型態，只要歷史勝率達到買進門檻
    try:
        from app.services import winrate
        m = winrate.model()
        if m:
            feats = winrate.today_features()
            by_id = {u["stock_id"]: u for u in uni}
            for sid, cond in feats.items():
                if sid not in by_id:
                    continue
                wr = winrate.assess(cond, m)
                if wr["decision"] != "buy":
                    continue
                fund = db.cache_get(f"fundamental:{sid}")
                if fund and fund.get("fatal"):
                    continue
                price = float(groups[sid]["close"].iloc[-1]) if sid in groups else None
                plan = price and winrate_plan(by_id[sid], wr, price, fund, db.cache_get(f"chips:{sid}:20"), rs)
                if plan and save_recommendation(plan, trade_date, intraday=False):
                    saved += 1
    except Exception as e:
        log.warning("winrate scan failed: %s", e)
    # 多數條件成立推薦：型態只差少數條件，但歷史上同樣情況兩期都達推薦門檻（基本面須合格）
    try:
        from app.services import nearmatch, winrate
        nm_model = nearmatch.model()
        if nm_model:
            by_id = {u["stock_id"]: u for u in uni}
            feats = winrate.today_features()
            for sid in nearmatch.today():
                if sid not in by_id:
                    continue
                nm = nearmatch.assess(sid, nm_model, feats.get(sid) or {})
                if not nm or not nm["best"] or nm["weak"]:
                    continue
                fund = db.cache_get(f"fundamental:{sid}")
                if not fund or not fund.get("pass"):
                    continue
                price = float(groups[sid]["close"].iloc[-1]) if sid in groups else None
                plan = price and near_plan(by_id[sid], nm["best"], nm["params"], price, fund, db.cache_get(f"chips:{sid}:20"), rs)
                if plan and save_recommendation(plan, trade_date, intraday=False):
                    saved += 1
    except Exception as e:
        log.warning("nearmatch scan failed: %s", e)
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
    """固定輸出格式：不論結論為何，都附上最接近成立的進場型態（closest）與大盤狀態。"""
    v = _stock_verdict(stock, tech, fund, chips, price, stance, sectors)
    sigs: list[technical.Signal] = tech.get("signals") or []
    if v.get("closest") is None and v["decision"] != "buy":
        cands = [s for s in sigs if not s.triggered and (s.reasons or s.missing)]
        if cands:
            near = max(cands, key=lambda s: len(s.reasons) / max(len(s.reasons) + len(s.missing), 1))
            v["closest"] = {"name": near.name, "met": near.reasons, "missing": near.missing}
    v["triggered"] = [s.name for s in sigs if s.triggered]
    v["stance"] = stance
    if stance == "防守" and "今日大盤判斷偏防守" not in v["cautions"]:
        v["cautions"].append("今日大盤判斷偏防守")
    return v


def _stock_verdict(stock: dict, tech: dict, fund: dict | None, chips: dict | None, price: float,
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
            day = trade_day()
            rid = save_recommendation(best, day, intraday=False)
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



# ======================= 歷史勝率 =======================
def winrate_for(stock_id: str) -> dict | None:
    from app.services import winrate
    m = winrate.model()
    if not m:
        return None
    cond = winrate.today_features().get(stock_id)
    if cond is None:
        return None
    return winrate.assess(cond, m)


def winrate_plan(stock: dict, wr: dict, price: float, fund: dict | None, chips: dict | None, rs: dict,
                 setup: str = "winrate", setup_name: str | None = None) -> dict | None:
    """勝率規則的操作計畫：停利、停損、持有天數與訓練時的交易定義完全相同。"""
    p = wr["params"]
    stop = risk.round_tick(price * (1 - p["stop"]))
    target = risk.round_tick(price * (1 + p["target"]))
    sizing = risk.size_position(price, stop, rs["per_trade_capital"], rs["max_loss_per_trade"], rs["fee_discount"])
    if not sizing.ok:
        return None
    rule = wr["rule"]
    return {
        "stock_id": stock["stock_id"], "name": stock.get("name"), "industry": stock.get("industry"), "market": stock.get("market"),
        "setup": setup, "setup_name": setup_name or "歷史勝率規則：" + " + ".join(rule["names"]),
        "score": round(60 + rule["exp"] * 10, 1), "price": price, "reward_risk": round(p["target"] / p["stop"], 2),
        "entry_low": risk.round_tick(price * 0.99), "entry_high": risk.round_tick(price * 1.01),
        "stop": stop, "stop_reason": f"訓練時的停損：進場價 −{p['stop']:.0%}",
        "targets": [{"price": target, "label": f"停利 +{p['target']:.0%}"}], "healthy_pullback": None,
        "exit_rules": [
            {"type": "price_below", "level": stop, "action": "sell_all", "label": f"跌破停損（−{p['stop']:.0%}）"},
            {"type": "target", "level": target, "action": "sell_all", "label": f"達到停利（+{p['target']:.0%}）"},
            {"type": "time_exit", "days": p["hold_days"], "action": "sell_all", "label": f"持有滿 {p['hold_days']} 個交易日出場"},
        ],
        "reasons": [f"歷史勝率 {rule['rate']:.0%}（訓練 {rule['train_rate']:.0%}／驗證 {rule['test_rate']:.0%}，{rule['n']} 筆）",
                    f"每筆期望報酬 {rule['exp']:+.2f}%"] + [f"符合：{n}" for n in rule["names"]],
        "warnings": ["勝率規則是統計結果，樣本只有約一年，未來不保證相同"],
        "tags": ["勝率規則"], "sizing": sizing.to_dict(), "context": None,
        "margin": margin_option(stock, setup, price, stop, fund, "中性", sizing.shares, rs),
        "fundamental": None if not fund else {"score": fund["score"], "grade": fund["grade"], "metrics": fund.get("metrics", {}),
                                             "highlights": [c for c in fund.get("checks", []) if c["status"] in ("good", "bad", "warn")][:8]},
        "chips": None if not chips else {"summary": chips["summary"], "notes": chips["notes"]},
        "winrate": {k: wr[k] for k in ("rate", "exp")},
    }


def apply_winrate(verdict: dict, wr: dict | None, stock: dict, price: float, fund: dict | None, chips: dict | None) -> dict:
    """以每筆期望報酬判斷：兩期期望 ≥ 推薦門檻且勝率 ≥ 最低勝率 → 推薦買進（不必符合講義型態）；
    期望報酬 ≤ 不推門檻 → 不推薦；介於中間依講義規則。"""
    if not wr or not wr.get("trained"):
        verdict["winrate"] = None
        return verdict
    p = wr["params"]
    verdict["winrate"] = {
        "rate": wr["rate"], "exp": wr["exp"], "decision": wr["decision"], "rule": wr["rule"], "n_rules": wr.get("n_rules", 0),
        "matched": wr["matched"], "true_conds": wr["true_conds"], "base_rate": wr["base_rate"],
        "params": p, "trained_at": wr["trained_at"],
    }
    fatal = bool(fund and fund.get("fatal"))
    if wr["decision"] == "buy" and not fatal:
        plan = verdict.get("plan") if verdict["decision"] == "buy" else None
        if plan is None:
            plan = winrate_plan(stock, wr, price, fund, chips, risk_settings())
        if plan:
            day = trade_day()
            save_recommendation(plan, day, intraday=False)
            row = db.one("SELECT id FROM recommendations WHERE trade_date=? AND stock_id=? AND setup=?", (day, plan["stock_id"], plan["setup"]))
            verdict.update(decision="buy", label="建議買進", plan=plan, rec_id=row["id"] if row else None,
                           summary=f"歷史每筆期望報酬 {wr['exp']:+.2f}%、勝率 {wr['rate']:.0%}：" + " + ".join(wr["rule"]["names"]))
            if fund and not fund.get("pass"):
                verdict["cautions"].append(f"基本面健檢未通過（{fund['grade']}）：依你的規則仍以勝率為準，請自行評估")
    elif wr["decision"] == "avoid":
        verdict.update(decision="avoid", label="不建議買進", plan=None,
                       summary=f"歷史每筆期望報酬 {wr['exp']:+.2f}%（≤ {p['avoid_exp']:+.1f}%），長期做下去是虧的，不推薦")
    return verdict


# ======================= 多數條件成立 =======================
def near_plan(stock: dict, best: dict, params: dict, price: float, fund: dict | None, chips: dict | None, rs: dict) -> dict | None:
    st = best["stats"]
    rule = {**st, "names": best["met"] + best.get("extra", [])}
    plan = winrate_plan(stock, {"params": params, "rule": rule, "rate": st["rate"], "exp": st["exp"]}, price, fund, chips, rs,
                        setup="nearmatch", setup_name=f"多數條件成立：{best['name']}（{len(best['met'])}／{best['total']}）")
    if plan:
        plan["reasons"] = [f"歷史勝率 {st['rate']:.0%}（訓練 {st['train_rate']:.0%}／驗證 {st['test_rate']:.0%}，{st['n']} 筆）",
                           f"每筆期望報酬 {st['exp']:+.2f}%（訓練 {st['train_exp']:+.2f}%／驗證 {st['test_exp']:+.2f}%）",
                           f"比對依據：{best['basis']}"] + [f"符合：{n}" for n in best["met"] + best.get("extra", [])]
        plan["warnings"] = [f"未成立：{n}（歷史上缺這項仍達推薦門檻）" for n in best["missing"]] + plan["warnings"]
        plan["tags"] = ["多數條件"]
    return plan


def apply_nearmatch(verdict: dict, stock: dict, price: float, fund: dict | None, chips: dict | None) -> dict:
    """型態只差少數條件時，用歷史上「同樣情況」的勝率與期望報酬決定：兩期都達推薦門檻就建議買進。"""
    from app.services import nearmatch
    try:
        nm = nearmatch.assess(stock["stock_id"])
    except Exception as e:
        log.warning("nearmatch %s: %s", stock["stock_id"], e)
        nm = None
    verdict["near"] = nm
    _upgrade_near(verdict, nm, stock, price, fund, chips)
    verdict["basis"] = decision_basis(verdict)
    return verdict


def _stat_fields(st: dict) -> dict:
    return {k: st.get(k) for k in ("rate", "exp", "n", "train_rate", "test_rate", "train_exp", "test_exp")}


def decision_basis(v: dict) -> dict | None:
    """這張判斷卡「唯一」採用的歷史數字：結論用哪個統計，畫面最上方就只顯示那一個。"""
    plan, wr, nm = v.get("plan") or {}, v.get("winrate"), v.get("near") or {}
    best = nm.get("best")
    params = (wr or {}).get("params") or nm.get("params")
    if v["decision"] == "buy" and plan.get("setup") == "nearmatch" and best:
        return {"source": "nearmatch", "title": f"多數條件：{best['name']}（{len(best['met'])}／{best['total']}）",
                "desc": best["basis"], **_stat_fields(best["stats"]), "params": params, "pass": True}
    if wr and (wr["decision"] in ("buy", "avoid") or not best):
        rule = wr.get("rule") or {}
        title = "條件組合：" + " + ".join(rule.get("names", [])) if rule else "全市場平均（沒有符合的條件組合）"
        return {"source": "winrate", "title": title, "desc": "今天成立的量價／市場條件組合",
                **_stat_fields({**rule, "rate": wr["rate"], "exp": wr["exp"]}), "params": params,
                "pass": wr["decision"] == "buy"}
    # 其餘：取此股今天最接近的型態統計（與「多數條件勝率」第一列一致）
    items = [i for i in nm.get("items", []) if i.get("stats")]
    if items:
        it = items[0]
        return {"source": "nearmatch", "title": f"多數條件：{it['name']}（{len(it['met'])}／{it['total']}）",
                "desc": it["basis"], **_stat_fields(it["stats"]), "params": params, "pass": it["decision"] == "buy"}
    return None


def _upgrade_near(verdict: dict, nm: dict | None, stock: dict, price: float, fund: dict | None, chips: dict | None) -> None:
    if not nm or not nm["best"] or verdict["decision"] != "wait":
        return
    best = nm["best"]
    st = best["stats"]
    if fund is None:
        verdict["cautions"].append(f"「{best['name']}」多數條件成立且歷史期望 {st['exp']:+.2f}%，但基本面尚未確認，先觀望")
        return
    if not fund.get("pass"):
        return
    plan = near_plan(stock, best, nm["params"], price, fund, chips, risk_settings())
    if not plan:
        return
    day = trade_day()
    save_recommendation(plan, day, intraday=False)
    row = db.one("SELECT id FROM recommendations WHERE trade_date=? AND stock_id=? AND setup=?", (day, plan["stock_id"], plan["setup"]))
    verdict.update(decision="buy", label="建議買進", plan=plan, rec_id=row["id"] if row else None,
                   summary=f"「{best['name']}」符合 {len(best['met'])}／{best['total']} 個條件；歷史上同樣情況勝率 {st['rate']:.0%}、"
                           f"每筆期望報酬 {st['exp']:+.2f}%（訓練、驗證兩期都達推薦門檻）")
    verdict["reasons"] = plan["reasons"] + verdict["reasons"]
    verdict["closest"] = {"name": best["name"], "met": best["met"], "missing": best["missing"]}

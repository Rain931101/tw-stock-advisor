"""持股：確認買進、賣出紀錄、出場規則判讀、持股健檢。"""
import json
from datetime import datetime

import pandas as pd

from app import db
from app.rules import indicators as ind
from app.rules import technical
from app.services import recommender


def _pos(row: dict) -> dict:
    row = dict(row)
    row["plan"] = json.loads(row["plan"]) if row.get("plan") else {}
    return row


def confirm_buy(rec_id: int | None, stock_id: str, price: float, shares: int, buy_date: str | None = None) -> dict:
    """記錄買進。停損一律沿用系統計畫（進場要素消失的位置），不因成交價改變。"""
    plan, name, setup = {}, None, "manual"
    if rec_id:
        rec = db.one("SELECT * FROM recommendations WHERE id=?", (rec_id,))
        if rec:
            plan = json.loads(rec["plan"])
            name, setup, stock_id = rec["name"], rec["setup"], rec["stock_id"]
            db.execute("UPDATE recommendations SET status='bought' WHERE id=?", (rec_id,))
    if not name:
        s = db.one("SELECT name FROM stocks WHERE stock_id=?", (stock_id,))
        name = s["name"] if s else stock_id
    if not plan:
        plan = manual_plan(stock_id, price)
    stop = plan.get("stop")
    plan["initial_stop"] = stop
    plan["stop_history"] = [{"date": buy_date or datetime.now().date().isoformat(), "stop": stop,
                             "reason": plan.get("stop_reason") or "進場要素消失的位置"}]
    plan["state"] = {"target_hit": False, "high_since_entry": price}
    # 成交價與計畫的落差只提醒、不改停損
    warnings = []
    rs = recommender.risk_settings()
    if stop and price <= stop:
        warnings.append(f"成交價 {price} 已低於停損價 {stop}：進場要素已消失，建議出場")
    elif stop:
        loss = (price - stop) * shares
        if loss > rs["max_loss_per_trade"]:
            warnings.append(f"以成交價 {price}、{shares:,} 股計，打到停損 {stop} 約虧 {loss:,.0f} 元，"
                            f"超過單筆可承受虧損 {rs['max_loss_per_trade']:,.0f} 元，建議減少股數")
    plan["entry_warnings"] = warnings
    pid = db.execute(
        "INSERT INTO positions(rec_id,stock_id,name,setup,buy_date,buy_price,shares,remaining,plan,status) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (rec_id, stock_id, name, setup, buy_date or datetime.now().date().isoformat(), price, shares, shares, db.dumps(plan), "open"),
    )
    db.execute("INSERT INTO trades(position_id,ts,side,price,shares,reason) VALUES (?,?,?,?,?,?)",
               (pid, db.now(), "buy", price, shares, plan.get("setup_name", "手動建立")))
    return {"id": pid, "stop": stop, "stop_reason": plan.get("stop_reason"), "warnings": warnings}


def trail_stop(pos: dict, df: pd.DataFrame) -> tuple[float, str] | None:
    """收盤後依講義規則上移停損（只上不下）。回傳 (新停損, 理由) 或 None。

    1. 獲利達 2R：停損上移至成本（保本），用對的虧損換取對的獲利
    2. 目標價滿足後：改用五日線移動停利（講義：我都用 5 日線當防守跟移動保利）
    3. 以五日線防守的型態：停損跟著五日線往上走
    """
    plan = pos["plan"]
    stop = plan.get("stop")
    init = plan.get("initial_stop") or stop
    if not stop or len(df) < 6:
        return None
    d = ind.add_indicators(df)
    r = d.iloc[-1]
    close, ma5 = float(r["close"]), float(r["ma5"])
    buy = pos["buy_price"]
    cands = []
    if init and buy > init and close >= buy + 2 * (buy - init):
        cands.append((buy, "獲利達 2R，停損上移至成本（保本）"))
    if plan.get("state", {}).get("target_hit") and ma5 < close:
        cands.append((ma5 * 0.995, "目標價已滿足，改用五日線移動停利"))
    if pos.get("setup") in ("ma5_pullback", "rebound_ma5") and ma5 < close:
        cands.append((ma5 * 0.985, "此型態以五日線防守，停損跟著五日線上移"))
    if not cands:
        return None
    new, reason = max(cands, key=lambda x: x[0])
    from app.rules.risk import round_tick
    new = round_tick(new)
    if new <= stop * 1.001 or new >= close:
        return None
    return new, reason


def apply_stop(pos: dict, new: float, reason: str, day: str) -> None:
    plan = pos["plan"]
    plan["stop"] = new
    plan["stop_reason"] = reason
    for r in plan.get("exit_rules", []):
        if r["type"] == "price_below":
            r["level"] = new
    plan.setdefault("stop_history", []).append({"date": day, "stop": new, "reason": reason})
    db.execute("UPDATE positions SET plan=? WHERE id=?", (db.dumps(plan), pos["id"]))


def manual_plan(stock_id: str, price: float) -> dict:
    """使用者自行買進（非推薦）時，依目前技術面套一個最接近的計畫：守五日線／季線。"""
    df = db.load_prices(stock_id)
    plan = {"setup": "manual", "setup_name": "手動建立持股", "targets": [], "reasons": [], "warnings": []}
    if len(df) >= 70:
        res = technical.evaluate(df)
        ctx = res["context"]
        best = max(res["signals"], key=lambda s: (s.triggered, s.score), default=None)
        if best and best.stop and best.stop < price:
            plan.update(best.to_dict())
            plan["setup_name"] = f"手動建立（參考：{best.name}）"
        else:
            ma60 = ctx["ma"][60] if ctx else None
            stop = round(min(price * 0.92, ma60 * 0.99) if ma60 else price * 0.92, 2)
            plan.update(stop=stop, stop_reason="預設：跌破季線或虧損 8%",
                        exit_rules=[{"type": "price_below", "level": stop, "action": "sell_all", "label": "跌破停損價"},
                                    {"type": "ma_below", "ma": 60, "action": "sell_half", "label": "跌破季線減碼一半"}])
        plan["context"] = ctx
    else:
        stop = round(price * 0.92, 2)
        plan.update(stop=stop, stop_reason="資料不足，預設虧損 8%",
                    exit_rules=[{"type": "price_below", "level": stop, "action": "sell_all", "label": "跌破停損價"}])
    return plan


def record_sell(position_id: int, price: float, shares: int, reason: str = "手動賣出") -> dict:
    p = db.one("SELECT * FROM positions WHERE id=?", (position_id,))
    if not p or p["status"] != "open":
        raise ValueError("持股不存在或已結束")
    shares = min(shares, p["remaining"])
    pnl = (price - p["buy_price"]) * shares - price * shares * 0.003 - (price + p["buy_price"]) * shares * 0.001425 * recommender.risk_settings()["fee_discount"]
    remaining = p["remaining"] - shares
    db.execute("INSERT INTO trades(position_id,ts,side,price,shares,reason) VALUES (?,?,?,?,?,?)",
               (position_id, db.now(), "sell", price, shares, reason))
    db.execute("UPDATE positions SET remaining=?, realized_pnl=realized_pnl+?, status=?, closed_at=? WHERE id=?",
               (remaining, round(pnl), "open" if remaining > 0 else "closed", None if remaining > 0 else db.now(), position_id))
    return {"remaining": remaining, "pnl": round(pnl)}


def open_positions() -> list[dict]:
    return [_pos(r) for r in db.query("SELECT p.*, s.market FROM positions p LEFT JOIN stocks s USING(stock_id) WHERE status='open' ORDER BY id")]


# ---------------- 出場規則判讀 ----------------
def evaluate_exits(pos: dict, df: pd.DataFrame, now: datetime | None, intraday: bool) -> list[dict]:
    """df 最後一根為今日（盤中合成或收盤）。回傳觸發的動作 [{action, label, level}]"""
    plan = pos["plan"]
    state = plan.setdefault("state", {})
    if len(df) < 5:
        return []
    d = ind.add_indicators(df)
    r = d.iloc[-1]
    price, low = float(r["close"]), float(r["low"])
    state["high_since_entry"] = max(state.get("high_since_entry", pos["buy_price"]), float(r["high"]))
    t = now.time() if now else None
    hits = []
    for rule in plan.get("exit_rules", []):
        typ = rule["type"]
        if typ == "price_below" and (price <= rule["level"] or (intraday and low <= rule["level"] and price < rule["level"] * 1.005)):
            hits.append(rule)
        elif typ == "close_below_ma":
            ma = r.get(f"ma{rule['ma']}")
            if ma is None or pd.isna(ma) or price >= ma:
                continue
            after = rule.get("intraday_after")
            if not intraday:
                hits.append(rule)
            elif after and t and t >= datetime.strptime(after, "%H:%M").time():
                hits.append({**rule, "label": rule["label"] + f"（{after} 後仍在 {rule['ma']} 日線下）"})
        elif typ == "ma_below":
            ma = r.get(f"ma{rule['ma']}")
            if ma is not None and not pd.isna(ma) and price < ma * (0.995 if intraday else 1):
                if not intraday or (t and t >= datetime.strptime("12:00", "%H:%M").time()):
                    hits.append(rule)
        elif typ == "target":
            if float(r["high"]) >= rule["level"]:
                if not state.get("target_hit"):
                    state["target_hit"] = True
                    hits.append({**rule, "action": "target_hit"})
                if rule["action"] == "sell_half" and not state.get("half_sold_alerted"):
                    hits.append(rule)
            if state.get("target_hit") and rule["action"] == "watch_pullback":
                # 目標價滿足後，盤中自高點壓回 2% 以上 = 弱勢，可提前出場
                if price <= state["high_since_entry"] * 0.98:
                    hits.append({**rule, "action": "sell_half", "label": "目標價滿足後盤中壓回轉弱，先減碼或出場"})
        elif typ == "stall_ma" and not intraday:
            ma = r.get(f"ma{rule['ma']}")
            n = rule.get("days", 5)
            held = (pd.Timestamp(datetime.now().date()) - pd.Timestamp(pos["buy_date"])).days
            if ma and held >= n and abs(price / ma - 1) < 0.02 and price < pos["buy_price"] * 1.02:
                hits.append(rule)
    hp = plan.get("healthy_pullback")
    if hp and state.get("target_hit") is False and price < hp and price > pos["buy_price"]:
        hits.append({"type": "healthy", "action": "notify", "label": f"跌破健康回檔價 {hp}：走勢轉弱或要走新的慣性"})
    return hits


# ---------------- 持股健檢 ----------------
def health_check() -> dict:
    rs = recommender.risk_settings()
    positions = open_positions()
    items, total_value, total_risk, by_ind = [], 0.0, 0.0, {}
    for p in positions:
        df = db.load_prices(p["stock_id"])
        last = float(df["close"].iloc[-1]) if len(df) else p["buy_price"]
        quote = db.cache_get(f"quote:{p['stock_id']}", 0.2)
        if quote and quote.get("price"):
            last = quote["price"]
        plan = p["plan"]
        stop = plan.get("stop")
        value = last * p["remaining"]
        pnl = (last - p["buy_price"]) * p["remaining"]
        risk_amt = max(0.0, (last - stop) * p["remaining"]) if stop else value
        total_value += value
        total_risk += risk_amt
        ind_name = (db.one("SELECT industry FROM stocks WHERE stock_id=?", (p["stock_id"],)) or {}).get("industry", "其他")
        by_ind[ind_name] = by_ind.get(ind_name, 0) + value
        tech = technical.evaluate(df) if len(df) >= 70 else {"context": None, "signals": []}
        ctx = tech["context"] or {}
        fund = db.cache_get(f"fundamental:{p['stock_id']}")
        notes, status = [], "ok"
        if stop and last <= stop:
            status = "danger"
            notes.append(f"已跌破停損價 {stop}：進場要素消失，應出場")
        elif stop and (last - stop) / last < 0.03:
            status = "warn"
            notes.append(f"距離停損價 {stop} 不到 3%")
        ma5 = (ctx.get("ma") or {}).get(5)
        if ma5 and last < ma5 and p["setup"] in ("ma5_pullback", "rebound_ma5", "breakout"):
            status = "warn" if status == "ok" else status
            notes.append("收在五日線之下：此型態以五日線為防守")
        ma60 = (ctx.get("ma") or {}).get(60)
        if ma60 and last < ma60:
            status = "warn" if status == "ok" else status
            notes.append("跌破季線（生命線）：可能進入時間整理或做頭")
        if ctx.get("structure") == "downtrend":
            notes.append("格局轉為低過低、高不過高")
        if ctx.get("warnings"):
            notes.extend(ctx["warnings"])
        if fund and not fund.get("pass"):
            status = "warn" if status == "ok" else status
            notes.append(f"基本面健檢未通過（{fund['grade']}）：留意公司體質變化")
        targets = plan.get("targets") or []
        next_t = next((t for t in targets if t["price"] > last), None)
        r_mult = None
        if stop and p["buy_price"] > stop:
            r_mult = round((last - p["buy_price"]) / (p["buy_price"] - stop), 2)
        if r_mult is not None and r_mult >= 2 and stop and stop < p["buy_price"]:
            notes.append("獲利已達 2R 以上：系統收盤後會自動把停損上移至成本（保本）")
        items.append({
            "id": p["id"], "stock_id": p["stock_id"], "name": p["name"], "industry": ind_name,
            "setup": plan.get("setup_name", p["setup"]), "buy_date": p["buy_date"], "buy_price": p["buy_price"],
            "shares": p["remaining"], "last": last, "value": round(value), "pnl": round(pnl),
            "pnl_pct": round((last / p["buy_price"] - 1) * 100, 2), "stop": stop,
            "stop_reason": plan.get("stop_reason"), "stop_history": plan.get("stop_history", []),
            "entry_warnings": plan.get("entry_warnings", []),
            "to_stop_pct": None if not stop else round((last / stop - 1) * 100, 2),
            "next_target": next_t, "r_multiple": r_mult, "status": status, "notes": notes,
            "fundamental_grade": fund.get("grade") if fund else None,
            "context": {k: ctx.get(k) for k in ("regime", "level", "structure", "angle", "volume_ratio")},
            "exit_rules": plan.get("exit_rules", []), "targets": targets,
            "realized_pnl": p["realized_pnl"],
        })
    capital = rs["capital"]
    port_notes = []
    if capital and total_value > capital:
        port_notes.append("持股市值超過帳戶資金設定，請確認是否使用融資")
    if total_value and by_ind:
        top_ind, top_val = max(by_ind.items(), key=lambda kv: kv[1])
        if top_val / total_value > 0.5 and len(positions) > 1:
            port_notes.append(f"持股集中在「{top_ind}」（{top_val/total_value:.0%}）：族群轉弱時會一起修正")
    if capital and total_risk / capital > 0.06:
        port_notes.append(f"全部持股同時打到停損的總虧損約 {total_risk:,.0f} 元（{total_risk/capital:.1%} 資金），偏高")
    closed = db.query("SELECT realized_pnl FROM positions WHERE status='closed'")
    wins = [c["realized_pnl"] for c in closed if c["realized_pnl"] > 0]
    stats = {
        "closed_trades": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "realized_pnl": round(sum(c["realized_pnl"] for c in closed)),
    }
    return {
        "positions": items,
        "summary": {
            "capital": capital, "market_value": round(total_value), "exposure_pct": round(total_value / capital * 100, 1) if capital else None,
            "unrealized_pnl": round(sum(i["pnl"] for i in items)), "risk_at_stop": round(total_risk),
            "industries": {k: round(v) for k, v in by_ind.items()}, **stats,
        },
        "notes": port_notes,
    }

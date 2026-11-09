"""盤中即時監控與收盤檢討。

盤中（9:00~13:30，每分鐘）：
  - 持股：依計畫的出場規則判斷停損／減碼／停利，觸發即推播
  - 觀察名單：用即時價組出今日 K 棒，判斷是否盤中符合進場條件
  - 有效推薦：價格進入進場區、或跌破停損價（推薦失效）
收盤後：用正式收盤價再跑一次出場規則（收破五日線等以收盤為準的規則）。
"""
import json
import logging
from datetime import datetime, time

from app import db
from app.data import twse
from app.rules import technical
from app.services import notifier, portfolio, recommender

log = logging.getLogger(__name__)
KIND = {"sell_all": "stop", "sell_half": "reduce", "target_hit": "target", "notify": "info", "watch_pullback": "target"}
ACTION_TEXT = {"sell_all": "建議出場", "sell_half": "建議減碼", "target_hit": "目標價達成", "notify": "注意", "watch_pullback": "目標價達成"}


def in_session(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    return now.weekday() < 5 and time(9, 0) <= now.time() <= time(13, 35)


def trading_day(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    key = f"trading_day:{now.date().isoformat()}"
    hit = db.cache_get(key)
    if hit is not None:
        return hit
    if now.weekday() >= 5:
        return False
    try:
        if time(9, 1) <= now.time() < time(13, 30):
            ok = twse.market_is_open_today()  # 盤中：即時看板今天有成交
        elif now.time() >= time(13, 30):
            from app.services import market  # 盤後：官方今天有公布收盤指數
            ok = (market.official_ohlc(now.date()).get("t00") or {}).get("date") == now.date().isoformat()
        else:
            return True  # 開盤前無法判斷，先當作交易日，不快取
    except Exception:
        return True
    if ok or now.time() >= time(9, 10):
        db.cache_set(key, ok)
    return ok


def _handle_exit_hits(pos: dict, hits: list[dict], price: float, day: str, intraday: bool) -> None:
    for h in hits:
        action = h.get("action", "notify")
        kind = KIND.get(action, "info")
        if action == "sell_all" and price >= pos["buy_price"]:
            kind = "target"  # 獲利中的移動停利
        pnl_pct = (price / pos["buy_price"] - 1) * 100
        notifier.alert(
            kind,
            f"{ACTION_TEXT.get(action, '注意')}：{pos['stock_id']} {pos['name']}",
            f"{h['label']}\n現價 {price}（成本 {pos['buy_price']}，{pnl_pct:+.1f}%），持有 {pos['remaining']:,} 股"
            + ("\n（盤中判斷）" if intraday else "\n（收盤判斷）"),
            dedup_key=f"pos:{pos['id']}:{day}:{h.get('type')}:{action}:{h.get('ma', h.get('level', ''))}",
            stock_id=pos["stock_id"], level="high" if action in ("sell_all", "sell_half") else "info",
            ref_type="position", ref_id=pos["id"],
        )
        if action == "sell_half":
            pos["plan"].setdefault("state", {})["half_sold_alerted"] = True
    db.execute("UPDATE positions SET plan=? WHERE id=?", (db.dumps(pos["plan"]), pos["id"]))


def intraday_tick(now: datetime | None = None, force: bool = False) -> dict:
    now = now or datetime.now()
    if not force and (not in_session(now) or not trading_day(now)):
        return {"skipped": True}
    day = now.date().isoformat()
    positions = portfolio.open_positions()
    recs = db.query("SELECT * FROM recommendations WHERE status='active'")
    iw = db.cache_get("intraday_watch") or {}
    watch = [x["stock_id"] for x in iw.get("items", [])] or (db.cache_get("watchlist") or [])[:150]
    ids = {p["stock_id"] for p in positions} | {r["stock_id"] for r in recs} | set(watch)
    markets = {r["stock_id"]: r["market"] for r in db.query("SELECT stock_id, market FROM stocks")}
    quotes = twse.fetch_realtime([(sid, markets.get(sid, "twse")) for sid in ids])
    for sid, q in quotes.items():
        db.cache_set(f"quote:{sid}", q)
    result = {"quotes": len(quotes), "exit_alerts": 0, "entry_alerts": 0}

    # 1) 持股出場
    for pos in positions:
        q = quotes.get(pos["stock_id"])
        if not q or q.get("price") is None:
            continue
        df = technical.synthetic_bar(db.load_prices(pos["stock_id"]), q, day, now=now)
        hits = portfolio.evaluate_exits(pos, df, now, intraday=True)
        _handle_exit_hits(pos, hits, q["price"], day, True)
        result["exit_alerts"] += len(hits)
        mi = pos["plan"].get("margin_info")
        if pos.get("mode") == "margin" and mi:
            mr = q["price"] / (pos["buy_price"] * mi["ratio"]) * 100
            if mr < mi["maintenance"] * 100 * 1.05:
                notifier.alert("stop", f"融資維持率告警：{pos['stock_id']} {pos['name']}",
                               f"現價 {q['price']}，維持率約 {mr:.0f}%（追繳線 {mi['maintenance'] * 100:.0f}%、追繳價 {mi['call_price']}）\n"
                               "跌破追繳線需補繳差額，否則會被斷頭", dedup_key=f"margincall:{pos['id']}:{day}",
                               stock_id=pos["stock_id"], level="high", ref_type="position", ref_id=pos["id"])

    # 2) 有效推薦：進入進場區／失效
    for r in recs:
        q = quotes.get(r["stock_id"])
        if not q or q.get("price") is None:
            continue
        plan = json.loads(r["plan"])
        price = q["price"]
        if price <= plan["stop"]:
            db.execute("UPDATE recommendations SET status='invalid' WHERE id=?", (r["id"],))
            notifier.alert("info", f"推薦失效：{r['stock_id']} {r['name']}",
                           f"現價 {price} 已跌破停損價 {plan['stop']}，進場要素消失，取消此筆推薦",
                           dedup_key=f"recinvalid:{r['id']}", stock_id=r["stock_id"], ref_type="recommendation", ref_id=r["id"])
        elif plan["entry_low"] <= price <= plan["entry_high"] and r["trade_date"] != day and now.time() >= time(9, 5):
            notifier.alert("buy", f"進入進場區：{r['stock_id']} {r['name']}",
                           f"現價 {price} 落在進場區 {plan['entry_low']}~{plan['entry_high']}（{plan['setup_name']}）\n停損 {plan['stop']}",
                           dedup_key=f"reczone:{r['id']}:{day}", stock_id=r["stock_id"], level="high",
                           ref_type="recommendation", ref_id=r["id"])
            result["entry_alerts"] += 1

    # 3) 盤中觀察名單：講義的盤中進場訊號（9:05 前不動作、9:30 後回測五日線、12:00 後站回季線…）成立就通知
    stocks = {r["stock_id"]: r for r in db.query("SELECT * FROM stocks")}
    active = {(r["stock_id"]) for r in recs}
    for sid in watch:
        q = quotes.get(sid)
        if not q or q.get("price") is None or sid in active:
            continue
        df = technical.synthetic_bar(db.load_prices(sid), q, day, now=now)   # 盤中量推估全日量
        res = technical.evaluate(df, now=now, intraday=True)
        if not any(s.triggered for s in res["signals"]):
            continue
        try:
            if recommender.intraday_entry(stocks.get(sid, {"stock_id": sid}), res, q["price"], day):
                result["entry_alerts"] += 1
        except Exception as e:
            log.warning("intraday entry %s: %s", sid, e)

    # 4) 大盤急殺提醒
    idx = twse.fetch_realtime([("t00", "index")]).get("t00")
    if idx and idx.get("price") and idx.get("prev_close"):
        chg = (idx["price"] / idx["prev_close"] - 1) * 100
        if chg <= -2:
            notifier.alert("market", "大盤急跌", f"加權指數 {idx['price']:.0f}（{chg:.2f}%），檢查持股停損位置；"
                           "盤勢跌時資金容易集中在強勢股", dedup_key=f"mkt:drop:{day}", level="high")
    notifier.flush()
    return result


def close_review(now: datetime | None = None) -> int:
    """收盤後以正式日 K 判讀出場規則。"""
    now = now or datetime.now()
    day = now.date().isoformat()
    n = 0
    for pos in portfolio.open_positions():
        df = db.load_prices(pos["stock_id"])
        if not len(df) or df.index[-1].date().isoformat() != day:
            continue
        hits = portfolio.evaluate_exits(pos, df, None, intraday=False)
        _handle_exit_hits(pos, hits, float(df["close"].iloc[-1]), day, False)
        n += len(hits)
        moved = portfolio.trail_stop(pos, df)
        if moved:
            old_stop = pos["plan"].get("stop")
            portfolio.apply_stop(pos, moved[0], moved[1], day)
            notifier.alert("info", f"停損上移：{pos['stock_id']} {pos['name']}",
                           f"{old_stop} → {moved[0]}（{moved[1]}）\n明天起盤中以新停損監控",
                           dedup_key=f"trail:{pos['id']}:{day}", stock_id=pos["stock_id"],
                           ref_type="position", ref_id=pos["id"])
    notifier.flush()
    return n

"""資料同步：股票清單、全市場日 K 回補與每日更新。"""
import logging
import threading
from datetime import date, datetime, timedelta

from app import db
from app.config import get_settings
from app.data import finmind, twse

log = logging.getLogger(__name__)
status = {"running": False, "done": 0, "total": 0, "message": "idle"}
_lock = threading.Lock()


def sync_stock_list(force: bool = False) -> int:
    if not force and db.cache_get("stock_list_synced", 24 * 7) is not None:
        return 0
    rows = finmind.fetch("TaiwanStockInfo", start="2020-01-01")
    seen = {}
    for r in rows:
        if r.get("type") not in ("twse", "tpex"):
            continue
        sid = r["stock_id"]
        if len(sid) != 4 or not sid.isdigit():
            continue
        seen[sid] = (sid, r["stock_name"], r["type"], r.get("industry_category") or "其他")
    with db.tx() as c:
        c.executemany("INSERT OR REPLACE INTO stocks(stock_id,name,market,industry) VALUES (?,?,?,?)", list(seen.values()))
    db.cache_set("stock_list_synced", len(seen))
    return len(seen)


def _store(rows: list[tuple], market: str, d: date) -> int:
    db.upsert_prices([r[:8] for r in rows])
    with db.tx() as c:
        c.executemany(
            "INSERT OR IGNORE INTO stocks(stock_id,name,market,industry) VALUES (?,?,?,?)",
            [(r[0], r[8], market, "其他") for r in rows],
        )
    db.execute("INSERT OR REPLACE INTO sync_log(market,date,rows) VALUES (?,?,?)", (market, d.isoformat(), len(rows)))
    return len(rows)


def sync_day(d: date) -> int:
    n = 0
    done = {r["market"] for r in db.query("SELECT market FROM sync_log WHERE date=? AND rows>0", (d.isoformat(),))}
    if "twse" not in done:
        n += _store(twse.fetch_twse_daily(d), "twse", d)
    if "tpex" not in done:
        n += _store(twse.fetch_tpex_daily(d), "tpex", d)
    return n


def backfill(days: int | None = None) -> None:
    if not _lock.acquire(blocking=False):
        return
    try:
        days = days or get_settings().backfill_days
        status.update(running=True, done=0, message="同步股票清單")
        try:
            sync_stock_list()
        except Exception as e:  # FinMind 掛掉時仍可用每日行情補名稱
            log.warning("stock list sync failed: %s", e)
        today = date.today()
        synced = {r["date"] for r in db.query(
            "SELECT date FROM sync_log GROUP BY date HAVING COUNT(*)>=2")}
        todo = [today - timedelta(days=i) for i in range(days, -1, -1)]
        todo = [d for d in todo if d.weekday() < 5 and d.isoformat() not in synced]
        if todo and todo[-1] == today and datetime.now().hour < 14:
            todo.pop()  # 今日收盤資料 14:00 後才完整
        status["total"] = len(todo)
        for i, d in enumerate(todo, 1):
            status["message"] = f"回補 {d.isoformat()}"
            try:
                sync_day(d)
            except Exception as e:
                log.warning("sync %s failed: %s", d, e)
            status["done"] = i
        status["message"] = "完成"
    finally:
        status["running"] = False
        _lock.release()


def daily_update() -> int:
    return sync_day(date.today())


def latest_trade_date() -> str | None:
    row = db.one("SELECT MAX(date) AS d FROM prices")
    return row["d"] if row else None

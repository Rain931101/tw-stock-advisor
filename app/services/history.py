"""長期歷史資料回補：日 K、三大法人、融資融券、除權息、加權指數，從最新往回補到 HISTORY_START。

- 由新到舊逐日抓：中途中斷也能用「最近連續的一段」先訓練，重新執行會從斷點繼續（已抓過的日期跳過）。
- 每個交易日：上市＋上櫃 日 K、三大法人、融資券，共 6 個請求；證交所／櫃買各自節流 3.5 秒，避免被封鎖 IP。
- 約 1,200 個交易日需 4～5 小時，在背景執行。
"""
import logging
import threading
from datetime import date, timedelta

import httpx

from app import db
from app.config import get_settings
from app.data import chipdata

log = logging.getLogger(__name__)
status = {"running": False, "done": 0, "total": 0, "message": "", "oldest": None}
_lock = threading.Lock()
_stop = threading.Event()


def coverage(max_age_min: float = 10) -> dict:
    """目前各資料的起訖日期與天數。

    全表統計在數百萬筆資料上要好幾秒，而且會佔住資料庫鎖、卡住其他頁面，所以快取 10 分鐘；回補中則每次重算。
    """
    if not status["running"]:
        hit = db.cache_get("history_coverage", max_age_hours=max_age_min / 60)
        if hit:
            return hit
    out = _coverage()
    db.cache_set("history_coverage", out)
    return out


def _coverage() -> dict:
    def rng(sql):
        r = db.one(sql)
        return {"from": r["a"], "to": r["b"], "days": r["n"]}
    chipdata.ensure_schema()
    return {
        "start_target": get_settings().history_start,
        "prices": rng("SELECT MIN(date) a, MAX(date) b, COUNT(DISTINCT date) n FROM prices"),
        "insti": rng("SELECT MIN(date) a, MAX(date) b, COUNT(*) n FROM ml_sync WHERE kind='insti'"),
        "margin": rng("SELECT MIN(date) a, MAX(date) b, COUNT(*) n FROM ml_sync WHERE kind='margin'"),
        "dividends": rng("SELECT MIN(date) a, MAX(date) b, COUNT(*) n FROM dividends"),
        "index": rng("SELECT MIN(date) a, MAX(date) b, COUNT(*) n FROM index_daily WHERE code='TAIEX'"),
    }


def stop() -> None:
    _stop.set()


def other_running() -> bool:
    """別的程序（例如指令列）正在回補：看資料庫心跳，2 分鐘內有更新就視為執行中，避免同時請求被證交所封鎖。"""
    hb = db.get_setting("history_heartbeat")
    if not hb:
        return False
    from datetime import datetime
    return (datetime.now() - datetime.fromisoformat(hb)).total_seconds() < 120


def backfill_history(start: str | None = None) -> dict:
    if other_running() and not status["running"]:
        return {"running": True, "message": "另一個程序正在回補"}
    if not _lock.acquire(blocking=False):
        return {"running": True}
    _stop.clear()
    try:
        from app.services import sync
        chipdata.ensure_schema()
        start_d = date.fromisoformat(start or get_settings().history_start)
        today = date.today()
        status.update(running=True, done=0, message="加權指數與除權息…")
        # 指數多抓 4 個月給 60MA 暖身
        months = (today.year - start_d.year) * 12 + today.month - start_d.month + 5
        chipdata.sync_index(months)
        cur = start_d
        while cur <= today:  # 除權息一次一季
            nxt = min(cur + timedelta(days=90), today)
            try:
                db.executemany("INSERT OR REPLACE INTO dividends VALUES (?,?,?,?,?)", chipdata.fetch_dividends(cur, nxt))
            except (httpx.HTTPError, ValueError) as e:
                log.warning("dividends %s: %s", cur, e)
            cur = nxt + timedelta(days=1)
        price_done = {r["date"] for r in db.query("SELECT date FROM sync_log GROUP BY date HAVING COUNT(*)>=2")}
        chip_done = {(r["kind"], r["date"]) for r in db.query("SELECT kind, date FROM ml_sync")}
        days = [today - timedelta(days=i) for i in range((today - start_d).days + 1)]  # 由新到舊
        days = [d for d in days if d.weekday() < 5]
        status["total"] = len(days)
        for d in days:
            if _stop.is_set():
                status["message"] = "已暫停"
                break
            status["done"] += 1
            iso = d.isoformat()
            db.set_setting("history_heartbeat", db.now())
            db.set_setting("history_progress", {"done": status["done"], "total": status["total"], "at": iso})
            try:
                if iso not in price_done:
                    status["message"] = f"日 K {iso}"
                    sync.sync_day(d)
                has_prices = db.one("SELECT 1 AS x FROM prices WHERE date=? LIMIT 1", (iso,)) is not None
                if not has_prices:
                    continue  # 假日：沒有交易，不抓籌碼
                for kind, fn, table in (("insti", chipdata.fetch_insti, "chips_daily"), ("margin", chipdata.fetch_margin, "margin_daily")):
                    if (kind, iso) in chip_done:
                        continue
                    status["message"] = f"{'三大法人' if kind == 'insti' else '融資融券'} {iso}"
                    rows = fn(d)
                    if rows:
                        db.executemany(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?,?,?)", rows)
                        db.execute("INSERT OR REPLACE INTO ml_sync VALUES (?,?,?)", (kind, iso, len(rows)))
                status["oldest"] = iso
            except (httpx.HTTPError, ValueError) as e:
                log.warning("history %s: %s", iso, e)
        else:
            status["message"] = "完成"
        return coverage()
    finally:
        db.set_setting("history_heartbeat", None)
        status["running"] = False
        _lock.release()

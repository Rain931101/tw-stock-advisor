"""機器學習用的全市場日資料（證交所／櫃買中心公開資料，每天一次請求、免額度）。

- 三大法人買賣超：TWSE T86、TPEx insti/dailyTrade
- 融資融券餘額：TWSE MI_MARGN、TPEx margin/balance
- 除權息（還原權值用）：TWSE TWT49U、TPEx bulletin/exDailyQ
- 加權／櫃買指數日 K：沿用 twse._twse_index_month / _tpex_index_month

主力分點（買超前 10 大分點）資料需要 FinMind 付費方案或證交所有驗證碼的買賣日報表，
本系統不抓取；若日後匯入 broker_daily 表，特徵會自動啟用。
"""
import logging
import threading
from datetime import date, datetime, timedelta

import httpx

from app import db
from app.data import twse
from app.data.twse import UA, _STOCK_RE, _num, _roc_or_ad, _throttle

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chips_daily (stock_id TEXT, date TEXT, foreign_net REAL, trust_net REAL, dealer_net REAL,
    total_net REAL, PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS margin_daily (stock_id TEXT, date TEXT, margin_bal REAL, margin_prev REAL, short_bal REAL,
    short_prev REAL, PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS dividends (stock_id TEXT, date TEXT, prev_close REAL, ref_price REAL, kind TEXT,
    PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS index_daily (code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
    PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS broker_daily (stock_id TEXT, date TEXT, top_buy REAL, top_sell REAL, PRIMARY KEY (stock_id, date));
CREATE TABLE IF NOT EXISTS ml_sync (kind TEXT, date TEXT, rows INTEGER, PRIMARY KEY (kind, date));
"""


def ensure_schema() -> None:
    with db._lock:
        db.conn().executescript(SCHEMA)


def _get(url: str, params: dict, host: str) -> dict:
    _throttle(host, 3.5)
    r = httpx.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def _lots(s) -> float:
    return (_num(s) or 0) / 1000  # 股 → 張


# ---------------- 三大法人 ----------------
def fetch_insti(d: date) -> list[tuple]:
    rows = []
    js = _get("https://www.twse.com.tw/rwd/zh/fund/T86", {"date": d.strftime("%Y%m%d"), "selectType": "ALLBUT0999",
                                                          "response": "json"}, "twse")
    if js.get("stat") == "OK":
        f = {k: i for i, k in enumerate(js["fields"])}
        for r in js.get("data", []):
            sid = r[0].strip()
            if not _STOCK_RE.match(sid):
                continue
            foreign = _lots(r[f["外陸資買賣超股數(不含外資自營商)"]]) + _lots(r[f["外資自營商買賣超股數"]])
            trust, dealer = _lots(r[f["投信買賣超股數"]]), _lots(r[f["自營商買賣超股數"]])
            rows.append((sid, d.isoformat(), foreign, trust, dealer, _lots(r[f["三大法人買賣超股數"]])))
    js = _get("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
              {"type": "Daily", "sect": "EW", "date": d.strftime("%Y/%m/%d"), "response": "json"}, "tpex")
    for t in js.get("tables", []):
        for r in t.get("data") or []:
            sid = r[0].strip()
            if not _STOCK_RE.match(sid) or len(r) < 24:
                continue
            # 欄位順序：外資(不含自營) 2-4、外資自營 5-7、外資合計 8-10、投信 11-13、自營(自行) 14-16、自營(避險) 17-19、自營合計 20-22、合計 23
            rows.append((sid, d.isoformat(), _lots(r[10]), _lots(r[13]), _lots(r[22]), _lots(r[23])))
    return rows


# ---------------- 融資融券 ----------------
def fetch_margin(d: date) -> list[tuple]:
    rows = []
    js = _get("https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
              {"date": d.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"}, "twse")
    for t in js.get("tables", []):
        if not t.get("fields") or t["fields"][0] != "代號":
            continue
        for r in t.get("data", []):
            sid = r[0].strip()
            if _STOCK_RE.match(sid):
                rows.append((sid, d.isoformat(), _num(r[6]), _num(r[5]), _num(r[12]), _num(r[11])))
    js = _get("https://www.tpex.org.tw/www/zh-tw/margin/balance", {"date": d.strftime("%Y/%m/%d"), "response": "json"}, "tpex")
    for t in js.get("tables", []):
        f = {k: i for i, k in enumerate(t.get("fields") or [])}
        if "資餘額" not in f:
            continue
        for r in t.get("data") or []:
            sid = r[0].strip()
            if _STOCK_RE.match(sid):
                rows.append((sid, d.isoformat(), _num(r[f["資餘額"]]), _num(r[f["前資餘額(張)"]]),
                             _num(r[f["券餘額"]]), _num(r[f["前券餘額(張)"]])))
    return rows


# ---------------- 除權息 ----------------
def fetch_dividends(start: date, end: date) -> list[tuple]:
    rows = []
    js = _get("https://www.twse.com.tw/rwd/zh/exRight/TWT49U",
              {"startDate": start.strftime("%Y%m%d"), "endDate": end.strftime("%Y%m%d"), "response": "json"}, "twse")
    if js.get("stat") == "OK":
        for r in js.get("data", []):
            day = _roc_or_ad(r[0].replace("年", "/").replace("月", "/").replace("日", ""))
            rows.append((r[1].strip(), day, _num(r[3]), _num(r[4]), r[6].strip()))
    js = _get("https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ",
              {"startDate": start.strftime("%Y/%m/%d"), "endDate": end.strftime("%Y/%m/%d"), "response": "json"}, "tpex")
    for t in js.get("tables", []):
        for r in t.get("data") or []:
            rows.append((r[1].strip(), _roc_or_ad(r[0]), _num(r[3]), _num(r[4]), r[8].strip()))
    return [r for r in rows if _STOCK_RE.match(r[0]) and r[2] and r[3]]


# ---------------- 指數 ----------------
def sync_index(months: int = 15) -> int:
    n = 0
    d = date.today().replace(day=1)
    for _ in range(months):
        for code, fn in (("TAIEX", twse._twse_index_month), ("TPEX", twse._tpex_index_month)):
            try:
                rows = [(code, r["date"], r["open"], r["high"], r["low"], r["close"]) for r in fn(d) if r["close"]]
            except (httpx.HTTPError, ValueError, KeyError) as e:
                log.warning("index %s %s: %s", code, d, e)
                continue
            db.executemany("INSERT OR REPLACE INTO index_daily VALUES (?,?,?,?,?,?)", rows)
            n += len(rows)
        d = (d - timedelta(days=1)).replace(day=1)
    return n


# ---------------- 回補 ----------------
status = {"running": False, "done": 0, "total": 0, "message": ""}
_lock = threading.Lock()


def _done(kind: str, day: str) -> bool:
    return db.one("SELECT 1 AS x FROM ml_sync WHERE kind=? AND date=?", (kind, day)) is not None


def backfill(days: int | None = None) -> dict:
    """依 prices 表的交易日回補法人、融資券；除權息與指數整段抓。已抓過的日期會跳過（可中斷續抓）。"""
    if not _lock.acquire(blocking=False):
        return {"running": True}
    try:
        ensure_schema()
        dates = [r["date"] for r in db.query("SELECT DISTINCT date FROM prices ORDER BY date")]
        if days:
            dates = dates[-days:]
        status.update(running=True, done=0, total=len(dates) * 2, message="指數與除權息…")
        sync_index()
        start, end = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
        cur = start
        while cur <= end:  # 除權息一次抓一季
            nxt = min(cur + timedelta(days=90), end)
            try:
                db.executemany("INSERT OR REPLACE INTO dividends VALUES (?,?,?,?,?)", fetch_dividends(cur, nxt))
            except (httpx.HTTPError, ValueError) as e:
                log.warning("dividends %s: %s", cur, e)
            cur = nxt + timedelta(days=1)
        for kind, fn, table, ncol in (("insti", fetch_insti, "chips_daily", 6), ("margin", fetch_margin, "margin_daily", 6)):
            for day in dates:
                status["done"] += 1
                if _done(kind, day):
                    continue
                status["message"] = f"{'三大法人' if kind == 'insti' else '融資融券'} {day}"
                try:
                    rows = fn(date.fromisoformat(day))
                except (httpx.HTTPError, ValueError) as e:
                    log.warning("%s %s: %s", kind, day, e)
                    continue
                if rows:
                    db.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?' * ncol)})", rows)
                    db.execute("INSERT OR REPLACE INTO ml_sync VALUES (?,?,?)", (kind, day, len(rows)))
        status["message"] = "完成"
        return {k: db.one(f"SELECT COUNT(*) AS n FROM {k}")["n"] for k in ("chips_daily", "margin_daily", "dividends", "index_daily")}
    finally:
        status["running"] = False
        _lock.release()


def sync_today() -> None:
    """收盤後每日增量（排程呼叫）。"""
    backfill(days=3)

"""臺指期（TX）日資料與夜盤即時報價（期交所公開資料）。

- 歷史：https://www.taifex.com.tw/cht/3/futDataDown（CSV，Big5；含「一般」與「盤後」交易時段）
  「盤後」列的交易日期 D = 前一天 15:00 到 D 當天 05:00 的夜盤，結束在 D 開盤之前 → 可用來預測 D 的大盤。
- 即時：https://mis.taifex.com.tw/futures/api/getQuoteList（MarketType 1 = 盤後／夜盤）
"""
import csv
import io
import logging
from datetime import date, datetime, timedelta

import httpx

from app import db
from app.data.twse import UA, _num, _throttle

log = logging.getLogger(__name__)

SCHEMA = """CREATE TABLE IF NOT EXISTS futures_daily (date TEXT, contract TEXT, session TEXT, open REAL, high REAL, low REAL,
    close REAL, change REAL, volume REAL, settle REAL, PRIMARY KEY (date, contract, session));"""


def ensure_schema() -> None:
    with db._lock:
        db.conn().executescript(SCHEMA)


def fetch_history(start: date, end: date) -> list[tuple]:
    _throttle("taifex", 3.5)
    r = httpx.post("https://www.taifex.com.tw/cht/3/futDataDown", headers=UA, timeout=60,
                   data={"down_type": "1", "commodity_id": "TX", "commodity_id2": "",
                         "queryStartDate": start.strftime("%Y/%m/%d"), "queryEndDate": end.strftime("%Y/%m/%d")})
    r.raise_for_status()
    text = r.content.decode("big5", errors="ignore")
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 18 or not row[0][:4].isdigit() or row[1].strip() != "TX":
            continue
        contract = row[2].strip()
        if "/" in contract:          # 價差單
            continue
        rows.append((row[0].replace("/", "-"), contract, row[17].strip(), _num(row[3]), _num(row[4]), _num(row[5]),
                     _num(row[6]), _num(row[7]), _num(row[9]), _num(row[10])))
    return [r for r in rows if r[6]]


def backfill(start: str = "2020-09-01") -> int:
    ensure_schema()
    have = db.one("SELECT MIN(date) a, MAX(date) b FROM futures_daily")
    cur = date.fromisoformat(start)
    today = date.today()
    n = 0
    while cur <= today:
        end = min((cur.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), today)
        covered = have["a"] and have["a"] <= cur.isoformat() and have["b"] >= end.isoformat()
        if not covered or end >= today - timedelta(days=7):
            try:
                rows = fetch_history(cur, end)
                db.executemany("INSERT OR REPLACE INTO futures_daily VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
                n += len(rows)
            except (httpx.HTTPError, ValueError) as e:
                log.warning("futures %s: %s", cur, e)
        cur = end + timedelta(days=1)
    return n


def sync_recent() -> int:
    ensure_schema()
    today = date.today()
    rows = fetch_history(today - timedelta(days=10), today)
    db.executemany("INSERT OR REPLACE INTO futures_daily VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def night_live() -> dict | None:
    """夜盤即時（近月合約）：{last, ref, high, low, ret%, time, contract}；夜盤沒有成交時回傳 None。"""
    _throttle("taifex_mis", 1.0)
    try:
        r = httpx.post("https://mis.taifex.com.tw/futures/api/getQuoteList", headers={**UA, "Content-Type": "application/json"},
                       json={"MarketType": "1", "SymbolType": "F", "KindID": "1", "CID": "TXF", "ExpireMonth": "",
                             "RowSize": "全部", "PageNo": "", "SortColumn": "", "AscDesc": "A"}, timeout=15)
        rows = r.json().get("RtData", {}).get("QuoteList", [])
    except (httpx.HTTPError, ValueError):
        return None
    futs = [x for x in rows if x.get("SymbolID", "").startswith("TXF") and x.get("SymbolID", "").endswith("-M") and _num(x.get("CLastPrice"))]
    if not futs:
        return None
    near = max(futs, key=lambda x: _num(x.get("CTotalVolume")) or 0)
    last, ref = _num(near["CLastPrice"]), _num(near["CRefPrice"])
    return {"contract": near["DispCName"], "last": last, "ref": ref, "high": _num(near.get("CHighPrice")),
            "low": _num(near.get("CLowPrice")), "ret": round((last / ref - 1) * 100, 2) if ref else None,
            "volume": _num(near.get("CTotalVolume")), "time": near.get("CTime"), "date": near.get("CDate"),
            "fetched_at": datetime.now().isoformat(timespec="seconds")}


# ======================= 三大法人期貨／選擇權未平倉（開高走低判斷用） =======================
INST_SCHEMA = """
CREATE TABLE IF NOT EXISTS fut_inst (date TEXT, who TEXT, long_oi REAL, short_oi REAL, net_oi REAL, PRIMARY KEY (date, who));
CREATE TABLE IF NOT EXISTS opt_inst (date TEXT, cp TEXT, who TEXT, buy_oi REAL, sell_oi REAL, PRIMARY KEY (date, cp, who));
"""
WHO = {"外資及陸資": "foreign", "投信": "trust", "自營商": "dealer"}


def _csv_post(url: str, data: dict) -> list[list[str]]:
    _throttle("taifex", 3.5)
    r = httpx.post(url, headers=UA, timeout=60, data=data)
    r.raise_for_status()
    return list(csv.reader(io.StringIO(r.content.decode("big5", errors="ignore"))))


def fetch_inst(start: date, end: date) -> tuple[list[tuple], list[tuple]]:
    q = {"firstDate": "2020/09/01", "queryStartDate": start.strftime("%Y/%m/%d"), "queryEndDate": end.strftime("%Y/%m/%d")}
    fut = []
    for row in _csv_post("https://www.taifex.com.tw/cht/3/futContractsDateDown", {**q, "commodityId": "TXF"}):
        if len(row) >= 15 and row[0][:4].isdigit() and row[2].strip() in WHO:
            fut.append((row[0].replace("/", "-"), WHO[row[2].strip()], _num(row[9]), _num(row[11]), _num(row[13])))
    opt = []
    for row in _csv_post("https://www.taifex.com.tw/cht/3/callsAndPutsDateDown", {**q, "commodityId": "TXO"}):
        if len(row) >= 16 and row[0][:4].isdigit() and row[3].strip() in WHO:
            opt.append((row[0].replace("/", "-"), row[2].strip().upper(), WHO[row[3].strip()], _num(row[10]), _num(row[12])))
    return fut, opt


def backfill_inst(start: str = "2020-09-01", recent_days: int | None = None) -> int:
    with db._lock:
        db.conn().executescript(INST_SCHEMA)
    today = date.today()
    cur = today - timedelta(days=recent_days) if recent_days else date.fromisoformat(start)
    have = {r["d"][:7] for r in db.query("SELECT DISTINCT substr(date,1,7) AS d FROM fut_inst")}
    n = 0
    while cur <= today:
        end = min((cur.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), today)
        if recent_days or cur.strftime("%Y-%m") not in have or end >= today - timedelta(days=7):
            try:
                fut, opt = fetch_inst(cur, end)
                if not fut and end >= today - timedelta(days=1):      # 迄日含尚未公布的今天 → 改抓到昨天
                    fut, opt = fetch_inst(cur, end - timedelta(days=1))
                db.executemany("INSERT OR REPLACE INTO fut_inst VALUES (?,?,?,?,?)", fut)
                db.executemany("INSERT OR REPLACE INTO opt_inst VALUES (?,?,?,?,?)", opt)
                n += len(fut)
            except (httpx.HTTPError, ValueError) as e:
                log.warning("taifex inst %s: %s", cur, e)
        cur = end + timedelta(days=1)
    return n


# ======================= 加權指數盤中走勢（每 5 秒，存成每分鐘） =======================
INTRA_SCHEMA = "CREATE TABLE IF NOT EXISTS index_intraday (date TEXT, hhmm TEXT, value REAL, PRIMARY KEY (date, hhmm));"


def fetch_index_intraday(d: date) -> list[tuple]:
    """證交所 MI_5MINS_INDEX：當天每 5 秒的加權指數，取每分鐘最後一筆（09:00:00 那筆是開盤前，略過）。"""
    _throttle("twse", 3.5)
    js = httpx.get("https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_INDEX", params={"date": d.strftime("%Y%m%d"), "response": "json"},
                   headers=UA, timeout=30).json()
    if js.get("stat") != "OK":
        return []
    last: dict[str, float] = {}
    for row in js.get("data", []):
        t, v = row[0], _num(row[1])
        if v is None or t <= "09:00:00":
            continue
        last[t[:5]] = v
    return [(d.isoformat(), k, v) for k, v in sorted(last.items())]


def backfill_intraday(days: list[str]) -> int:
    with db._lock:
        db.conn().executescript(INTRA_SCHEMA)
    have = {r["date"] for r in db.query("SELECT DISTINCT date FROM index_intraday")}
    n = 0
    for d in days:
        if d in have:
            continue
        try:
            rows = fetch_index_intraday(date.fromisoformat(d))
            db.executemany("INSERT OR REPLACE INTO index_intraday VALUES (?,?,?)", rows)
            n += len(rows)
        except (httpx.HTTPError, ValueError) as e:
            log.warning("intraday %s: %s", d, e)
    return n

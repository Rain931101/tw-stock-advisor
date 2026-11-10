"""美股日資料（Nasdaq 公開 API，免金鑰）：S&P 500（SPY）、那斯達克、費城半導體、台積電 ADR。

- 歷史：https://api.nasdaq.com/api/quote/{代號}/historical（約晚 1 個交易日才更新）
- 最新一天：https://api.nasdaq.com/api/quote/{代號}/info（收盤後的正規盤收盤價）

時間對照：美股日期 d 的交易在台灣時間 d+1 清晨 4:00（夏令）或 5:00（冬令）收盤，
所以台股交易日 D 開盤前已知的美股，是「日期 < D」的最近一個交易日。
"""
import logging
import re
from datetime import date, datetime, timedelta

import httpx

from app import db
from app.data.twse import _num, _throttle

log = logging.getLogger(__name__)

SYMBOLS = {  # 代號: (assetclass, 中文名)
    "SPY": ("etf", "S&P 500"),
    "COMP": ("index", "那斯達克"),
    "SOX": ("index", "費城半導體"),
    "TSM": ("stocks", "台積電 ADR"),
}
SCHEMA = """CREATE TABLE IF NOT EXISTS us_daily (date TEXT, symbol TEXT, open REAL, high REAL, low REAL, close REAL,
    PRIMARY KEY (date, symbol));"""
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
       "Accept": "application/json"}
BASE = "https://api.nasdaq.com/api/quote"


def ensure_schema() -> None:
    with db._lock:
        db.conn().executescript(SCHEMA)


def _usd(s) -> float | None:
    return _num(str(s).replace("$", "")) if s is not None else None


def _get(url: str, params: dict | None = None) -> dict:
    _throttle("nasdaq", 1.5)
    r = httpx.get(url, params=params, headers=HDR, timeout=30)
    r.raise_for_status()
    return r.json().get("data") or {}


def fetch_history(symbol: str, start: date, end: date) -> list[tuple]:
    ac = SYMBOLS[symbol][0]
    d = _get(f"{BASE}/{symbol}/historical", {"assetclass": ac, "fromdate": start.isoformat(),
                                             "todate": end.isoformat(), "limit": 9999})
    rows = (d.get("tradesTable") or {}).get("rows") or []
    out = []
    for x in rows:
        m, dd, y = x["date"].split("/")
        c = _usd(x.get("close"))
        if c:
            out.append((f"{y}-{m}-{dd}", symbol, _usd(x.get("open")), _usd(x.get("high")), _usd(x.get("low")), c))
    return out


def fetch_latest(symbol: str) -> tuple | None:
    """最近一個已收盤的正規盤（盤後交易時段時 primaryData 是盤後價，正規盤收盤在 secondaryData）。"""
    ac = SYMBOLS[symbol][0]
    d = _get(f"{BASE}/{symbol}/info", {"assetclass": ac})
    status = (d.get("marketStatus") or "").lower().strip()
    # 盤前／盤後時段 primaryData 是延長交易的價格，正規盤收盤在 secondaryData（"Closed at ..."）
    pick = d.get("secondaryData") or (d.get("primaryData") if status in ("closed", "") else None)
    if not pick:                                       # 盤中：還沒收盤，不存
        return None
    ts = pick.get("lastTradeTimestamp") or ""
    m = re.search(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", ts)
    c = _usd(pick.get("lastSalePrice"))
    if not m or not c:
        return None
    day = datetime.strptime(" ".join(m.groups()), "%b %d %Y").date()
    hi = lo = None
    if ac == "index":
        try:
            s = _get(f"{BASE}/{symbol}/summary", {"assetclass": ac}).get("summaryData") or {}
            hi, lo = _num((s.get("TodaysHigh") or {}).get("value")), _num((s.get("TodaysLow") or {}).get("value"))
        except (httpx.HTTPError, ValueError):
            pass
    return (day.isoformat(), symbol, None, hi, lo, c)


def backfill(start: str = "2020-09-01") -> int:
    ensure_schema()
    n = 0
    for sym in SYMBOLS:
        try:
            rows = fetch_history(sym, date.fromisoformat(start), date.today())
            db.executemany("INSERT OR REPLACE INTO us_daily VALUES (?,?,?,?,?,?)", rows)
            n += len(rows)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("us history %s: %s", sym, e)
    return n + sync_latest()


def sync_latest() -> int:
    """抓最近 10 天歷史＋最新一天收盤（歷史 API 晚一天，用 info 補上）。"""
    ensure_schema()
    n = 0
    for sym in SYMBOLS:
        try:
            rows = fetch_history(sym, date.today() - timedelta(days=10), date.today())
            last = fetch_latest(sym)
            if last and last[0] not in {r[0] for r in rows}:
                rows.append(last)
            db.executemany("INSERT OR REPLACE INTO us_daily VALUES (?,?,?,?,?,?)", rows)
            n += len(rows)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("us latest %s: %s", sym, e)
    return n


def latest_date() -> str | None:
    if not db.one("SELECT name FROM sqlite_master WHERE name='us_daily'"):
        return None
    return db.one("SELECT MAX(date) d FROM us_daily")["d"]

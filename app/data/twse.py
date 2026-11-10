"""證交所 / 櫃買中心公開資料：每日全市場收盤行情、盤中即時報價。

- 上市每日行情：https://www.twse.com.tw/exchangeReport/MI_INDEX?type=ALLBUT0999
- 上櫃每日行情：https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes
- 盤中即時（約 5 秒更新）：https://mis.twse.com.tw/stock/api/getStockInfo.jsp
證交所對高頻請求會暫時封鎖 IP，因此所有請求都經過節流。
"""
import re
import threading
import time
from datetime import date, datetime, timedelta

import httpx

UA = {"User-Agent": "Mozilla/5.0 (tw-stock-advisor)"}
_throttle_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _throttle(host: str, min_interval: float) -> None:
    with _throttle_lock:
        wait = _last_call.get(host, 0) + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.time()


def _num(s) -> float | None:
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if s in ("", "--", "---", "-", "除權息", "除息", "除權"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


_STOCK_RE = re.compile(r"^\d{4}$")  # 只收一般普通股（4 碼），排除 ETF、權證、債券


def fetch_twse_daily(d: date) -> list[tuple]:
    _throttle("twse", 3.5)
    r = httpx.get(
        "https://www.twse.com.tw/exchangeReport/MI_INDEX",
        params={"response": "json", "date": d.strftime("%Y%m%d"), "type": "ALLBUT0999"},
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("stat") != "OK":
        return []
    table = next((t for t in js.get("tables", []) if t.get("fields") and t["fields"][0] == "證券代號"), None)
    if not table:
        return []
    f = {k: i for i, k in enumerate(table["fields"])}
    rows = []
    for row in table["data"]:
        sid = row[f["證券代號"]].strip()
        if not _STOCK_RE.match(sid):
            continue
        o, h, l, c = (_num(row[f[k]]) for k in ("開盤價", "最高價", "最低價", "收盤價"))
        if c is None:
            continue
        vol = (_num(row[f["成交股數"]]) or 0) / 1000  # 張
        val = _num(row[f["成交金額"]]) or 0
        rows.append((sid, d.isoformat(), o or c, h or c, l or c, c, vol, val, row[f["證券名稱"]].strip()))
    return rows


def fetch_tpex_daily(d: date) -> list[tuple]:
    _throttle("tpex", 3.5)
    r = httpx.get(
        "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
        params={"date": d.strftime("%Y/%m/%d"), "type": "EW", "response": "json"},
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    js = r.json()
    tables = js.get("tables") or []
    if not tables or not tables[0].get("data"):
        return []
    t = tables[0]
    f = {k.replace(" ", ""): i for i, k in enumerate(t["fields"])}
    rows = []
    for row in t["data"]:
        sid = row[f["代號"]].strip()
        if not _STOCK_RE.match(sid):
            continue
        o, h, l, c = (_num(row[f[k]]) for k in ("開盤", "最高", "最低", "收盤"))
        if c is None:
            continue
        vol = (_num(row[f["成交股數"]]) or 0) / 1000
        val = _num(row[f["成交金額(元)"]]) or 0
        rows.append((sid, d.isoformat(), o or c, h or c, l or c, c, vol, val, row[f["名稱"]].strip()))
    return rows


# ---------------- 盤中即時 ----------------
def _channel(stock_id: str, market: str) -> str:
    return f"{'otc' if market == 'tpex' else 'tse'}_{stock_id}.tw"


def fetch_realtime(items: list[tuple[str, str]]) -> dict[str, dict]:
    """items: [(stock_id, market)]；market 為 twse / tpex / index。

    回傳 {stock_id: {price, open, high, low, prev_close, volume(張), time, date, name}}
    指數代號：t00（加權）、o00（櫃買）。
    """
    out: dict[str, dict] = {}
    for i in range(0, len(items), 50):
        chunk = items[i : i + 50]
        chans = []
        for sid, mkt in chunk:
            if sid == "t00":
                chans.append("tse_t00.tw")
            elif sid == "o00":
                chans.append("otc_o00.tw")
            else:
                chans.append(_channel(sid, mkt))
        _throttle("mis", 1.0)
        try:
            r = httpx.get(
                "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
                params={"ex_ch": "|".join(chans), "json": "1", "delay": "0", "_": int(time.time() * 1000)},
                headers=UA,
                timeout=15,
            )
            msgs = r.json().get("msgArray", [])
        except Exception:
            continue
        for m in msgs:
            sid = m.get("c")
            price = _num(m.get("z"))
            if price is None:  # 該快照無成交時退回最近一筆成交或最佳買價
                price = _num((m.get("trade") or {}).get("z"))
            if price is None:
                b = (m.get("b") or "").split("_")[0]
                price = _num(b)
            if price is None:
                price = _num(m.get("y"))
            out[sid] = {
                "price": price,
                "open": _num(m.get("o")),
                "high": _num(m.get("h")),
                "low": _num(m.get("l")),
                "prev_close": _num(m.get("y")),
                "volume": _num(m.get("v")) or 0,
                "limit_up": _num(m.get("u")),
                "limit_down": _num(m.get("w")),
                "time": m.get("t") or m.get("%"),
                "date": m.get("d"),
                "name": m.get("n"),
            }
    return out


def _roc_or_ad(s: str) -> str:
    """'115/11/06' 或 '2026/11/06' → '2026-11-06'"""
    y, m, d = s.strip().split("/")
    y = int(y) + 1911 if int(y) < 1911 else int(y)
    return f"{y:04d}-{int(m):02d}-{int(d):02d}"


def _twse_index_month(d: date) -> list[dict]:
    _throttle("twse", 3.5)
    js = httpx.get("https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
                   params={"response": "json", "date": d.strftime("%Y%m01")}, headers=UA, timeout=20).json()
    if js.get("stat") != "OK":
        return []
    return [{"date": _roc_or_ad(r[0]), "open": _num(r[1]), "high": _num(r[2]), "low": _num(r[3]), "close": _num(r[4])}
            for r in js.get("data", [])]


def _tpex_index_month(d: date) -> list[dict]:
    _throttle("tpex", 3.5)
    js = httpx.get("https://www.tpex.org.tw/www/zh-tw/indexInfo/inx",
                   params={"date": d.strftime("%Y/%m/01"), "response": "json"}, headers=UA, timeout=20).json()
    rows = []
    for t in js.get("tables", []):
        f = {k: i for i, k in enumerate(t.get("fields", []))}
        for r in t.get("data", []):
            rows.append({"date": _roc_or_ad(r[f["日期"]]), "open": _num(r[f["開市"]]), "high": _num(r[f["最高"]]),
                         "low": _num(r[f["最低"]]), "close": _num(r[f["收市"]])})
    return rows


def fetch_official_index_ohlc(d: date) -> dict[str, dict]:
    """官方每日指數 OHLC（取 d 當天或之前最近一個交易日），含前一交易日收盤。

    MIS 即時看板約 13:20 後停止更新、晚上還會重置，收盤後與開盤前一律以官方資料為準。
    回傳 {"t00": {date, open, high, low, close, prev_close}, "o00": {...}}
    """
    out: dict[str, dict] = {}
    prev_month = (d.replace(day=1) - timedelta(days=1))
    for key, fn in (("t00", _twse_index_month), ("o00", _tpex_index_month)):
        try:
            rows = [r for r in fn(d) if r["date"] <= d.isoformat() and r["close"]]
            if len(rows) < 2:
                rows = [r for r in fn(prev_month) if r["close"]] + rows
            if len(rows) >= 2:
                out[key] = {**rows[-1], "prev_close": rows[-2]["close"]}
        except (httpx.HTTPError, ValueError, KeyError):
            continue
    return out


def market_is_open_today() -> bool:
    """以加權指數即時資料的日期判斷今天是否為交易日（自動處理國定假日與颱風假）。"""
    # MIS 晚上會重置成「今天日期、沒有最高最低」的空快照，因此必須真的有成交（有最高價）才算開市
    q = fetch_realtime([("t00", "index")]).get("t00")
    return bool(q and q.get("date") == datetime.now().strftime("%Y%m%d") and q.get("high"))


def fetch_valuation_all() -> dict[str, dict]:
    """全市場本益比、股價淨值比、殖利率（證交所 BWIBBU_ALL ＋ 櫃買 peratio），一次兩個請求。"""
    out: dict[str, dict] = {}
    try:
        _throttle("twse", 3.5)
        for r in httpx.get("https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL", headers=UA, timeout=30).json():
            out[r["Code"]] = {"PER": _num(r.get("PEratio")), "PBR": _num(r.get("PBratio")),
                              "dividend_yield": _num(r.get("DividendYield"))}
    except (httpx.HTTPError, ValueError):
        pass
    try:
        _throttle("tpex", 3.5)
        for r in httpx.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis", headers=UA, timeout=30).json():
            out[r["SecuritiesCompanyCode"]] = {"PER": _num(r.get("PriceEarningRatio")), "PBR": _num(r.get("PriceBookRatio")),
                                               "dividend_yield": _num(r.get("YieldRatio"))}
    except (httpx.HTTPError, ValueError):
        pass
    return out

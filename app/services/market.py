"""大盤總覽與判斷（技術分析講義第七章〈判斷大盤〉）。

1. 法人選擇權：BP 增加 > BC 增加 → 隔日偏弱；BC 增加 > BP 增加 → 隔日偏強
2. 正價差擴大 + 外資期貨空單增加 → 隔日開高容易走低
3. 權值 vs 櫃買：權值動櫃買休息、櫃買動權值休息
4. 強勢族群：要帶動盤勢市值一定要夠大；資金撤出要看其他族群有沒有接
5. 台幣：升值台股強、貶值台股弱（升貶的當下影響最深）
"""
from datetime import date, datetime, timedelta

import pandas as pd

from app import db
from app.data import finmind, twse


def _pct(a, b):
    return None if not a or not b else round((a - b) / b * 100, 2)


def official_ohlc(d: date) -> dict:
    """當天已公布就永久快取；否則 10 分鐘內不重抓。"""
    key = f"index_ohlc:{d.isoformat()}"
    hit = db.cache_get(key)
    if hit and all(v.get("date") == d.isoformat() for v in hit.values()) and len(hit) == 2:
        return hit
    hit = db.cache_get(key, 0.15)
    if hit is not None:
        return hit
    data = twse.fetch_official_index_ohlc(d)
    db.cache_set(key, data)
    return data


def indices(now: datetime | None = None) -> dict:
    """盤中（9:00~13:30）用 MIS 即時；其餘時間用官方每日 OHLC。"""
    now = now or datetime.now()
    t = now.time()
    in_session = now.weekday() < 5 and datetime.strptime("09:00", "%H:%M").time() <= t < datetime.strptime("13:30", "%H:%M").time()
    names = {"t00": "加權指數", "o00": "櫃買指數"}
    out = {}
    q = {}
    if in_session:
        q = twse.fetch_realtime([("t00", "index"), ("o00", "index")])
        q = {k: v for k, v in q.items() if v.get("date") == now.strftime("%Y%m%d") and v.get("high")}
    official = {} if len(q) == 2 else official_ohlc(now.date())
    for k, name in names.items():
        v = q.get(k)
        if v:
            out[k] = {"name": name, "price": v["price"], "prev_close": v["prev_close"],
                      "change_pct": _pct(v["price"], v["prev_close"]), "time": f"盤中 {v['time']}",
                      "open": v["open"], "high": v["high"], "low": v["low"], "source": "realtime"}
        elif official.get(k):
            o = official[k]
            label = "收盤" if o["date"] == now.date().isoformat() else f"{o['date'][5:].replace('-', '/')} 收盤"
            out[k] = {"name": name, "price": o["close"], "prev_close": o["prev_close"],
                      "change_pct": _pct(o["close"], o["prev_close"]), "time": label,
                      "open": o["open"], "high": o["high"], "low": o["low"], "source": "official"}
    return out


def _start(days=10):
    return (date.today() - timedelta(days=days)).isoformat()


def institutional_spot() -> dict | None:
    rows = finmind.fetch("TaiwanStockTotalInstitutionalInvestors", None, _start(), cache_hours=1)
    if not rows:
        return None
    last = max(r["date"] for r in rows)
    agg = {"foreign": 0.0, "trust": 0.0, "dealer": 0.0}
    for r in rows:
        if r["date"] != last or r["name"] == "total":
            continue
        k = "foreign" if r["name"].startswith("Foreign") else "trust" if r["name"] == "Investment_Trust" else "dealer"
        agg[k] += (r["buy"] - r["sell"]) / 1e8
    return {"date": last, **{k: round(v, 1) for k, v in agg.items()}}


def futures_foreign() -> dict | None:
    rows = finmind.fetch("TaiwanFuturesInstitutionalInvestors", "TX", _start(), cache_hours=1)
    rows = sorted([r for r in rows if r["institutional_investors"] == "外資"], key=lambda r: r["date"])
    if len(rows) < 2:
        return None
    net = lambda r: r["long_open_interest_balance_volume"] - r["short_open_interest_balance_volume"]  # noqa: E731
    a, b = rows[-1], rows[-2]
    return {"date": a["date"], "net_oi": net(a), "change": net(a) - net(b),
            "short_oi": a["short_open_interest_balance_volume"],
            "short_change": a["short_open_interest_balance_volume"] - b["short_open_interest_balance_volume"]}


def options_foreign() -> dict | None:
    rows = finmind.fetch("TaiwanOptionInstitutionalInvestors", "TXO", _start(), cache_hours=1)
    rows = [r for r in rows if r["institutional_investors"] == "外資"]
    dates = sorted({r["date"] for r in rows})
    if len(dates) < 2:
        return None

    def get(d, cp):
        r = next((x for x in rows if x["date"] == d and x["call_put"] == cp), None)
        return r["long_open_interest_balance_volume"] if r else 0

    d1, d0 = dates[-1], dates[-2]
    bc = get(d1, "買權") - get(d0, "買權")
    bp = get(d1, "賣權") - get(d0, "賣權")
    weekday = datetime.fromisoformat(d1).weekday()
    if bp > bc:
        view, tone = "BP 增加 > BC 增加：隔日盤勢偏弱（收黑或衝高回落）", "bear"
    elif bc > bp:
        view, tone = "BC 增加 > BP 增加：隔日盤勢偏強", "bull"
    else:
        view, tone = "買權賣權變化相當", "neutral"
    note = "（週三、五結算日未平倉自然減少，隔日判斷參考性較低）" if weekday in (2, 4) else ""
    return {"date": d1, "bc_change": bc, "bp_change": bp, "view": view + note, "tone": tone}


def basis() -> dict | None:
    """正價差 = 台指期近月收盤 − 加權收盤（未處理除權息蒸發點數，除息旺季請自行留意）。"""
    rows = finmind.fetch("TaiwanFuturesDaily", "TX", _start(), cache_hours=1)
    rows = [r for r in rows if r.get("trading_session") == "position" and r["close"]]
    if not rows:
        return None
    by_date: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r["contract_date"]):
        if "/" in r["contract_date"]:
            continue
        by_date.setdefault(r["date"], r)  # 最近月
    dates = sorted(by_date)
    if len(dates) < 2:
        return None
    idx = _taiex_closes()
    out = []
    for d in dates[-2:]:
        if d in idx:
            out.append(by_date[d]["close"] - idx[d])
    if len(out) < 2:
        return {"date": dates[-1], "basis": None}
    return {"date": dates[-1], "basis": round(out[-1], 1), "prev": round(out[-2], 1), "widening": out[-1] > out[-2] and out[-1] > 0}


def _taiex_closes() -> dict[str, float]:
    rows = finmind.fetch("TaiwanStockPrice", "TAIEX", _start(), cache_hours=1)
    return {r["date"]: r["close"] for r in rows}


def fx() -> dict | None:
    rows = finmind.fetch("TaiwanExchangeRate", "USD", _start(), cache_hours=1)
    rows = sorted([r for r in rows if r["spot_sell"] and r["spot_sell"] > 0], key=lambda r: r["date"])
    if len(rows) < 2:
        return None
    a, b = rows[-1]["spot_sell"], rows[-2]["spot_sell"]
    chg = a - b
    view = "台幣升值：台股偏強" if chg < -0.03 else "台幣貶值：台股偏弱" if chg > 0.03 else "匯率持穩"
    return {"date": rows[-1]["date"], "usd_twd": a, "change": round(chg, 3), "view": view}


def sector_strength(top: int = 8) -> list[dict]:
    """以最新交易日各產業「成交金額加權漲跌幅」與資金占比排強勢族群。"""
    dates = db.query("SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT 6")
    if len(dates) < 2:
        return []
    d1, d0, d5 = dates[0]["date"], dates[1]["date"], dates[-1]["date"]
    rows = db.query(
        """SELECT p.stock_id, s.industry, p.close, p.value, q.close AS prev, w.close AS prev5
           FROM prices p JOIN stocks s USING(stock_id)
           JOIN prices q ON q.stock_id=p.stock_id AND q.date=?
           LEFT JOIN prices w ON w.stock_id=p.stock_id AND w.date=?
           WHERE p.date=?""",
        (d0, d5, d1),
    )
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["chg"] = (df["close"] / df["prev"] - 1) * 100
    df["chg5"] = (df["close"] / df["prev5"] - 1) * 100
    total = df["value"].sum()
    out = []
    for indname, g in df.groupby("industry"):
        if g["value"].sum() < total * 0.005 or len(g) < 3:
            continue
        w = g["value"] / g["value"].sum()
        leader = g.sort_values("value", ascending=False).iloc[0]
        out.append({
            "industry": indname,
            "change_pct": round(float((g["chg"] * w).sum()), 2),
            "change_5d": round(float((g["chg5"].fillna(0) * w).sum()), 2),
            "value_share": round(float(g["value"].sum() / total * 100), 1),
            "breadth": round(float((g["chg"] > 0).mean() * 100)),
            "leader": leader["stock_id"],
        })
    out.sort(key=lambda x: x["change_pct"] * (1 + x["value_share"] / 10), reverse=True)
    return out[:top]


def top_value(n: int = 30) -> list[dict]:
    """講義〈無展覽的時間〉第一步：列出每日成交排行前 30，抓出龍頭帶動的主題。"""
    d = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    prev = db.one("SELECT MAX(date) AS d FROM prices WHERE date<?", (d,))["d"]
    return db.query(
        """SELECT p.stock_id, s.name, s.industry, p.close, p.value/1e8 AS value_e,
           ROUND((p.close/q.close-1)*100,2) AS change_pct
           FROM prices p JOIN stocks s USING(stock_id) LEFT JOIN prices q ON q.stock_id=p.stock_id AND q.date=?
           WHERE p.date=? ORDER BY p.value DESC LIMIT ?""",
        (prev, d, n),
    )


def overview() -> dict:
    out: dict = {"updated": db.now(), "errors": []}

    def safe(key, fn):
        try:
            out[key] = fn()
        except Exception as e:  # 個別資料源失敗不影響整頁
            out[key] = None
            out["errors"].append(f"{key}: {e}")

    safe("indices", indices)
    safe("spot", institutional_spot)
    safe("futures", futures_foreign)
    safe("options", options_foreign)
    safe("basis", basis)
    safe("fx", fx)
    safe("sectors", sector_strength)
    safe("top_value", top_value)

    # ---- 綜合判斷 ----
    signals, score = [], 0
    o = out.get("options")
    if o:
        signals.append(o["view"])
        score += {"bull": 1, "bear": -1}.get(o["tone"], 0)
    b, f = out.get("basis"), out.get("futures")
    if b and f and b.get("widening") and f["short_change"] > 0:
        signals.append("正價差擴大＋外資期貨空單增加：小心隔日開高走低（加權開高 1~1.5% 容易回平盤）")
        score -= 1
    if f:
        signals.append(f"外資台指期淨未平倉 {f['net_oi']:,} 口（{f['change']:+,}）——重點是變化，不是預設空單多寡")
    s = out.get("spot")
    if s and f:
        if s["foreign"] > 0 and f["change"] > 0:
            signals.append("外資現貨買超＋期貨多單增加：強勢")
            score += 1
        elif s["foreign"] < 0 and f["change"] < 0:
            signals.append("外資現貨賣超＋期貨空單增加：弱勢")
            score -= 1
    x = out.get("fx")
    if x:
        signals.append(x["view"])
        score += 1 if x["change"] < -0.03 else -1 if x["change"] > 0.03 else 0
    idx = out.get("indices") or {}
    t, otc = idx.get("t00"), idx.get("o00")
    if t and otc and t["change_pct"] is not None and otc["change_pct"] is not None:
        if t["change_pct"] - otc["change_pct"] > 0.8 and t["change_pct"] > 0:
            signals.append("權值動、櫃買休息")
        elif otc["change_pct"] - t["change_pct"] > 0.8 and otc["change_pct"] > 0:
            signals.append("櫃買動、權值休息")
    secs = out.get("sectors") or []
    if secs:
        signals.append("強勢族群：" + "、".join(x["industry"] for x in secs[:3]) + "——多頭前提下拿著強勢族群最省力")
    out["verdict"] = {
        "score": score,
        "stance": "進攻" if score >= 2 else "防守" if score <= -2 else "中性",
        "signals": signals,
        "note": "臺指選擇權波動率指數（台版 VIX）> 40 代表過度恐慌，可在此時加大 DCA 部位；請至期交所網站查看。",
    }
    return out

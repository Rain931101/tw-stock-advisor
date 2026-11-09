"""開高走低判斷（技術分析講義〈判斷大盤〉）＋ 歷史驗證。

講義的盤前規則：
  A. 正價差擴大 ＋ 外資台指期空單增加 → 隔日開高容易走低（加權開高 1~1.5% 容易回平盤）
  B. 外資選擇權 BP（買賣權）增加 > BC（買買權）增加 → 隔日偏弱（收黑或衝高回落）
盤中確認（開盤後）：
  開高後跌破開盤價 → 開高走低的機率大增（以 2020/9 起每個開高日的盤中走勢驗證）

每條規則都附上「歷史上真的發生的比例」，與所有開高日的基準比較，讓使用者知道該信多少。
"""
import logging
from datetime import datetime, time

import numpy as np
import pandas as pd

from app import db

log = logging.getLogger(__name__)
GAP = 0.5          # 開高定義：開盤 ≥ 前一日收盤 +0.5%
FLAT = 0.3         # 回平盤：收盤 ≤ 前一日收盤 +0.3%
CHECKS = ["09:05", "09:10", "09:15", "09:30", "10:00"]


def _daily() -> pd.DataFrame:
    idx = pd.DataFrame(db.query("SELECT date, open, high, low, close FROM index_daily WHERE code='TAIEX' ORDER BY date"))
    idx = idx.set_index("date").astype(float)
    idx["prev"] = idx["close"].shift()
    idx["gap"] = (idx["open"] / idx["prev"] - 1) * 100
    idx["fade"] = idx["close"] < idx["open"]
    idx["flat"] = (idx["close"] / idx["prev"] - 1) * 100 <= FLAT
    idx["oc"] = (idx["close"] / idx["open"] - 1) * 100
    return idx


def _premarket() -> pd.DataFrame:
    """每個交易日 t 收盤後的講義訊號（用於預測 t+1）。"""
    has = db.one("SELECT name FROM sqlite_master WHERE name='fut_inst'")
    if not has:
        return pd.DataFrame()
    idx = _daily()
    fd = pd.DataFrame(db.query("SELECT date, close, volume FROM futures_daily WHERE session='一般'"))
    near = fd.sort_values("volume").groupby("date").tail(1).set_index("date")["close"]
    fi = pd.DataFrame(db.query("SELECT date, short_oi FROM fut_inst WHERE who='foreign'")).set_index("date")
    oi = pd.DataFrame(db.query("SELECT date, cp, buy_oi FROM opt_inst WHERE who='foreign'")).pivot(index="date", columns="cp", values="buy_oi")
    d = idx.join(near.rename("fut")).join(fi).join(oi)
    d = d[d["short_oi"].notna()].copy()
    d["basis"] = d["fut"] - d["close"]
    d["basis_prev"] = d["basis"].shift()
    d["short_chg"] = d["short_oi"].diff()
    d["bc_chg"], d["bp_chg"] = d["CALL"].diff(), d["PUT"].diff()
    d["A"] = (d["basis"] > d["basis_prev"]) & (d["basis"] > 0) & (d["short_chg"] > 0)
    d["B"] = d["bp_chg"] > d["bc_chg"]
    nxt = idx.shift(-1)
    d["n_gap"], d["n_fade"], d["n_flat"], d["n_oc"] = nxt["gap"], nxt["fade"], nxt["flat"], nxt["oc"]
    return d


def _rate(s: pd.Series) -> float | None:
    return round(float(s.mean()), 4) if len(s) else None


def stats() -> dict:
    """各規則的歷史命中率（每天快取）。"""
    last = db.one("SELECT MAX(date) d FROM index_daily")["d"]
    key = f"gapfade_stats:{last}"
    hit = db.cache_get(key)
    if hit:
        return hit
    out = {"gap": GAP, "flat": FLAT}
    d = _premarket().dropna(subset=["n_gap"])
    if len(d):
        g = d[d["n_gap"] >= GAP]
        out["premarket"] = {
            "period": [d.index[0], d.index[-1]], "gap_days": int(len(g)),
            "base": {"fade": _rate(g["n_fade"]), "flat": _rate(g["n_flat"]), "oc": round(float(g["n_oc"].mean()), 2)},
            "A": {"n": int(g["A"].sum()), "fade": _rate(g.loc[g["A"], "n_fade"]), "flat": _rate(g.loc[g["A"], "n_flat"])},
            "B": {"n": int(g["B"].sum()), "fade": _rate(g.loc[g["B"], "n_fade"]), "flat": _rate(g.loc[g["B"], "n_flat"])},
        }
    # 盤中：開高日，各檢查時間點指數已跌破開盤價 → 收盤低於開盤／回平盤的比例
    if db.one("SELECT name FROM sqlite_master WHERE name='index_intraday'"):
        idx = _daily()
        intra = pd.DataFrame(db.query("SELECT date, hhmm, value FROM index_intraday"))
        if len(intra):
            piv = intra.pivot(index="date", columns="hhmm", values="value")
            days = idx.loc[(idx["gap"] >= GAP) & idx.index.isin(piv.index)]
            rows = {}
            for t in CHECKS:
                col = piv.columns[piv.columns <= t]
                if not len(col):
                    continue
                now_v = piv.loc[days.index, col[-1]]
                below = now_v < days["open"]
                rows[t] = {"n_below": int(below.sum()), "fade_below": _rate(days.loc[below, "fade"]),
                           "flat_below": _rate(days.loc[below, "flat"]),
                           "n_above": int((~below).sum()), "fade_above": _rate(days.loc[~below, "fade"]),
                           "flat_above": _rate(days.loc[~below, "flat"])}
            out["intraday"] = {"days": int(len(days)), "base_fade": _rate(days["fade"]), "base_flat": _rate(days["flat"]),
                               "period": [days.index[0], days.index[-1]] if len(days) else None, "checks": rows}
    db.cache_set(key, out)
    return out


def premarket_today() -> dict | None:
    """今天收盤後的講義盤前訊號（給明日預測卡）。"""
    d = _premarket()
    if not len(d):
        return None
    r = d.iloc[-1]
    st = stats().get("premarket") or {}
    return {"date": d.index[-1], "A": bool(r["A"]), "B": bool(r["B"]),
            "basis": None if pd.isna(r["basis"]) else round(float(r["basis"]), 1),
            "basis_prev": None if pd.isna(r["basis_prev"]) else round(float(r["basis_prev"]), 1),
            "short_chg": None if pd.isna(r["short_chg"]) else int(r["short_chg"]),
            "bc_chg": None if pd.isna(r["bc_chg"]) else int(r["bc_chg"]),
            "bp_chg": None if pd.isna(r["bp_chg"]) else int(r["bp_chg"]), "stats": st}


def intraday_check(quote: dict, now: datetime) -> dict | None:
    """盤中（9:05～10:00）：加權開高 ≥0.5% 且目前已跌破開盤價 → 回傳警示與歷史機率。"""
    o, prev, px = quote.get("open"), quote.get("prev_close"), quote.get("price")
    if not (o and prev and px) or not (time(9, 5) <= now.time() <= time(10, 5)):
        return None
    gap = (o / prev - 1) * 100
    if gap < GAP or px >= o:
        return None
    st = (stats().get("intraday") or {}).get("checks", {})
    t = max([k for k in CHECKS if k <= now.strftime("%H:%M")], default=None)
    s = st.get(t) if t else None
    return {"gap": round(gap, 2), "open": o, "price": px, "prev_close": prev, "drop_from_open": round((px / o - 1) * 100, 2),
            "check": t, "stats": s, "premarket": premarket_today()}

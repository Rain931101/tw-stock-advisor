"""條件組合勝率：用歷史資料「訓練」哪些條件同時成立時，交易的勝率最高。

交易定義（與系統給的操作計畫一致）：
  訊號日收盤後確認 → 隔日開盤買進 → 最多持有 HOLD 個交易日
  先觸及 +TARGET 停利 → 贏；先觸及 −STOP 停損 → 輸（同一天兩者都碰到，保守算輸）
  到期仍未觸及 → 扣手續費與證交稅後賺錢算贏，否則算輸

避免過度擬合：
  - 依日期切成訓練期（前 70%）與驗證期（後 30%），規則要在沒看過的驗證期也成立才採用
  - 樣本需來自足夠多的不同日期、不同股票（同一檔連續幾天的訊號高度相關）
"""
import itertools
import logging
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app import db
from app.rules import features as F

log = logging.getLogger(__name__)

HOLD, TARGET, STOP = 20, 0.08, 0.05
ROUND_TRIP_COST = 0.001425 * 0.3 * 2 + 0.003
MIN_TRAIN, MIN_TEST, MIN_DATES, MIN_STOCKS = 60, 25, 15, 15
# 判斷以「每筆期望報酬」為主：勝率高不代表賺錢（小賺大賠也能有高勝率）
BUY_EXP, MIN_RATE, AVOID_EXP = 1.0, 0.50, 0.0  # 推薦：兩期期望 ≥ +1%、勝率 ≥ 50%；不推薦：期望 ≤ 0
MAX_ORDER = 3
status = {"running": False, "message": "", "trained_at": None}
_lock = threading.Lock()


def params() -> dict:
    """勝率定義與門檻（設定頁可調）。"""
    g = db.get_setting
    return {"hold_days": int(g("winrate_hold", HOLD)), "target": g("winrate_target", TARGET), "stop": g("winrate_stop", STOP),
            "buy_exp": g("winrate_buy_exp", BUY_EXP), "min_rate": g("winrate_min_rate", MIN_RATE),
            "avoid_exp": g("winrate_avoid_exp", AVOID_EXP)}


def label_trades(df: pd.DataFrame, hold: int = HOLD, target: float = TARGET, stop: float = STOP) -> pd.DataFrame:
    """每個訊號日的交易結果：y（1 贏、0 輸）與 ret（扣成本後報酬）；未來資料不足為 NaN。"""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(df)
    out = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    for t in range(n - hold - 1):
        entry = o[t + 1]
        if not entry or np.isnan(entry):
            continue
        tp, sl = entry * (1 + target), entry * (1 - stop)
        res, r = None, None
        for k in range(t + 1, t + 1 + hold):
            if l[k] <= sl:
                res, r = 0, -stop
                break
            if h[k] >= tp:
                res, r = 1, target
                break
        if res is None:
            r = c[t + hold] / entry - 1
            res = 1 if r - ROUND_TRIP_COST > 0 else 0
        out[t], rets[t] = res, r - ROUND_TRIP_COST
    return pd.DataFrame({"y": out, "ret": rets}, index=df.index)


def _panel(min_avg_volume: float, min_price: float, days: int = 420):
    last = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    since = (datetime.fromisoformat(last) - timedelta(days=days)).date().isoformat()
    allp = db.load_all_prices(since)
    allp["date"] = pd.to_datetime(allp["date"])
    panel = {}
    for sid, g in allp.groupby("stock_id"):
        if sid.startswith("00") or len(g) < 100:
            continue
        g = g.set_index("date").drop(columns="stock_id")
        if g["volume"].tail(60).mean() < min_avg_volume or g["close"].iloc[-1] < min_price:
            continue
        panel[sid] = g
    industry = {r["stock_id"]: r["industry"] for r in db.query("SELECT stock_id, industry FROM stocks")}
    return panel, industry


def _features_for(sid: str, g: pd.DataFrame, industry: dict, market_bull: pd.Series, sector_rank: pd.DataFrame) -> pd.DataFrame:
    f = F.stock_features(g)
    ind = industry.get(sid, "其他")
    f["market_bull"] = market_bull.reindex(f.index).fillna(False).astype(bool)
    rank = sector_rank[ind].reindex(f.index) if ind in sector_rank.columns else pd.Series(np.nan, index=f.index)
    f["sector_strong"] = (rank <= 5).fillna(False).astype(bool)
    return f


def build_dataset(min_avg_volume: float = 1000, min_price: float = 10, p: dict | None = None):
    p = p or params()
    panel, industry = _panel(min_avg_volume, min_price)
    market_bull, sector_rank = F.cross_section(panel, industry)
    rows = []
    for sid, g in panel.items():
        f = _features_for(sid, g, industry, market_bull, sector_rank)
        lab = label_trades(g, p["hold_days"], p["target"], p["stop"])
        d = f[f["_valid"]].drop(columns="_valid")
        d["y"] = lab["y"].reindex(d.index)
        d["ret"] = lab["ret"].reindex(d.index)
        d = d.dropna(subset=["y"])
        d["sid"] = sid
        rows.append(d.reset_index())
    data = pd.concat(rows, ignore_index=True)
    return data, panel, industry, market_bull, sector_rank


def mine_rules(data: pd.DataFrame, p: dict | None = None) -> dict:
    p = p or params()
    keys = list(F.FEATURES)
    dates = np.sort(data["date"].unique())
    split = dates[int(len(dates) * 0.7)]
    tr, te = data[data["date"] < split], data[data["date"] >= split]
    Xtr = {k: tr[k].values for k in keys}
    Xte = {k: te[k].values for k in keys}
    ytr, yte = tr["y"].values, te["y"].values
    rtr, rte = tr["ret"].values, te["ret"].values
    dtr, str_ = tr["date"].values, tr["sid"].values
    base_tr, base_te = float(ytr.mean()), float(yte.mean())
    rules = []
    for order in range(1, MAX_ORDER + 1):
        for combo in itertools.combinations(keys, order):
            m = np.logical_and.reduce([Xtr[k] for k in combo])
            n_tr = int(m.sum())
            if n_tr < MIN_TRAIN:
                continue
            if len(np.unique(dtr[m])) < MIN_DATES or len(np.unique(str_[m])) < MIN_STOCKS:
                continue
            mt = np.logical_and.reduce([Xte[k] for k in combo])
            n_te = int(mt.sum())
            if n_te < MIN_TEST:
                continue
            rules.append({
                "conds": list(combo), "names": [F.FEATURES[k] for k in combo],
                "train_n": n_tr, "train_rate": round(float(ytr[m].mean()), 4),
                "test_n": n_te, "test_rate": round(float(yte[mt].mean()), 4),
                "train_exp": round(float(rtr[m].mean()) * 100, 2), "test_exp": round(float(rte[mt].mean()) * 100, 2),
            })
    for r in rules:
        # 採用的勝率：訓練與驗證期合併（兩期都要達標才算可靠）
        r["rate"] = round((r["train_rate"] * r["train_n"] + r["test_rate"] * r["test_n"]) / (r["train_n"] + r["test_n"]), 4)
        r["n"] = r["train_n"] + r["test_n"]
        r["exp"] = round((r["train_exp"] * r["train_n"] + r["test_exp"] * r["test_n"]) / r["n"], 2)  # 每筆期望報酬 %
        r["reliable_buy"] = (min(r["train_exp"], r["test_exp"]) >= p["buy_exp"]
                             and min(r["train_rate"], r["test_rate"]) >= p["min_rate"])
        r["reliable_avoid"] = max(r["train_exp"], r["test_exp"]) <= p["avoid_exp"]
    rules.sort(key=lambda r: (min(r["train_exp"], r["test_exp"]), r["n"]), reverse=True)
    return {
        "trained_at": db.now(),
        "params": p,
        "period": {"from": str(pd.Timestamp(dates[0]).date()), "split": str(pd.Timestamp(split).date()),
                   "to": str(pd.Timestamp(dates[-1]).date())},
        "samples": {"train": int(len(tr)), "test": int(len(te)), "stocks": int(data["sid"].nunique())},
        "base_rate": {"train": round(base_tr, 4), "test": round(base_te, 4), "all": round(float(data["y"].mean()), 4)},
        "base_exp": round(float(data["ret"].mean()) * 100, 2),
        "best": max(rules, key=lambda r: min(r["train_exp"], r["test_exp"])) if rules else None,
        "single": sorted([{"cond": r["conds"][0], "name": r["names"][0], "rate": r["rate"], "n": r["n"], "exp": r["exp"],
                           "train_rate": r["train_rate"], "test_rate": r["test_rate"]}
                          for r in rules if len(r["conds"]) == 1], key=lambda x: -x["rate"]),
        "rules": rules,
    }


def train() -> dict:
    if not _lock.acquire(blocking=False):
        return {"running": True}
    try:
        from app.services.recommender import risk_settings
        rs = risk_settings()
        status.update(running=True, message="建立訓練資料…")
        p = params()
        data, *_ = build_dataset(rs["min_avg_volume"], rs["min_price"], p)
        status["message"] = f"在 {len(data):,} 筆樣本中尋找條件組合…"
        model = mine_rules(data, p)
        db.cache_set("winrate_model", model)
        status.update(message="完成", trained_at=model["trained_at"])
        return {k: model[k] for k in ("trained_at", "period", "samples", "base_rate")} | {
            "rules": len(model["rules"]), "buy_rules": sum(r["reliable_buy"] for r in model["rules"])}
    finally:
        status["running"] = False
        _lock.release()


def model() -> dict | None:
    return db.cache_get("winrate_model")


# ---------------- 即時比對 ----------------
_today_cache: dict = {}


def today_features() -> dict[str, dict[str, bool]]:
    """最新交易日每檔股票的條件（含市場、族群），每日快取一次。"""
    last = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    hit = db.cache_get(f"winrate_today:{last}")
    if hit:
        return hit
    from app.services.recommender import risk_settings
    rs = risk_settings()
    panel, industry = _panel(0, 0, days=260)
    market_bull, sector_rank = F.cross_section(panel, industry)
    out = {}
    for sid, g in panel.items():
        f = _features_for(sid, g, industry, market_bull, sector_rank)
        row = f.iloc[-1]
        if row["_valid"]:
            out[sid] = {k: bool(row[k]) for k in F.FEATURES}
    db.cache_set(f"winrate_today:{last}", out)
    return out


def assess(cond: dict[str, bool], m: dict | None = None) -> dict:
    """依目前成立的條件，找出符合的歷史規則並給勝率判斷。"""
    m = m or model()
    if not m:
        return {"trained": False}
    true = {k for k, v in cond.items() if v}
    matched = [r for r in m["rules"] if set(r["conds"]) <= true]
    buy = [r for r in matched if r["reliable_buy"]]
    # 代表規則：最具體（條件最多）的組合中樣本最多者；沒有則用全市場平均
    if matched:
        top_order = max(len(r["conds"]) for r in matched)
        rep = max((r for r in matched if len(r["conds"]) == top_order), key=lambda r: r["n"])
    else:
        rep = None
    best_buy = max(buy, key=lambda r: (min(r["train_exp"], r["test_exp"]), r["n"])) if buy else None
    # 綜合估計：符合的最具體組合依樣本數加權平均（不挑對自己有利的單一組合）
    if matched:
        top = [r for r in matched if len(r["conds"]) == top_order]
        w = sum(r["n"] for r in top)
        avg_rate = sum(r["rate"] * r["n"] for r in top) / w
        avg_exp = round(sum(r["exp"] * r["n"] for r in top) / w, 2)
        n_top = len(top)
    else:
        avg_rate, avg_exp, n_top = m["base_rate"]["all"], m.get("base_exp"), 0
    rate = best_buy["rate"] if best_buy else avg_rate
    exp = best_buy["exp"] if best_buy else avg_exp
    decision = "buy" if best_buy else ("avoid" if exp is not None and exp <= m["params"]["avoid_exp"] else "neutral")
    return {
        "trained": True, "decision": decision, "rate": round(rate, 4), "exp": exp, "n_rules": n_top,
        "rule": best_buy or rep, "matched": sorted(matched, key=lambda r: -r["exp"])[:5],
        "true_conds": [F.FEATURES[k] for k in F.FEATURES if cond.get(k)],
        "base_rate": m["base_rate"]["all"], "params": m["params"], "trained_at": m["trained_at"],
        "period": m["period"],
    }

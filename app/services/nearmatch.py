"""多數條件勝率：講義型態「大部分條件成立、少數未成立」時，歷史上的勝率與期望報酬。

一個條件沒到位（例如與五日線乖離 > 8%）不必然代表不能做：
用過去一年的資料回測「同一型態、同樣缺這幾個條件」的所有樣本，
訓練期與驗證期都達到推薦門檻（期望報酬、勝率）才建議買進。

交易定義與勝率模型相同（winrate.label_policy）：隔日開盤買進，統一的移動停利出場規則。
"""
import itertools
import logging
import math

import numpy as np
import pandas as pd

from app import db
from app.rules import features as F
from app.rules.technical import SETUP_NAMES
from app.services import winrate as W

log = logging.getLogger(__name__)

MIN_TRAIN, MIN_TEST, MIN_DATES, MIN_STOCKS = 40, 20, 10, 10


def need(k: int) -> int:
    """「多數條件」：至少三分之二（3 條要 2 條、5 條要 4 條、6 條要 4 條）。"""
    return math.ceil(k * 2 / 3)


def _stock_rows(sid: str, g: pd.DataFrame, p: dict) -> list[pd.DataFrame]:
    tables, weak = F.setup_conditions(g)
    lab = W.label_policy(sid, g)
    bias = (g["close"] / g["close"].rolling(5).mean() - 1 > 0.08).values
    out = []
    for s, t in tables.items():
        keys = list(t.columns)
        met = t.values.sum(axis=1)
        m = (met >= need(len(keys))) & ~weak.values & lab["y"].notna().values
        if not m.any():
            continue
        miss = ["+".join(k for k, ok in zip(keys, row) if not ok) for row in t.values[m]]
        out.append(pd.DataFrame({"setup": s, "miss": miss, "bias": bias[m], "date": g.index[m], "sid": sid,
                                 "y": lab["y"].values[m], "ret": lab["ret"].values[m]}))
    return out


def _stats(d: pd.DataFrame, split, p: dict) -> dict | None:
    tr, te = d[d["date"] < split], d[d["date"] >= split]
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST or d["date"].nunique() < MIN_DATES or d["sid"].nunique() < MIN_STOCKS:
        return None
    r = {"train_n": len(tr), "test_n": len(te), "n": len(d),
         "train_rate": round(float(tr["y"].mean()), 4), "test_rate": round(float(te["y"].mean()), 4),
         "train_exp": round(float(tr["ret"].mean()) * 100, 2), "test_exp": round(float(te["ret"].mean()) * 100, 2),
         "rate": round(float(d["y"].mean()), 4), "exp": round(float(d["ret"].mean()) * 100, 2),
         "stocks": int(d["sid"].nunique()),
         "big": round(float(max((tr["ret"] < -p["big_loss"]).mean(), (te["ret"] < -p["big_loss"]).mean())), 4)}
    r["reliable_buy"] = (min(r["train_exp"], r["test_exp"]) > p["buy_exp"] and min(r["train_rate"], r["test_rate"]) >= p["min_rate"]
                         and r["big"] <= p["max_big"])
    r["reliable_avoid"] = max(r["train_exp"], r["test_exp"]) <= p["avoid_exp"]
    return r


def train(panel: dict[str, pd.DataFrame], p: dict, industry: dict | None = None) -> dict:
    industry = industry if industry is not None else W._panel(0, 0, days=30)[1]
    mb, sr = F.cross_section(panel, industry)
    parts = []
    for sid, g in panel.items():
        rows = _stock_rows(sid, g, p)
        if rows:
            f = W._features_for(sid, g, industry, mb, sr).drop(columns="_valid")
            parts += [r.join(f, on="date") for r in rows]
    data = pd.concat(parts, ignore_index=True)
    dates = np.sort(data["date"].unique())
    split = dates[int(len(dates) * 0.7)]
    stats = {}
    for s, ds in data.groupby("setup"):
        for bias_key, db_ in (("any", ds), ("yes", ds[ds["bias"]]), ("no", ds[~ds["bias"]])):
            st = _stats(db_, split, p)
            if st:
                stats[f"{s}|*|{bias_key}"] = st
            for miss, dm in db_.groupby("miss"):
                st = _stats(dm, split, p)
                if st:
                    stats[f"{s}|{miss}|{bias_key}"] = st
    # 多數條件成立 ＋ 1~2 個市場／量價條件：找出訓練期與驗證期都達推薦門檻的組合
    extra = []
    keys = list(F.FEATURES)
    for s, ds in data.groupby("setup"):
        X = {k: ds[k].values for k in keys}
        for order in (1, 2):
            for combo in itertools.combinations(keys, order):
                m = np.logical_and.reduce([X[k] for k in combo])
                if m.sum() < MIN_TRAIN + MIN_TEST:
                    continue
                st = _stats(ds[m], split, p)
                if st and st["reliable_buy"]:
                    extra.append({"setup": s, "conds": list(combo), "names": [F.FEATURES[k] for k in combo], **st})
    extra.sort(key=lambda r: -min(r["train_exp"], r["test_exp"]))
    model = {"trained_at": db.now(), "params": p, "split": str(pd.Timestamp(split).date()),
             "samples": int(len(data)), "stats": stats, "extra": extra}
    db.cache_set("nearmatch_model", model)
    return model


def model() -> dict | None:
    return db.cache_get("nearmatch_model")


def today() -> dict[str, dict]:
    """最新交易日每檔股票、每個型態成立了哪些條件（每日快取）。"""
    last = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    key = f"nearmatch_today:v1:{last}"
    hit = db.cache_get(key)
    if hit:
        return hit
    panel, _ = W._panel(0, 0, days=420)
    last_ts = pd.Timestamp(last)
    out = {}
    for sid, g in panel.items():
        if g.index[-1] != last_ts:
            continue
        tables, weak = F.setup_conditions(g)
        out[sid] = {"weak": bool(weak.iloc[-1]),
                    "bias": bool(g["close"].iloc[-1] / g["close"].tail(5).mean() - 1 > 0.08),
                    "setups": {s: {k: bool(v) for k, v in t.iloc[-1].items()} for s, t in tables.items()}}
    db.cache_set(key, out)
    return out


def assess(stock_id: str, m: dict | None = None, feats: dict | None = None) -> dict | None:
    """回傳此股所有「多數條件成立」的型態與其歷史勝率；best 為可建議買進者中期望報酬最高的。"""
    m = m or model()
    if not m:
        return None
    t = today().get(stock_id)
    if t is None:
        return None
    bias = "yes" if t["bias"] else "no"
    if feats is None:
        feats = W.today_features().get(stock_id) or {}
    items = []
    for s, conds in t["setups"].items():
        keys = list(conds)
        met = [k for k in keys if conds[k]]
        if len(met) < need(len(keys)):
            continue
        miss = "+".join(k for k in keys if not conds[k])
        names = F.SETUP_CONDS[s]
        basis, st = None, None
        for key, label in ((f"{s}|{miss}|{bias}", "同型態、缺同樣條件、乖離狀態相同"), (f"{s}|{miss}|any", "同型態、缺同樣條件"),
                           (f"{s}|*|{bias}", "同型態多數條件成立、乖離狀態相同"), (f"{s}|*|any", "同型態多數條件成立")):
            if key in m["stats"]:
                basis, st = label, m["stats"][key]
                break
        boost = None
        if not (st and st["reliable_buy"]):
            hits = [r for r in m.get("extra", []) if r["setup"] == s and all(feats.get(k) for k in r["conds"])]
            if hits:
                boost = hits[0]
                st, basis = boost, "同型態多數條件成立，且同時符合：" + "、".join(boost["names"])
        decision = "unknown" if st is None else ("buy" if st["reliable_buy"] else "avoid" if st["reliable_avoid"] else "neutral")
        items.append({"setup": s, "name": SETUP_NAMES[s], "met": [names[k] for k in met],
                      "missing": [names[k] for k in keys if not conds[k]], "total": len(keys),
                      "stats": st, "basis": basis, "decision": decision, "extra": boost["names"] if boost else []})
    items.sort(key=lambda x: (x["decision"] != "buy", -(min(x["stats"]["train_exp"], x["stats"]["test_exp"]) if x["stats"] else -99)))
    best = items[0] if items and items[0]["decision"] == "buy" else None
    return {"items": items, "best": best, "weak": t["weak"], "bias": t["bias"], "params": m["params"], "trained_at": m["trained_at"]}

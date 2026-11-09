"""明日大盤預測：加權指數「明天收漲的機率」「預估高低區間」與「主要原因」。

只用今天收盤（含）以前就知道的資料：
  指數本身：1／5／20 日報酬、與 5／20／60 日線乖離、ATR 波動、連漲連跌天數、K 棒位置
  市場廣度：站上月線比例、上漲家數比例、創 20 日新高／新低家數差
  籌碼：外資、投信、自營商全市場買賣超金額（億）與 5 日累計、融資餘額變化
  星期幾
模型：LightGBM（淺層、強正則化）＋ 走勢前進式驗證（每段只用更早的資料訓練），
回報樣本外準確率並與「每天都猜漲」的基準比較。區間預測以 ATR 為單位的歷史分位數。

注意：台股隔天開盤很受美股夜盤影響，這裡收盤後就預測，沒有用到美股資料。
"""
import logging
import threading
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from app import db

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*eval_set.*")
_lock = threading.Lock()

NAMES = {
    "ret1": "今日漲跌", "ret5": "5 日漲跌", "ret20": "20 日漲跌", "gap5": "與 5 日線乖離", "gap20": "與月線乖離",
    "gap60": "與季線乖離", "atr_pct": "波動度（ATR）", "streak": "連漲／連跌天數", "body": "今日 K 棒位置（收在高點或低點）",
    "above20": "站上月線家數比例", "adv": "上漲家數比例", "nh_nl": "創新高減新低家數", "above20_chg5": "站上月線比例 5 日變化",
    "foreign": "外資買賣超（億）", "foreign5": "外資 5 日累計（億）", "trust": "投信買賣超（億）", "trust5": "投信 5 日累計（億）",
    "dealer": "自營商買賣超（億）", "margin_chg": "融資餘額變化", "margin_chg5": "融資 5 日變化", "dow": "星期幾",
}
NIGHT = {"night_ret": "台指期夜盤漲跌", "night_hi": "夜盤最高", "night_lo": "夜盤最低"}


# ======================= 特徵 =======================
_frame_cache: dict = {}


def market_frame() -> pd.DataFrame:
    """特徵表（依最新日 K／法人／夜盤日期快取在記憶體，盤後資料沒變就不重算）。"""
    sig = tuple(db.one(q)["d"] for q in ("SELECT MAX(date) d FROM prices", "SELECT MAX(date) d FROM index_daily",
                                         "SELECT MAX(date) d FROM ml_sync", "SELECT COUNT(*) d FROM futures_daily"))
    if _frame_cache.get("sig") != sig:
        _frame_cache.update(sig=sig, frame=_market_frame())
    return _frame_cache["frame"].copy()


def _market_frame() -> pd.DataFrame:
    idx = pd.DataFrame(db.query("SELECT date, open, high, low, close FROM index_daily WHERE code='TAIEX' ORDER BY date"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date").astype(float)
    c, h, l, o = idx["close"], idx["high"], idx["low"], idx["open"]
    f = pd.DataFrame(index=idx.index)
    for n in (1, 5, 20):
        f[f"ret{n}"] = c.pct_change(n) * 100
    for n in (5, 20, 60):
        f[f"gap{n}"] = (c / c.rolling(n).mean() - 1) * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr_pct"] = atr / c * 100
    sign = np.sign(c.diff().fillna(0))
    f["streak"] = sign * sign.groupby((sign != sign.shift()).cumsum()).cumcount().add(1)
    f["body"] = ((c - l) / (h - l).replace(0, np.nan))
    f["dow"] = idx.index.dayofweek.astype(float)

    # 市場廣度（全部個股，只用當天以前）
    p = pd.DataFrame(db.query("SELECT stock_id, date, high, low, close, volume FROM prices WHERE date>=?",
                              ((idx.index[0] - pd.Timedelta(days=60)).date().isoformat(),)))
    if len(p):
        p["date"] = pd.to_datetime(p["date"])
        cl = p.pivot(index="date", columns="stock_id", values="close")
        hi = p.pivot(index="date", columns="stock_id", values="high")
        lo = p.pivot(index="date", columns="stock_id", values="low")
        valid = cl.notna()
        above = (cl > cl.rolling(20, min_periods=15).mean()).where(valid)
        b = pd.DataFrame(index=cl.index)
        b["above20"] = above.mean(axis=1) * 100
        chg = cl.pct_change()
        b["adv"] = (chg > 0).where(chg.notna()).mean(axis=1) * 100
        b["nh_nl"] = ((hi >= hi.rolling(20, min_periods=15).max()).sum(axis=1)
                      - (lo <= lo.rolling(20, min_periods=15).min()).sum(axis=1)) / valid.sum(axis=1) * 100
        b["above20_chg5"] = b["above20"].diff(5)
        f = f.join(b)
        # 法人金額（張 × 收盤 × 1000 → 億）
        ch = pd.DataFrame(db.query("SELECT stock_id, date, foreign_net, trust_net, dealer_net FROM chips_daily"))
        if len(ch):
            ch["date"] = pd.to_datetime(ch["date"])
            close_long = p[["stock_id", "date", "close"]]
            ch = ch.merge(close_long, on=["stock_id", "date"], how="left")
            for k in ("foreign_net", "trust_net", "dealer_net"):
                ch[k] = ch[k] * ch["close"] * 1000 / 1e8
            agg = ch.groupby("date")[["foreign_net", "trust_net", "dealer_net"]].sum()
            agg.columns = ["foreign", "trust", "dealer"]
            # 沒有法人資料的日子是 NaN，不是 0
            have = pd.Series(1, index=pd.to_datetime([r["date"] for r in db.query("SELECT date FROM ml_sync WHERE kind='insti'")]))
            agg = agg.reindex(f.index).where(have.reindex(f.index).notna())
            agg["foreign5"] = agg["foreign"].rolling(5).sum()
            agg["trust5"] = agg["trust"].rolling(5).sum()
            f = f.join(agg)
        mg = pd.DataFrame(db.query("SELECT date, SUM(margin_bal) AS bal FROM margin_daily GROUP BY date"))
        if len(mg):
            mg["date"] = pd.to_datetime(mg["date"])
            mg = mg.set_index("date")["bal"].reindex(f.index)
            f["margin_chg"] = mg.pct_change(fill_method=None) * 100
            f["margin_chg5"] = mg.pct_change(5, fill_method=None) * 100
    # 台指期夜盤（明天日期的「盤後」列 = 今晚 15:00 到明早 05:00，在明天開盤前就知道）
    nf = night_frame()
    if len(nf):
        f = f.join(nf.reindex(f.index.union(nf.index)).shift(-1).reindex(f.index))
    # 目標（明天）：收漲與否、報酬、最高／最低相對今天收盤（以 ATR 為單位）
    f["y"] = (c.shift(-1) > c).astype(float).where(c.shift(-1).notna())
    f["next_ret"] = (c.shift(-1) / c - 1) * 100
    f["next_open"] = (o.shift(-1) / c - 1) * 100
    f["next_hi_atr"] = (h.shift(-1) - c) / atr
    f["next_lo_atr"] = (l.shift(-1) - c) / atr
    f["close"], f["atr"] = c, atr
    return f


FEATS = list(NAMES)
FEATS_NIGHT = FEATS + list(NIGHT)


def night_frame() -> pd.DataFrame:
    """每個交易日 D 的夜盤（近月＝成交量最大的合約）：漲跌 %、最高、最低（相對參考價）。"""
    rows = db.query("SELECT date, contract, open, high, low, close, change, volume FROM futures_daily WHERE session='盤後'") \
        if db.one("SELECT name FROM sqlite_master WHERE name='futures_daily'") else []
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d = d.sort_values("volume").groupby("date").tail(1)
    ref = d["close"] - d["change"]
    out = pd.DataFrame({"night_ret": (d["close"] / ref - 1) * 100, "night_hi": (d["high"] / ref - 1) * 100,
                        "night_lo": (d["low"] / ref - 1) * 100}).set_index(pd.to_datetime(d["date"]))
    return out.sort_index()


# ======================= 訓練與驗證 =======================
def _model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1)


def _walk(data: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """走勢前進式驗證：前 50% 起，每段 60 個交易日，只用更早的資料訓練（留 1 天間隔）。"""
    X, y = data[feats].astype(float), data["y"].astype(int).to_numpy()
    start = int(len(data) * 0.5)
    probs = pd.Series(np.nan, index=data.index)
    for s in range(start, len(data), 60):
        e = min(s + 60, len(data))
        m = _model().fit(X.iloc[:s - 1], y[:s - 1])
        probs.iloc[s:e] = m.predict_proba(X.iloc[s:e])[:, 1]
    return data.loc[probs.notna()].assign(prob=probs.dropna())


def _score(oos: pd.DataFrame) -> dict:
    hit = (oos["prob"] >= 0.5) == (oos["y"] == 1)
    conf = oos[(oos["prob"] >= 0.6) | (oos["prob"] <= 0.4)]
    base_up = float(oos["y"].mean())
    return {"n": int(len(oos)), "from": str(oos.index[0].date()), "accuracy": round(float(hit.mean()), 4),
            "base_up": round(base_up, 4), "baseline": round(max(base_up, 1 - base_up), 4),
            "conf_n": int(len(conf)),
            "conf_accuracy": round(float(((conf["prob"] >= 0.5) == (conf["y"] == 1)).mean()), 4) if len(conf) else None,
            "up_when_high": round(float(oos[oos["prob"] >= 0.6]["y"].mean()), 4) if (oos["prob"] >= 0.6).any() else None,
            "up_when_low": round(float(oos[oos["prob"] <= 0.4]["y"].mean()), 4) if (oos["prob"] <= 0.4).any() else None}


def _band(data: pd.DataFrame, center_col: str | None) -> dict:
    """區間：明天最高／最低相對「預估中心」的距離（ATR 為單位）的歷史分位數，並回報樣本外涵蓋率。"""
    if center_col:
        a, b = np.polyfit(data[center_col], data["next_ret"], 1)   # 明天漲跌 ≈ a × 夜盤漲跌 + b
    else:
        a, b = 0.0, 0.0
    ctr = (a * (data[center_col] if center_col else 0) + b) / 100 * data["close"] / data["atr"]
    hi, lo = data["next_hi_atr"] - ctr, data["next_lo_atr"] - ctr
    half = len(data) // 2
    q = lambda s, p: float(s.iloc[:half].quantile(p))
    band = {"slope": round(float(a), 4), "intercept": round(float(b), 4), "hi50": float(hi.median()), "lo50": float(lo.median()),
            "hi90": float(hi.quantile(0.9)), "lo10": float(lo.quantile(0.1))}
    # 樣本外涵蓋率：用前半段的分位數，看後半段實際最高、最低落在 80% 區間內的比例
    inside = (hi.iloc[half:] <= q(hi, 0.9)) & (lo.iloc[half:] >= q(lo, 0.1))
    band["coverage80"] = round(float(inside.mean()), 4)
    return band


def train() -> dict:
    if not _lock.acquire(blocking=False):
        return {"running": True}
    try:
        f = market_frame()
        meta = {"trained_at": db.now()}
        for key, feats, need in (("close", FEATS, []), ("night", FEATS_NIGHT, ["night_ret"])):
            data = f.dropna(subset=["y", "next_hi_atr", "ret20", "gap60", "atr_pct"] + need)
            oos = _walk(data, feats)
            final = _model().fit(data[feats].astype(float), data["y"].astype(int))
            final.booster_.save_model(str(_path(key)))
            imp = pd.Series(final.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
            names = {**NAMES, **NIGHT}
            meta[key] = {"samples": int(len(data)), "period": {"from": str(data.index[0].date()), "to": str(data.index[-1].date())},
                         "oos": _score(oos), "band": _band(data, "night_ret" if key == "night" else None),
                         "importance": [{"feature": names[k], "gain": round(float(v), 1)} for k, v in imp.head(10).items()]}
        db.cache_set("market_forecast_model", meta)
        return meta
    finally:
        _lock.release()


def _path(key: str = "close"):
    from app.config import get_settings
    return get_settings().db_path.parent / f"market_forecast_{key}.txt"


# ======================= 預測 =======================
def _night_now(last_close_day: pd.Timestamp) -> dict | None:
    """明天的夜盤資料：已下載的歷史（明天日期的盤後列）優先，否則用即時報價（今天 15:00 後到明天開盤前）。"""
    nf = night_frame()
    nxt = nf[nf.index > last_close_day]
    if len(nxt):
        r = nxt.iloc[0]
        return {"night_ret": float(r["night_ret"]), "night_hi": float(r["night_hi"]), "night_lo": float(r["night_lo"]),
                "source": "夜盤收盤（期交所）", "time": "05:00"}
    now = datetime.now()
    if not (now.hour >= 15 or now.hour < 9) or now.date() < last_close_day.date():
        return None
    from app.data import futures
    q = futures.night_live()
    if not q or not q.get("ref") or not q.get("volume"):
        return None
    t = q.get("time") or ""
    return {"night_ret": q["ret"], "night_hi": (q["high"] / q["ref"] - 1) * 100 if q.get("high") else q["ret"],
            "night_lo": (q["low"] / q["ref"] - 1) * 100 if q.get("low") else q["ret"],
            "source": f"夜盤即時（{q['contract']}）", "time": f"{t[:2]}:{t[2:4]}" if len(t) >= 4 else t, "last": q["last"]}


def predict(force: bool = False) -> dict | None:
    import lightgbm as lgb
    meta = db.cache_get("market_forecast_model")
    if meta is None or "night" not in meta or not _path("close").exists():
        meta = train()
    f = market_frame()
    last = f.dropna(subset=["ret20", "gap60", "atr_pct"]).iloc[-1:]
    day = last.index[0]
    night = _night_now(day)
    key = f"market_forecast:{day.date()}:{meta['trained_at']}:{'n' if night else 'c'}"
    if not force and not (night and night["source"].startswith("夜盤即時")):
        hit = db.cache_get(key)
        if hit:
            return hit
    mode = "night" if night else "close"
    feats = FEATS_NIGHT if night else FEATS
    x = last[FEATS].astype(float).copy()
    if night:
        for k in NIGHT:
            x[k] = night[k]
        x = x[feats]
    bst = lgb.Booster(model_file=str(_path(mode)))
    prob = float(bst.predict(x)[0])
    contrib = bst.predict(x, pred_contrib=True)[0][:-1]       # 每個特徵把機率往上／往下推多少（log-odds）
    names = {**NAMES, **NIGHT}
    reasons = []
    for i in np.argsort(-np.abs(contrib))[:5]:
        k, v = feats[i], x.iloc[0, i]
        if np.isnan(v) or abs(contrib[i]) < 1e-3:
            continue
        reasons.append({"feature": names[k], "value": round(float(v), 2), "effect": "偏多" if contrib[i] > 0 else "偏空",
                        "weight": round(float(contrib[i]), 3)})
    m = meta[mode]
    close, atr = float(last["close"].iloc[0]), float(last["atr"].iloc[0])
    b = m["band"]
    center = close * (1 + (b["slope"] * (night["night_ret"] if night else 0) + b["intercept"]) / 100)
    o = m["oos"]
    lean = "偏多" if prob >= 0.6 else "偏空" if prob <= 0.4 else "中性（接近五五波）"
    out = {
        "date": str(day.date()), "mode": mode, "close": close, "atr": round(atr, 1), "prob_up": round(prob, 4), "lean": lean,
        "center": round(center), "night": night,
        "range": {"high": round(center + b["hi50"] * atr), "low": round(center + b["lo50"] * atr),
                  "high90": round(center + b["hi90"] * atr), "low10": round(center + b["lo10"] * atr),
                  "coverage80": b["coverage80"]},
        "reasons": reasons, "model": {"trained_at": meta["trained_at"], "samples": m["samples"], "period": m["period"], "oos": o},
        "confident": prob >= 0.6 or prob <= 0.4, "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out["note"] = (f"樣本外準確率 {o['accuracy']:.0%}（每天都猜漲是 {o['baseline']:.0%}）"
                   + (f"；機率 ≥60% 或 ≤40% 的日子準確率 {o['conf_accuracy']:.0%}（{o['conf_n']} 天）" if o["conf_accuracy"] else ""))
    db.cache_set(key, out)
    return out


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    m = train()
    print(json.dumps({k: {kk: m[k][kk] for kk in ("samples", "oos", "band")} for k in ("close", "night")}, ensure_ascii=False, indent=1))
    print(json.dumps(predict(force=True), ensure_ascii=False, indent=1))

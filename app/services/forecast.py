"""明日大盤預測：加權指數「明天收漲的機率」「預估高低區間」與「主要原因」。

只用今天收盤（含）以前就知道的資料：
  指數本身：1／5／20 日報酬、與 5／20／60 日線乖離、ATR 波動、連漲連跌天數、K 棒位置
  市場廣度：站上月線比例、上漲家數比例、創 20 日新高／新低家數差
  籌碼：外資、投信、自營商全市場買賣超金額（億）與 5 日累計、融資餘額變化
  星期幾
模型：LightGBM（淺層、強正則化）＋ 走勢前進式驗證（每段只用更早的資料訓練），
回報樣本外準確率並與「每天都猜漲」的基準比較。區間預測以 ATR 為單位的歷史分位數。

三個模型依「現在知道多少」自動切換：
  close：台股收盤後（只用台股資料）
  night：台指期夜盤 15:00 開盤後（加夜盤漲跌）
  us   ：美股收盤後（台灣清晨 4～5 點；再加 S&P 500、那斯達克、費半、台積電 ADR 漲跌）
夜盤與美股另外用線性回歸估計「明天開盤」漲跌。
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
    sig += ((db.one("SELECT MAX(date) d, COUNT(*) n FROM us_daily") or {}).get("n")
            if db.one("SELECT name FROM sqlite_master WHERE name='us_daily'") else None,)
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
    f["atr60_pct"] = tr.ewm(alpha=1 / 60, adjust=False, min_periods=60).mean() / c * 100   # 只用來說明「近期波動是平常的幾倍」
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
    # 美股：今天以前已知的（昨晚美股）＋ 明天開盤前才知道的（今晚美股）
    usf = us_features(f.index)
    if len(usf):
        f = f.join(usf)
    # 目標（明天）：收漲與否、報酬、最高／最低相對今天收盤（以 ATR 為單位）
    f["y"] = (c.shift(-1) > c).astype(float).where(c.shift(-1).notna())
    f["next_ret"] = (c.shift(-1) / c - 1) * 100
    f["next_open"] = (o.shift(-1) / c - 1) * 100
    f["next_hi_atr"] = (h.shift(-1) - c) / atr
    f["next_lo_atr"] = (l.shift(-1) - c) / atr
    f["close"], f["atr"] = c, atr
    return f


# 美股：只用「台股收盤後到下一個台股日開盤前」的那一段（昨晚以前的美股台股當天已反應，驗證後沒有幫助，不納入）
US_NEXT = {"us_spy": "美股 S&P 500 漲跌", "us_comp": "那斯達克漲跌", "us_sox": "費城半導體漲跌", "us_tsm": "台積電 ADR 漲跌"}
US_COL = {"spy": "SPY", "comp": "COMP", "sox": "SOX", "tsm": "TSM"}
FEATS = list(NAMES)
FEATS_NIGHT = FEATS + list(NIGHT)
FEATS_US = FEATS_NIGHT + list(US_NEXT)
ALL_NAMES = {**NAMES, **NIGHT, **US_NEXT}
MODES = {  # 模型: (特徵, 必要欄位, 預估中心／開盤用的線性欄位)
    "close": (FEATS, [], []),
    "night": (FEATS_NIGHT, ["night_ret"], ["night_ret"]),
    "us": (FEATS_US, ["night_ret", "us_sox"], ["night_ret", "us_spy", "us_sox", "us_tsm"]),
}


def us_closes() -> pd.DataFrame:
    if not db.one("SELECT name FROM sqlite_master WHERE name='us_daily'"):
        return pd.DataFrame()
    u = pd.DataFrame(db.query("SELECT date, symbol, close FROM us_daily"))
    if not len(u):
        return u
    u = u.pivot(index="date", columns="symbol", values="close")
    u.index = pd.to_datetime(u.index)
    return u.sort_index()


def _us_before(us: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """每個台股日 D 開盤前已收盤的最近一個美股交易日（美股日期 < D）的收盤價。"""
    left = pd.DataFrame({"tw": days}).sort_values("tw")
    right = us.rename_axis("us_date").reset_index().sort_values("us_date")
    m = pd.merge_asof(left, right, left_on="tw", right_on="us_date", allow_exact_matches=False, direction="backward")
    return m.set_index("tw").drop(columns="us_date")


def us_features(days: pd.DatetimeIndex) -> pd.DataFrame:
    """美股報酬（%）：台股 D 收盤後到下一個台股日開盤前的美股漲跌（今晚美股；明早 4～5 點收盤後才知道）。
    遇台股連假，美股多個交易日會累計起來。"""
    us = us_closes()
    if not len(us):
        return pd.DataFrame()
    at = _us_before(us, days)
    out = pd.DataFrame(index=days)
    for k, s in US_COL.items():
        if s in at:
            out[f"us_{k}"] = (at[s].shift(-1) / at[s] - 1) * 100
    return out


def _us_now(last_close_day: pd.Timestamp) -> dict | None:
    """今晚美股是否已收盤（美股日期 ≥ 台股最後收盤日）→ 回傳 us_* 特徵。"""
    us = us_closes()
    if not len(us) or us.index[-1] < last_close_day:
        return None
    base = _us_before(us, pd.DatetimeIndex([last_close_day])).iloc[0]
    last = us.iloc[-1]
    out = {f"us_{k}": float((last[s] / base[s] - 1) * 100) for k, s in US_COL.items()
           if s in us and pd.notna(last[s]) and pd.notna(base[s])}
    if len(out) < len(US_NEXT):
        return None
    out["us_date"] = str(us.index[-1].date())
    return out


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


def _lin(data: pd.DataFrame, cols: list[str], target: str) -> tuple[np.ndarray, float]:
    """線性：target ≈ cols · w + b（前半段估計），回傳 (係數含截距, 後半段平均絕對誤差)。"""
    if not cols:
        w = np.array([float(data[target].iloc[: len(data) // 2].mean())])
        return np.r_[np.zeros(0), w], float((data[target].iloc[len(data) // 2:] - w[0]).abs().mean())
    X = np.c_[data[cols].to_numpy(float), np.ones(len(data))]
    y = data[target].to_numpy(float)
    half = len(data) // 2
    w_half = np.linalg.lstsq(X[:half], y[:half], rcond=None)[0]
    mae = float(np.abs(y[half:] - X[half:] @ w_half).mean())
    return np.linalg.lstsq(X, y, rcond=None)[0], mae


def _apply(w: list[float], cols: list[str], vals: dict) -> float:
    return float(sum(w[i] * vals[c] for i, c in enumerate(cols)) + w[-1])


def _band(data: pd.DataFrame, cols: list[str]) -> dict:
    """區間：明天最高／最低相對「預估中心」的距離（ATR 為單位）的歷史分位數，並回報樣本外涵蓋率。
    預估中心 = 線性（夜盤／美股 → 明天漲跌）；預估開盤 = 線性（夜盤／美股 → 明天開盤漲跌）。"""
    w, _ = _lin(data, cols, "next_ret")
    w_open, open_mae = _lin(data, cols, "next_open")
    X = np.c_[data[cols].to_numpy(float), np.ones(len(data))] if cols else np.ones((len(data), 1))
    ctr = (X @ w) / 100 * data["close"] / data["atr"]
    hi, lo = data["next_hi_atr"] - ctr, data["next_lo_atr"] - ctr
    half = len(data) // 2
    q = lambda s, p: float(s.iloc[:half].quantile(p))
    band = {"cols": cols, "w": [round(float(v), 5) for v in w], "w_open": [round(float(v), 5) for v in w_open],
            "open_mae": round(open_mae, 3), "hi50": float(hi.median()), "lo50": float(lo.median()),
            "hi90": float(hi.quantile(0.9)), "lo10": float(lo.quantile(0.1))}
    # 樣本外涵蓋率：用前半段的分位數，看後半段實際最高、最低落在 80% 區間內的比例
    inside = (hi.iloc[half:] <= q(hi, 0.9)) & (lo.iloc[half:] >= q(lo, 0.1))
    band["coverage80"] = round(float(inside.mean()), 4)
    if cols:   # 開盤方向（開高／開低）的樣本外準確率
        pred = np.c_[data[cols].to_numpy(float), np.ones(len(data))][half:] @ _lin(data.iloc[:half], cols, "next_open")[0]
        band["open_dir_acc"] = round(float(((pred > 0) == (data["next_open"].iloc[half:] > 0)).mean()), 4)
    return band


def train() -> dict:
    if not _lock.acquire(blocking=False):
        return {"running": True}
    try:
        f = market_frame()
        meta = {"trained_at": db.now()}
        for key, (feats, need, cols) in MODES.items():
            if any(c not in f for c in need):
                continue
            data = f.dropna(subset=["y", "next_hi_atr", "next_open", "ret20", "gap60", "atr_pct"] + need + cols)
            oos = _walk(data, feats)
            final = _model().fit(data[feats].astype(float), data["y"].astype(int))
            final.booster_.save_model(str(_path(key)))
            imp = pd.Series(final.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
            meta[key] = {"samples": int(len(data)), "period": {"from": str(data.index[0].date()), "to": str(data.index[-1].date())},
                         "oos": _score(oos), "band": _band(data, cols),
                         "importance": [{"feature": ALL_NAMES[k], "gain": round(float(v), 1)} for k, v in imp.head(10).items()]}
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
        return {"night_ret": round(float(r["night_ret"]), 2), "night_hi": round(float(r["night_hi"]), 2),
                "night_lo": round(float(r["night_lo"]), 2),
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
    if meta is None or "us" not in meta or not _path("us").exists():
        meta = train()
    f = market_frame()
    last = f.dropna(subset=["ret20", "gap60", "atr_pct"]).iloc[-1:]
    day = last.index[0]
    night = _night_now(day)
    us = _us_now(day) if night else None
    mode = "us" if us and "us" in meta else "night" if night else "close"
    key = f"market_forecast:{day.date()}:{meta['trained_at']}:{mode}:{(us or {}).get('us_date', '')}"
    if not force and not (night and night["source"].startswith("夜盤即時")):
        hit = db.cache_get(key)
        if hit:
            return hit
    feats, _, cols = MODES[mode]
    x = last[FEATS].astype(float).copy()
    for k, v in {**(night or {}), **(us or {})}.items():
        if k in ALL_NAMES:
            x[k] = v
    x = x[feats]
    bst = lgb.Booster(model_file=str(_path(mode)))
    prob = float(bst.predict(x)[0])
    contrib = bst.predict(x, pred_contrib=True)[0][:-1]       # 每個特徵把機率往上／往下推多少（log-odds）
    reasons = []
    for i in np.argsort(-np.abs(contrib))[:5]:
        k, v = feats[i], x.iloc[0, i]
        if np.isnan(v) or abs(contrib[i]) < 1e-3:
            continue
        reasons.append({"feature": ALL_NAMES[k], "value": round(float(v), 2), "effect": "偏多" if contrib[i] > 0 else "偏空",
                        "weight": round(float(contrib[i]), 3)})
    m = meta[mode]
    close, atr = float(last["close"].iloc[0]), float(last["atr"].iloc[0])
    b = m["band"]
    vals = {c: float(x[c].iloc[0]) for c in cols}
    center = close * (1 + _apply(b["w"], cols, vals) / 100)
    o = m["oos"]
    lean = "偏多" if prob >= 0.6 else "偏空" if prob <= 0.4 else "中性（接近五五波）"
    out = {
        "date": str(day.date()), "mode": mode, "close": close, "atr": round(atr, 1), "prob_up": round(prob, 4), "lean": lean,
        "center": round(center), "night": night,
        "us": ({**{US_NEXT[k]: round(v, 2) for k, v in us.items() if k in US_NEXT}, "date": us["us_date"]} if us else None),
        "range": {"high": round(center + b["hi50"] * atr), "low": round(center + b["lo50"] * atr),
                  "high90": round(center + b["hi90"] * atr), "low10": round(center + b["lo10"] * atr),
                  "coverage80": b["coverage80"]},
        "reasons": reasons, "model": {"trained_at": meta["trained_at"], "samples": m["samples"], "period": m["period"], "oos": o},
        "confident": prob >= 0.6 or prob <= 0.4, "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    a60 = float(last["atr60_pct"].iloc[0]) if "atr60_pct" in last and pd.notna(last["atr60_pct"].iloc[0]) else None
    out["vol"] = {"atr_pct": round(atr / close * 100, 2), "atr60_pct": round(a60, 2) if a60 else None,
                  "ratio": round(atr / close * 100 / a60, 2) if a60 else None,
                  "wide_pct": round((out["range"]["high90"] - out["range"]["low10"]) / close * 100, 2)}
    if cols:
        op = _apply(b["w_open"], cols, vals)
        out["open"] = {"pct": round(op, 2), "price": round(close * (1 + op / 100)), "mae": b["open_mae"],
                       "dir_acc": b.get("open_dir_acc")}
    out["note"] = (f"樣本外準確率 {o['accuracy']:.0%}（每天都猜漲是 {o['baseline']:.0%}）"
                   + (f"；機率 ≥60% 或 ≤40% 的日子準確率 {o['conf_accuracy']:.0%}（{o['conf_n']} 天）" if o["conf_accuracy"] else ""))
    db.cache_set(key, out)
    return out


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    m = train()
    print(json.dumps({k: {kk: m[k][kk] for kk in ("samples", "oos", "band")} for k in MODES if k in m}, ensure_ascii=False, indent=1))
    print(json.dumps(predict(force=True), ensure_ascii=False, indent=1))

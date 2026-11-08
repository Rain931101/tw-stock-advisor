"""步驟 4：LightGBM 訓練、防過度擬合與實戰門檻。

防止資料洩漏 / 過度擬合：
  1. Purged Walk-Forward CV：依日期切成數段，每段只用「更早」的資料訓練；
     訓練集最後 horizon+1 天的樣本會被清除（purge），因為它們的標籤結果落在測試期內。
  2. early stopping 用「訓練期最後一段」當驗證集，不碰測試期。
  3. 不平衡樣本：class_weight="balanced"（好的進場點通常少於一半）。
  4. 推薦門檻用「樣本外（out-of-sample）」預測決定：
     前半段測試期選機率門檻、後半段測試期驗證，兩段都達「期望報酬 ≥ +1%、勝率 ≥ 50%」才啟用。
  5. 大盤體制濾網：market_trend = 0（加權指數在 20MA 或 60MA 之下）時不發多頭訊號。

指令列執行：uv run python -m app.ml.model
"""
import json
import logging
import warnings
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app import db
from app.config import get_settings
from app.ml import features as MF
from app.ml.labels import triple_barrier

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*eval_set.*")

PARAMS = {"up": 2.0, "down": 1.5, "horizon": 10}   # 三欄：+2 ATR 停利、−1.5 ATR 停損、10 個交易日
N_FOLDS = 5
LGB = dict(n_estimators=3000, learning_rate=0.02, num_leaves=15, max_depth=5, min_child_samples=300, subsample=0.7,
           subsample_freq=1, colsample_bytree=0.6, reg_lambda=10.0, class_weight="balanced", verbose=-1, n_jobs=-1)
MIN_LIFT = 0.5   # 期望報酬至少要比同期間全體樣本多 0.5 個百分點，證明是「選股」而不是「剛好遇到大漲期」
status = {"running": False, "message": "", "trained_at": None}
_lock = threading.Lock()


def model_path():
    return get_settings().db_path.parent / "ml_lgbm.txt"


def gate() -> dict:
    """推薦門檻沿用勝率頁設定（期望報酬、最低勝率）。"""
    from app.services import winrate
    p = winrate.params()
    return {"buy_exp": p["buy_exp"], "min_rate": p["min_rate"]}


# ======================= 資料集 =======================
def build_dataset(min_avg_volume: float = 1000, min_price: float = 10, days: int | None = None, with_labels: bool = True):
    from app.services import winrate
    panel, industry = winrate._panel(min_avg_volume, min_price, days)
    since = min(g.index[0] for g in panel.values()).date().isoformat()
    inputs = MF.load_inputs(since)
    mb, sr = MF.F.cross_section(panel, industry)
    rows = []
    for sid, g in panel.items():
        ind = industry.get(sid, "其他")
        extra = pd.DataFrame({"h_market_bull": mb.astype(float),
                              "h_sector_rank": sr[ind] if ind in sr.columns else np.nan}, index=mb.index)
        f = MF.build_features(sid, g, inputs, extra)
        if with_labels:
            f = f.join(triple_barrier(f, **PARAMS))
        f["sid"] = sid
        rows.append(f.reset_index().rename(columns={"index": "date"}))
    data = pd.concat(rows, ignore_index=True)
    return data


# ======================= Purged Walk-Forward =======================
def purged_walk_forward(dates: np.ndarray, n_folds: int, purge: int, min_train_frac: float = 0.4):
    """回傳 [(train_dates, test_dates)]。訓練一律在測試之前，並清除與測試期重疊的標籤期間。"""
    u = np.sort(np.unique(dates))
    start = int(len(u) * min_train_frac)
    edges = np.linspace(start, len(u), n_folds + 1).astype(int)
    out = []
    for i in range(n_folds):
        test = u[edges[i]:edges[i + 1]]
        train = u[:max(edges[i] - purge, 0)]       # purge：測試期開始前 horizon+1 天不拿來訓練
        if len(train) and len(test):
            out.append((train, test))
    return out


def _fit(X: pd.DataFrame, y: np.ndarray, dates: np.ndarray, purge: int, n_estimators: int | None = None):
    """訓練一個模型；以訓練期最後 15% 日期（再 purge）當 early stopping 驗證集。"""
    import lightgbm as lgb
    u = np.sort(np.unique(dates))
    cut = u[int(len(u) * 0.85)]
    gap = u[max(int(len(u) * 0.85) - purge, 0)]
    tr, va = dates < gap, dates >= cut
    params = dict(LGB)
    if n_estimators:
        params["n_estimators"] = n_estimators
        m = lgb.LGBMClassifier(**params)
        m.fit(X, y)
        return m
    m = lgb.LGBMClassifier(**params)
    # early stopping 看 AUC（排序能力）：balanced 權重下 logloss 會被樣本權重扭曲，容易一開始就停
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False), lgb.log_evaluation(0)])
    return m


def _bucket(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "rate": round(float(df["y"].mean()), 4) if len(df) else None,
            "exp": round(float(df["ret"].mean()) * 100, 2) if len(df) else None,
            "stocks": int(df["sid"].nunique()), "dates": int(df["date"].nunique())}


def select_threshold(oos: pd.DataFrame, g: dict) -> dict:
    """在樣本外預測上找機率門檻：前半段選、後半段驗證，兩段都達標才啟用。"""
    oos = oos[oos["market_trend"] != 0]                      # 大盤空頭體制不做多
    u = np.sort(oos["date"].unique())
    mid = u[len(u) // 2]
    a, b = oos[oos["date"] < mid], oos[oos["date"] >= mid]
    base_a, base_b = _bucket(a), _bucket(b)
    table = []
    # 用「當天全市場排名」選股（例如每天機率前 5%），不受各折模型機率尺度不同影響
    for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
        sa, sb = a[a["rank"] >= q], b[b["rank"] >= q]
        ra, rb = _bucket(sa), _bucket(sb)
        enough = ra["n"] >= 60 and rb["n"] >= 30 and ra["stocks"] >= 15 and rb["stocks"] >= 10
        lift_a = round(ra["exp"] - base_a["exp"], 2) if enough else None
        lift_b = round(rb["exp"] - base_b["exp"], 2) if enough else None
        ok = (enough and min(ra["exp"], rb["exp"]) >= g["buy_exp"] and min(ra["rate"], rb["rate"]) >= g["min_rate"]
              and min(lift_a, lift_b) >= MIN_LIFT)
        table.append({"quantile": q, "threshold": q, "select": ra, "verify": rb, "pass": ok,
                      "lift": [lift_a, lift_b], "all": _bucket(oos[oos["rank"] >= q])})
    passed = [t for t in table if t["pass"]]
    # 通過者中取「驗證期期望報酬」最高的；都不通過就不啟用推薦（只顯示機率）
    best = max(passed, key=lambda t: (min(t["select"]["exp"], t["verify"]["exp"]), t["verify"]["n"])) if passed else None
    return {"table": table, "best": best, "split": str(pd.Timestamp(mid).date()), "base": [base_a, base_b]}


# ======================= 訓練 =======================
def train() -> dict:
    if not _lock.acquire(blocking=False):
        return {"running": True}
    try:
        from app.services.recommender import risk_settings
        rs = risk_settings()
        status.update(running=True, message="建立特徵與三欄式標記…")
        data = build_dataset(rs["min_avg_volume"], rs["min_price"])
        data = data[data["y"].notna()].reset_index(drop=True)
        feats = MF.feature_columns(data.drop(columns=["y", "ret", "barrier", "days", "no_fill", "sid", "date"]))
        X, y, dates = data[feats].astype(float), data["y"].to_numpy(int), data["date"].to_numpy()
        purge = PARAMS["horizon"] + 1
        folds = purged_walk_forward(dates, N_FOLDS, purge)
        oos, iters, fold_info = [], [], []
        for i, (tr_d, te_d) in enumerate(folds, 1):
            status["message"] = f"Purged walk-forward 第 {i}/{len(folds)} 折…"
            tr, te = np.isin(dates, tr_d), np.isin(dates, te_d)
            m = _fit(X[tr], y[tr], dates[tr], purge)
            iters.append(m.best_iteration_ or LGB["n_estimators"])
            p = m.predict_proba(X[te])[:, 1]
            part = data.loc[te, ["date", "sid", "y", "ret", "barrier", "market_trend"]].copy()
            part["prob"] = p
            part["rank"] = part.groupby("date")["prob"].rank(pct=True)   # 當天全市場百分位排名
            oos.append(part)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else None
            fold_info.append({"fold": i, "train_to": str(pd.Timestamp(tr_d[-1]).date()), "test_from": str(pd.Timestamp(te_d[0]).date()),
                              "test_to": str(pd.Timestamp(te_d[-1]).date()), "auc": None if auc is None else round(float(auc), 4),
                              "best_iter": iters[-1], "base": _bucket(part)})
        oos = pd.concat(oos, ignore_index=True)
        g = gate()
        th = select_threshold(oos, g)
        status["message"] = "以全部資料訓練最終模型…"
        n_final = int(np.median(iters))
        final = _fit(X, y, dates, purge, n_estimators=max(n_final, 50))
        final.booster_.save_model(str(model_path()))
        imp = pd.Series(final.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
        meta = {
            "trained_at": db.now(), "params": PARAMS, "gate": g, "features": feats, "n_estimators": n_final,
            "samples": int(len(data)), "stocks": int(data["sid"].nunique()),
            "period": {"from": str(pd.Timestamp(dates.min()).date()), "to": str(pd.Timestamp(dates.max()).date())},
            "base": _bucket(data), "base_bull": _bucket(data[data["market_trend"] == 1]),
            "barriers": data["barrier"].value_counts().to_dict(),
            "folds": fold_info, "threshold": th, "importance": [{"feature": k, "gain": round(float(v), 1)} for k, v in imp.head(25).items()],
            "no_fill": None,
        }
        db.cache_set("ml_model", meta)
        status.update(message="完成", trained_at=meta["trained_at"])
        _today_cache.clear()
        return {k: meta[k] for k in ("trained_at", "samples", "stocks", "period")} | {"enabled": th["best"] is not None}
    finally:
        status["running"] = False
        _lock.release()


def meta() -> dict | None:
    return db.cache_get("ml_model")


# ======================= 預測 =======================
_today_cache: dict = {}


def _booster():
    import lightgbm as lgb
    p = model_path()
    return lgb.Booster(model_file=str(p)) if p.exists() else None


def predict_today() -> dict:
    """最新交易日全市場預測：{stock_id: {prob, market_trend, atr, close, signal}}（每日快取）。"""
    m = meta()
    if not m:
        return {}
    last = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    key = f"ml_today:{last}:{m['trained_at']}"
    if key in _today_cache:
        return _today_cache[key]
    hit = db.cache_get(key)
    if hit:
        _today_cache[key] = hit
        return hit
    bst = _booster()
    if bst is None:
        return {}
    data = build_dataset(0, 0, days=420, with_labels=False)
    today = data[data["date"] == pd.Timestamp(last)]
    X = today.reindex(columns=m["features"]).astype(float)
    prob = bst.predict(X)
    rank = pd.Series(prob).rank(pct=True).to_numpy()
    best = (m["threshold"] or {}).get("best")
    out = {}
    for (_, r), p, rk in zip(today.iterrows(), prob, rank):
        mt = r.get("market_trend")
        out[r["sid"]] = {"prob": round(float(p), 4), "rank": round(float(rk), 4), "market_trend": None if pd.isna(mt) else int(mt),
                         "atr": None if pd.isna(r["atr14"]) else float(r["atr14"]), "adj_close": float(r["adj_close"]),
                         "signal": bool(best and rk >= best["threshold"] and mt == 1 and not r.get("limit_up_close"))}
    db.cache_set(key, out)
    _today_cache[key] = out
    return out


def assess(stock_id: str) -> dict | None:
    m = meta()
    if not m:
        return None
    p = predict_today().get(stock_id)
    if p is None:
        return None
    best = (m["threshold"] or {}).get("best")
    # 機率落在哪個門檻區間 → 用該區間的樣本外統計當作「歷史上同樣信心的勝率與期望報酬」
    rows = [t for t in m["threshold"]["table"] if p["rank"] >= t["threshold"]]
    bucket = rows[-1] if rows else None
    return {**p, "enabled": best is not None, "threshold": best["threshold"] if best else None,
            "best": best, "bucket": bucket, "params": m["params"], "gate": m["gate"], "trained_at": m["trained_at"],
            "base": m["base"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = train()
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    mt = meta()
    print(json.dumps({"folds": mt["folds"], "best": mt["threshold"]["best"]}, ensure_ascii=False, indent=2, default=str))

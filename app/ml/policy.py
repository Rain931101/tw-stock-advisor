"""出場策略搜尋：在「不大賠」的限制下找勝率最高的移動停利設定。

訊號（主訊號）：講義六種進場型態「多數條件成立」且不在弱位階的日子（與多數條件勝率相同的母體）。
每一組出場設定都對全部訊號回測，依日期切成前 70%（選取）與後 30%（驗證）：

  限制（兩段都要）：每筆期望報酬 > MIN_EXP、虧損超過 5% 的比例 ≤ MAX_BIG、最差 1% 平均虧損 ≥ −WORST1
  目標：兩段勝率中較低者最大

指令列：uv run python -m app.ml.policy
"""
import itertools
import json
import logging
import threading

import numpy as np
import pandas as pd

from app import db
from app.ml import exits as X
from app.ml import features as MF
from app.rules import features as F

log = logging.getLogger(__name__)

# 大賠 = 虧損超過 6%（超出計畫停損：多半是跳空跌破或連續跌停）
MIN_EXP, MAX_BIG, WORST1, BIG_LOSS = 0.2, 0.02, 9.0, 0.06   # 與勝率頁的統一門檻相同
GRID = {
    "sl_atr": [1.0, 1.5, 2.0],
    "sl_cap": [0.03, 0.04, 0.05],
    "be_atr": [None, 0.5, 1.0, 1.5],
    "trail_atr": [None, 1.0, 1.5, 2.0, 3.0],
    "ma5_mode": ["none", "after_be"],
    "horizon": [10, 20],
}
status = {"running": False, "message": "", "trained_at": None}
_lock = threading.Lock()


def policies() -> list[X.Policy]:
    out = []
    for v in itertools.product(*GRID.values()):
        d = dict(zip(GRID, v))
        if d["ma5_mode"] == "after_be" and d["be_atr"] is None:
            continue
        out.append(X.Policy(**d))
    return out


def current() -> X.Policy:
    """目前採用的出場策略（搜尋結果；尚未搜尋時用預設）。"""
    m = db.cache_get("exit_policy")
    return X.Policy(**m["best"]["policy"]) if m and m.get("best") else X.Policy()


def stock_arrays(sid: str, g: pd.DataFrame, divs: dict) -> dict:
    empty = pd.DataFrame(columns=["date", "prev_close", "ref_price"])
    a = MF.adjust_prices(g, divs.get(sid, empty))
    ma5 = a["adj_close"].rolling(5).mean()
    return {"o": a["adj_open"].to_numpy(float), "h": a["adj_high"].to_numpy(float), "l": a["adj_low"].to_numpy(float),
            "c": a["adj_close"].to_numpy(float), "ma5": ma5.to_numpy(float), "atr": MF.atr(a, 14).to_numpy(float),
            "raw_c": g["close"].to_numpy(float), "raw_h": g["high"].to_numpy(float), "raw_l": g["low"].to_numpy(float),
            "ex_day": np.asarray(a["ex_day"], bool), "dates": g.index.to_numpy()}


def events_for(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """主訊號：任一型態成立 ≥ 2/3 條件且非弱位階。回傳 (索引, 當天最高的條件符合比例)。"""
    from app.services.nearmatch import need
    tables, weak = F.setup_conditions(g)
    best = np.zeros(len(g))
    ok = np.zeros(len(g), bool)
    for t in tables.values():
        met = t.values.sum(axis=1)
        ok |= met >= need(t.shape[1])
        best = np.maximum(best, met / t.shape[1])
    ok &= ~weak.to_numpy(bool)
    idx = np.where(ok)[0]
    return idx, best[idx]


def run(policy: X.Policy, data: list[dict]) -> pd.DataFrame:
    parts = []
    for d in data:
        r = X.simulate(d["o"], d["h"], d["l"], d["c"], d["ma5"], d["atr"], d["raw_c"], d["raw_h"], d["raw_l"],
                       d["ex_day"], d["events"], policy)
        if len(r["idx"]):
            parts.append(pd.DataFrame({"date": d["dates"][r["idx"]], "sid": d["sid"], "ret": r["ret"],
                                       "reason": r["reason"], "days": r["days"]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "sid", "ret", "reason", "days"])


def search() -> dict:
    if not _lock.acquire(blocking=False):
        return {"running": True}
    try:
        from app.services import winrate
        from app.services.recommender import risk_settings
        rs = risk_settings()
        status.update(running=True, message="準備資料…")
        panel, _ = winrate._panel(rs["min_avg_volume"], rs["min_price"])
        divs = MF.load_inputs("1900-01-01")["divs"]
        data = []
        for sid, g in panel.items():
            ev, _ = events_for(g)
            d = stock_arrays(sid, g, divs)
            d.update(sid=sid, events=ev)
            data.append(d)
        all_dates = np.sort(np.unique(np.concatenate([d["dates"] for d in data])))
        split = all_dates[int(len(all_dates) * 0.7)]
        pols = policies()
        table = []
        for i, p in enumerate(pols, 1):
            status["message"] = f"回測出場策略 {i}/{len(pols)}"
            res = run(p, data)
            a, b = res[res["date"] < split], res[res["date"] >= split]
            sa, sb = X.stats(a["ret"].to_numpy(), BIG_LOSS), X.stats(b["ret"].to_numpy(), BIG_LOSS)
            if not sa.get("n") or not sb.get("n"):
                continue
            ok = all(s["exp"] > MIN_EXP and s["big_loss_rate"] <= MAX_BIG and s["worst1"] >= -WORST1 for s in (sa, sb))
            table.append({"policy": p.to_dict(), "desc": p.describe(), "select": sa, "verify": sb, "pass": ok,
                          "all": X.stats(res["ret"].to_numpy(), BIG_LOSS),
                          "reasons": res["reason"].value_counts().to_dict(), "avg_days": round(float(res["days"].mean()), 1)})
        passed = [t for t in table if t["pass"]]
        key = lambda t: (min(t["select"]["rate"], t["verify"]["rate"]), min(t["select"]["exp"], t["verify"]["exp"]))
        best = max(passed, key=key) if passed else None
        if best is None and table:
            # 沒有任何設定兩期都達門檻：仍採用「大賠比例最低的一半」中勝率最高者當統一出場規則，並標示未達門檻
            safe = sorted(table, key=lambda t: max(t["select"]["big_loss_rate"], t["verify"]["big_loss_rate"]))[: max(1, len(table) // 2)]
            best = {**max(safe, key=key), "below_gate": True}
        table.sort(key=lambda t: (not t["pass"], -min(t["select"]["rate"], t["verify"]["rate"])))
        out = {"trained_at": db.now(), "split": str(pd.Timestamp(split).date()),
               "period": {"from": str(pd.Timestamp(all_dates[0]).date()), "to": str(pd.Timestamp(all_dates[-1]).date())},
               "limits": {"min_exp": MIN_EXP, "max_big": MAX_BIG, "worst1": WORST1, "big_loss": BIG_LOSS},
               "tested": len(table), "passed": len(passed), "best": best, "table": table[:40]}
        db.cache_set("exit_policy", out)
        status.update(message="完成", trained_at=out["trained_at"])
        return out
    finally:
        status["running"] = False
        _lock.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = search()
    print("tested", r["tested"], "passed", r["passed"], r["period"], "split", r["split"])
    for t in r["table"][:15]:
        print(t["pass"], t["desc"], "|", {k: t["select"][k] for k in ("n", "rate", "exp", "avg_win", "avg_loss", "big_loss_rate", "worst1")},
              "|", {k: t["verify"][k] for k in ("n", "rate", "exp", "big_loss_rate", "worst1")})
    print("BEST", json.dumps(r["best"], ensure_ascii=False, default=str)[:600] if r["best"] else None)

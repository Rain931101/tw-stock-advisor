"""多數條件成立：條件表、門檻與判斷。"""
import numpy as np
import pandas as pd

from app.rules import features as F
from app.services import nearmatch as N


def _df(n=200):
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.02, n)))
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"open": c * 0.99, "high": c * 1.02, "low": c * 0.98, "close": c,
                         "volume": rng.integers(1000, 5000, n).astype(float)}, index=idx)


def test_setup_conditions_shape():
    tables, weak = F.setup_conditions(_df())
    assert set(tables) == set(F.SETUP_CONDS)
    for s, t in tables.items():
        assert list(t.columns) == list(F.SETUP_CONDS[s]) and t.dtypes.eq(bool).all()
    assert len(weak) == 200


def test_need():
    assert [N.need(k) for k in (3, 5, 6)] == [2, 4, 4]


def test_assess_buy_and_fallback(monkeypatch):
    conds = {k: True for k in F.SETUP_CONDS["ma20_ride"]} | {"hold": False}
    setups = {s: {k: False for k in c} for s, c in F.SETUP_CONDS.items()} | {"ma20_ride": conds}
    monkeypatch.setattr(N, "today", lambda: {"1101": {"weak": False, "bias": True, "setups": setups}})
    good = {"rate": 0.55, "exp": 1.5, "train_rate": 0.55, "test_rate": 0.56, "train_exp": 1.2, "test_exp": 1.8, "n": 300,
            "reliable_buy": True, "reliable_avoid": False}
    m = {"params": {}, "trained_at": "x", "stats": {"ma20_ride|hold|any": good}, "extra": []}
    a = N.assess("1101", m, feats={})
    assert a["best"]["setup"] == "ma20_ride" and a["best"]["missing"] == ["收在月線之上"]
    assert "缺同樣條件" in a["best"]["basis"]
    # 沒有統計時退回「樣本不足」，不建議買進
    a = N.assess("1101", {**m, "stats": {}}, feats={})
    assert a["best"] is None and a["items"][0]["decision"] == "unknown"


def test_decision_basis_single_source():
    from app.services.recommender import decision_basis
    st = {"rate": 0.51, "exp": 1.05, "n": 367, "train_rate": 0.51, "test_rate": 0.52, "train_exp": 1.03, "test_exp": 1.12}
    best = {"name": "沿月線", "met": ["a", "b", "c", "d"], "total": 5, "basis": "x", "stats": st}
    wr = {"decision": "neutral", "rate": 0.44, "exp": 0.25, "rule": {"names": ["q"], "n": 10}, "params": {"buy_exp": 1}}
    v = {"decision": "buy", "plan": {"setup": "nearmatch"}, "winrate": wr, "near": {"best": best, "items": [best]}}
    b = decision_basis(v)
    assert b["source"] == "nearmatch" and b["exp"] == 1.05 and b["pass"]
    # 勝率模型判定不推薦時，以勝率模型為依據
    v = {"decision": "avoid", "plan": None, "winrate": {**wr, "decision": "avoid", "exp": -0.5}, "near": {"best": best, "items": [best]}}
    assert decision_basis(v)["source"] == "winrate"

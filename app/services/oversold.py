"""全面超跌搶反彈提示（兩階段）。

回測發現（5 年、約 600 檔流動性足夠的股票；隔日開盤買、抱 10 個交易日收盤賣、扣手續費與證交稅）：
  - 個股「自己」急跌（當天只有少數股票超跌）時搶反彈沒有優勢：平均 ≈ 0、約 1 成交易虧超過 10%（多半是有利空）。
  - 「全面超跌」——超過 15% 的股票同時低於月線 10% 以上——時，恐慌賣壓錯殺好股，平均報酬明顯為正。
  - 但全面超跌的「第一天」進場平均是虧的（還在殺），要等超跌比例比前一天低（恐慌開始緩解）再進場，
    平均報酬更高、大虧比例更低，而且每一年都不是負的。

所以分兩階段：
  1. 全面超跌（觀察）：超跌比例 ≥ 15% 且還在增加 → 先不要接刀。
  2. 恐慌緩解（可分批進場）：仍 ≥ 15%，但比前一天低 → 發通知，列出基本面 A／B 且超跌最深的股票。
個股獨自急跌不給訊號。

注意：回測股票池是「現在仍在交易、流動性足夠」的股票，當年跌下去就沒再起來的股票不在裡面，結果會偏樂觀。
"""
import logging

import numpy as np
import pandas as pd

from app import db
from app.services import notifier
from app.services.winrate import ROUND_TRIP_COST, _panel

log = logging.getLogger(__name__)

BIAS = -0.10       # 收盤低於月線 10% 以上 = 超跌
TRIGGER = 0.15     # 超過 15% 的股票同時超跌 = 全面超跌
HOLD = 10          # 持有交易日數
MIN_VOL, MIN_PRICE = 1000, 10
BIG_LOSS = -0.10
LONE = 0.02        # 對照組：超跌股票不到 2%（個股獨自急跌）


def _wide(panel: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = pd.DataFrame({s: g["close"] for s, g in panel.items()}).sort_index()
    opens = pd.DataFrame({s: g["open"] for s, g in panel.items()}).reindex(closes.index)
    return closes, opens


def breadth(closes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """每檔每天與月線的乖離，以及每天超跌股票的比例（當天至少 50 檔有月線才算）。"""
    bias = closes / closes.rolling(20).mean() - 1
    valid = bias.notna().sum(axis=1)
    share = (bias < BIAS).sum(axis=1) / valid.where(valid >= 50)
    return bias, share


def stage(share: pd.Series) -> pd.Series:
    """每天的階段：none／watch（全面超跌、比例還在增加）／entry（全面超跌、比例比前一天低）。"""
    trig = share >= TRIGGER
    easing = trig & (share < share.shift(1))
    return pd.Series(np.where(easing, "entry", np.where(trig, "watch", "none")), index=share.index)


def _summary(r: pd.Series) -> dict:
    if r.empty:
        return {"n": 0, "days": 0}
    per_day = r.groupby(level=0).mean()
    return {"n": int(len(r)), "days": int(per_day.size), "avg": round(float(r.mean()) * 100, 2),
            "median": round(float(r.median()) * 100, 2), "win": round(float((r > 0).mean()), 4),
            "big_loss": round(float((r < BIG_LOSS).mean()), 4), "worst": round(float(r.min()) * 100, 2),
            "per_day_avg": round(float(per_day.mean()) * 100, 2), "day_win": round(float((per_day > 0).mean()), 4)}


def backtest(closes: pd.DataFrame, opens: pd.DataFrame) -> dict:
    bias, share = breadth(closes)
    ret = closes.shift(-HOLD) / opens.shift(-1) - 1 - ROUND_TRIP_COST
    st = stage(share)
    over = bias < BIAS

    def pick(day_mask: pd.Series) -> pd.Series:
        return ret.where(over & day_mask.values[:, None]).stack().dropna()

    trig = st != "none"
    entry = pick(st == "entry")
    years = entry.groupby(entry.index.get_level_values(0).year)
    out = {
        "period": [str(closes.index[0].date()), str(closes.index[-1 - HOLD].date())],
        "entry": _summary(entry),                                            # 恐慌緩解日（建議進場）
        "all": _summary(pick(trig)),                                         # 所有全面超跌日
        "first": _summary(pick(trig & ~trig.shift(1, fill_value=False))),    # 全面超跌第一天
        "lone": _summary(pick(share < LONE)),                                # 個股獨自急跌
        "years": [{"year": int(y), "avg": round(float(x.mean()) * 100, 2), "n": int(len(x))} for y, x in years],
    }
    # 每一次崩跌（觸發日間隔 ≤ 5 個交易日算同一次）
    days = list(st.index[trig])
    pos = {d: i for i, d in enumerate(closes.index)}
    episodes, cur = [], []
    for d in days:
        if cur and pos[d] - pos[cur[-1]] > 5:
            episodes.append(cur)
            cur = []
        cur.append(d)
    if cur:
        episodes.append(cur)
    eps = []
    for e in episodes:
        x = entry[entry.index.get_level_values(0).isin(e)]
        f = ret.loc[e[0]].where(over.loc[e[0]]).dropna()
        eps.append({"start": str(e[0].date()), "end": str(e[-1].date()), "days": len(e),
                    "peak_share": round(float(share.loc[e].max()), 4),
                    "entry_days": int(x.index.get_level_values(0).nunique()) if len(x) else 0,
                    "entry_avg": round(float(x.mean()) * 100, 2) if len(x) else None,
                    "first_avg": round(float(f.mean()) * 100, 2) if len(f) else None})
    out["episodes"] = eps[::-1]
    return out


def stats(force: bool = False) -> dict:
    """歷史統計（每個交易日重算一次並快取）。"""
    last = db.one("SELECT MAX(date) AS d FROM prices")["d"]
    key = f"oversold_stats:v2:{last}"
    hit = None if force else db.cache_get(key)
    if hit:
        return hit
    panel, _ = _panel(MIN_VOL, MIN_PRICE)
    out = backtest(*_wide(panel))
    db.cache_set(key, out)
    return out


def today() -> dict:
    """最新交易日：超跌比例、階段、超跌最深的股票（基本面 A／B 排前面）。"""
    panel, industry = _panel(MIN_VOL, MIN_PRICE, days=200)
    closes, _ = _wide(panel)
    bias, share = breadth(closes)
    st = stage(share)
    day = closes.index[-1]
    s = float(share.iloc[-1]) if pd.notna(share.iloc[-1]) else None
    b = bias.iloc[-1].dropna()
    b = b[b < BIAS].sort_values()
    names = {r["stock_id"]: r["name"] for r in db.query("SELECT stock_id, name FROM stocks")}
    cands = []
    for sid, v in b.items():
        f = db.cache_get(f"fundamental:{sid}") or {}
        c = closes[sid].dropna()
        cands.append({"stock_id": sid, "name": names.get(sid, ""), "industry": industry.get(sid, ""),
                      "close": float(c.iloc[-1]), "bias20": round(float(v) * 100, 2),
                      "chg5": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2) if len(c) > 6 else None,
                      "grade": f.get("grade"), "score": f.get("score")})
    good = [x for x in cands if x["grade"] in ("A", "B")]
    rest = [x for x in cands if x["grade"] not in ("A", "B")]
    prev = share.dropna()
    return {"date": str(day.date()), "share": round(s, 4) if s is not None else None,
            "prev_share": round(float(prev.iloc[-2]), 4) if len(prev) > 1 else None,
            "stage": st.iloc[-1], "n_oversold": int(len(b)), "n_stocks": int(bias.iloc[-1].notna().sum()),
            "threshold": {"bias": round(BIAS * 100), "share": TRIGGER, "hold": HOLD},
            "recent": [{"date": str(d.date()), "share": round(float(v), 4), "stage": st.loc[d]}
                       for d, v in prev.tail(15).items()],
            "candidates": (good + rest)[:20], "n_good": len(good)}


def check_and_alert() -> dict:
    """收盤資料更新後呼叫：恐慌緩解（可進場）發高優先通知；全面超跌剛開始發觀察通知。同一天各只發一次。"""
    t = today()
    if t["stage"] == "none":
        return {"date": t["date"], "share": t["share"], "stage": "none"}
    st = stats()
    pct = round(t["share"] * 100)
    if t["stage"] == "entry":
        e = st["entry"]
        top = "、".join(f"{x['stock_id']} {x['name']}（{x['bias20']}%）" for x in t["candidates"] if x["grade"] in ("A", "B"))
        top = "、".join(top.split("、")[:5])
        notifier.alert(
            "market", "恐慌緩解：全面超跌搶反彈",
            f"{t['date']} 仍有 {pct}% 的股票低於月線 10% 以上，但比前一天（{round(t['prev_share'] * 100)}%）少，恐慌開始緩解。"
            f"歷史上這種日子隔日開盤買超跌股、抱 {HOLD} 個交易日：平均 {e['avg']:+}%、勝率 {round(e['win'] * 100)}%，"
            f"但仍有 {round(e['big_loss'] * 100)}% 的交易虧超過 10%，部位要縮小、分批。"
            + (f"基本面 A／B 且超跌最深：{top}" if top else ""),
            dedup_key=f"oversold:entry:{t['date']}", level="high")
    else:
        f = st["first"]
        notifier.alert(
            "market", "全面超跌：先觀察，不要接刀",
            f"{t['date']} 有 {pct}% 的股票低於月線 10% 以上，而且還在增加。"
            f"歷史上全面超跌第一天就進場，抱 {HOLD} 天平均 {f['avg']:+}%、{round(f['big_loss'] * 100)}% 的交易虧超過 10%；"
            "等超跌比例開始下降（恐慌緩解）會再通知。",
            dedup_key=f"oversold:watch:{t['date']}", level="info")
    return {"date": t["date"], "share": t["share"], "stage": t["stage"]}

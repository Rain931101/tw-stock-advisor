"""把講義的技術條件編成每日的「是／否」特徵（向量化，供勝率訓練與即時比對共用）。

只用當天收盤（含）以前的資料計算，避免偷看未來。
"""
import numpy as np
import pandas as pd

# 特徵代碼 → 中文名稱（講義用語）
FEATURES = {
    "above_ma5": "站上 5 日線",
    "above_ma20": "站上月線",
    "above_ma60": "站上季線",
    "bull_align": "均線多頭排列（5>10>20>60）",
    "ma_converge": "短中長期均線靠攏",
    "ma60_up": "季線上揚",
    "vol_up": "量增（量比 ≥ 1.2）",
    "vol_burst": "爆量（≥ 2 倍 20 日均量）",
    "vol_steady": "穩定放量（5 日均量 > 20 日均量）",
    "vol_dry": "量縮（< 0.7 倍 20 日均量）",
    "red_k": "收紅 K",
    "long_red": "長紅（漲幅 ≥ 4%）",
    "high_20": "創 20 日新高",
    "high_60": "創 60 日新高",
    "high_120": "創 120 日新高（擁抱創高股）",
    "touch_ma5": "回測五日線有撐",
    "touch_ma20": "回測月線有撐",
    "reclaim_ma60": "跌破季線後站回",
    "reclaim_ma5": "站回五日線",
    "bias5_high": "與五日線乖離 > 8%",
    "angle_trend": "角度 30~60 度（趨勢盤）",
    "angle_steep": "角度 > 60 度（急拉盤）",
    "angle_range": "角度 −30~30 度（震盪盤）",
    "below_ma20": "跌破月線",
    "climax_prev": "前一日爆量長黑",
    "box_breakout": "突破 60 日整理區上緣",
    "sector_strong": "所屬族群今日強勢（前 5 名）",
    "market_bull": "市場多頭（過半股票站上月線）",
}


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """df：單一股票日 K（index=date；open/high/low/close/volume）。回傳同 index 的布林特徵表。"""
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60)}
    vma5, vma20 = v.rolling(5).mean(), v.rolling(20).mean()
    chg = c.pct_change()
    logc = np.log(c)
    # 20 日對數回歸斜率 → 角度（與 indicators.trend_angle 同基準：日均 0.5% = 45 度）
    x = np.arange(20) - 9.5
    slope = logc.rolling(20).apply(lambda y: np.dot(x, y - y.mean()) / np.dot(x, x), raw=True)
    angle = np.degrees(np.arctan(slope * 100 / 0.5))
    box_top = df[["open", "close"]].max(axis=1).shift(1).rolling(60).quantile(0.95)
    box_bot = l.shift(1).rolling(60).quantile(0.03)
    f = pd.DataFrame(index=df.index)
    f["above_ma5"] = c > ma[5]
    f["above_ma20"] = c > ma[20]
    f["above_ma60"] = c > ma[60]
    f["bull_align"] = (ma[5] > ma[10]) & (ma[10] > ma[20]) & (ma[20] > ma[60])
    spread = pd.concat([ma[5], ma[20], ma[60]], axis=1)
    f["ma_converge"] = (spread.max(axis=1) - spread.min(axis=1)) / c < 0.06
    f["ma60_up"] = ma[60] > ma[60].shift(20)
    f["vol_up"] = v >= v.shift(1) * 1.2
    f["vol_burst"] = v >= vma20 * 2
    f["vol_steady"] = vma5 > vma20
    f["vol_dry"] = v < vma20 * 0.7
    f["red_k"] = c > o
    f["long_red"] = chg >= 0.04
    f["high_20"] = h >= h.rolling(20).max()
    f["high_60"] = h >= h.rolling(60).max()
    f["high_120"] = h >= h.rolling(120, min_periods=120).max()
    f["touch_ma5"] = (l <= ma[5] * 1.015) & (c >= ma[5]) & (c > ma[20])
    f["touch_ma20"] = (l <= ma[20] * 1.02) & (c >= ma[20])
    f["reclaim_ma60"] = (c > ma[60]) & (c.shift(1) < ma[60].shift(1)).rolling(5).max().astype(bool)
    f["reclaim_ma5"] = (c > ma[5]) & (c.shift(1) <= ma[5].shift(1))
    f["bias5_high"] = c / ma[5] - 1 > 0.08
    f["angle_trend"] = (angle >= 30) & (angle < 60)
    f["angle_steep"] = angle >= 60
    f["angle_range"] = (angle > -30) & (angle < 30)
    f["below_ma20"] = c < ma[20]
    f["climax_prev"] = ((v.shift(1) >= vma20.shift(2) * 2.5) & (chg.shift(1) <= -0.04)) & (c > c.shift(1))
    f["box_breakout"] = (c > box_top) & ((box_top / box_bot) < 1.3)
    valid = ma[60].notna() & vma20.notna() & angle.notna()
    return f.where(valid, other=False).astype(bool).assign(_valid=valid)


def cross_section(panel: dict[str, pd.DataFrame], industry: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """跨股票的每日特徵：市場多頭比例、族群強弱。回傳 (market_bull: date→bool, sector_rank: date×industry→rank)。"""
    closes = pd.DataFrame({s: d["close"] for s, d in panel.items()})
    values = pd.DataFrame({s: d.get("value", d["close"] * d["volume"]) for s, d in panel.items()})
    above20 = (closes > closes.rolling(20).mean()).mean(axis=1)
    market_bull = above20 > 0.5
    chg = closes.pct_change()
    ind = pd.Series(industry).reindex(closes.columns).fillna("其他")
    w = values.div(values.T.groupby(ind).transform("sum").T)
    sector_chg = (chg * w).T.groupby(ind).sum().T  # date × industry
    sector_rank = sector_chg.rank(axis=1, ascending=False)
    return market_bull, sector_rank

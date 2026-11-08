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


# ================= 講義六種進場型態的條件（向量化，供「多數條件成立」回測） =================
# 與 technical.py 各型態的條件一一對應；年線資料不足時以半年線代替（價格資料約一年）。
SETUP_CONDS = {
    "breakout": {"box": "先有 40 天以上整理區間（箱型）", "break": "帶量突破前高（收盤站上箱頂）", "vol_up": "量增（量比 ≥ 1.2）",
                 "vol_steady": "穩定放量（5 日均量 > 20 日均量）", "above_all": "站上所有短中期均線", "converge": "短中長期均線靠攏"},
    "ma60_reclaim": {"broke": "近 5 日內曾跌破季線", "reclaim": "今日強勢站回季線", "above5": "站上五日線",
                     "ma60_up": "季線仍上揚（強勢拉回而非做頭）", "above_year": "位於年線之上"},
    "ma5_pullback": {"above_all": "站在所有均線之上", "align": "均線多頭排列", "new_high": "近 20 日創波段新高",
                     "angle": "角度 ≥ 30° 屬趨勢／急拉盤", "touch": "回測五日線", "hold": "五日線收腳有撐"},
    "ma20_ride": {"ride": "近 60 日 85% 時間沿月線之上", "angle60": "60 日角度落在 30~60 度", "touch": "回測月線附近",
                  "hold": "收在月線之上", "ma60_up": "季線上揚"},
    "rebound_ma5": {"above_year": "在年線之上做單", "broke20": "前一段明顯修正（曾跌破月線）", "reclaim5": "今日站上五日線（昨日在五日線下）"},
    "climax_bottom": {"climax": "前一日回檔爆量（≥2.5 倍均量、跌幅 ≥4%）", "follow": "第二根延續：收紅且守住爆量 K 低點",
                      "support": "在季線之上"},
}


def _angle(logc: pd.Series, n: int) -> pd.Series:
    t = pd.Series(np.arange(len(logc), dtype=float), index=logc.index)
    slope = t.rolling(n).cov(logc) / t.rolling(n).var()
    return np.degrees(np.arctan(slope * 100 / 0.5))


def setup_conditions(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """回傳 ({型態: 條件布林表}, 弱位階布林)；只用當天（含）以前的資料。"""
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60, 120, 240)}
    year = ma[240].fillna(ma[120])
    vma5, vma20 = v.rolling(5).mean(), v.rolling(20).mean()
    chg = c.pct_change()
    pc, pma5 = c.shift(1), ma[5].shift(1)
    above_all = (c > ma[5]) & (c > ma[10]) & (c > ma[20]) & (c > ma[60])
    spread = pd.concat([ma[5], ma[20], ma[60]], axis=1)
    conv_prev = ((spread.max(axis=1) - spread.min(axis=1)) / c).shift(1) < 0.08
    box_top = df[["open", "close"]].max(axis=1).shift(1).rolling(40).max()
    box_bot = l.shift(1).rolling(40).min()
    ma60_up = ma[60] >= ma[60].shift(20)
    logc = np.log(c)
    out = {
        "breakout": {"box": box_top / box_bot < 1.25, "break": (c > box_top) & (pc <= box_top * 1.02),
                     "vol_up": v >= v.shift(1) * 1.2, "vol_steady": vma5 > vma20, "above_all": above_all, "converge": conv_prev},
        "ma60_reclaim": {"broke": (c < ma[60]).shift(1).rolling(5).max().fillna(0).astype(bool), "reclaim": c > ma[60],
                         "above5": c > ma[5], "ma60_up": ma60_up, "above_year": c > year},
        "ma5_pullback": {"above_all": (c > ma[10]) & (c > ma[20]) & (c > ma[60]),
                         "align": ((ma[5] > ma[10]) & (ma[10] > ma[20]) & (ma[20] > ma[60])).shift(1).fillna(False).astype(bool),
                         "new_high": h.rolling(20).max() >= h.rolling(120, min_periods=60).max() * 0.999,
                         "angle": _angle(logc, 20) >= 30, "touch": l <= ma[5] * 1.015, "hold": c >= ma[5]},
        "ma20_ride": {"ride": (c > ma[20] * 0.99).astype(float).rolling(60).mean() >= 0.85,
                      "angle60": _angle(logc, 60).between(25, 65), "touch": l <= ma[20] * 1.02, "hold": c >= ma[20],
                      "ma60_up": ma[60] > ma[60].shift(20)},
        "rebound_ma5": {"above_year": c > year, "broke20": (c < ma[20]).shift(1).rolling(9).max().fillna(0).astype(bool),
                        "reclaim5": (c > ma[5]) & (pc <= pma5)},
        "climax_bottom": {"climax": (v.shift(1) >= vma20.shift(2) * 2.5) & (chg.shift(1) <= -0.04),
                          "follow": (c > pc) & (l >= l.shift(1)), "support": c > ma[60]},
    }
    valid = ma[60].notna() & vma20.notna()
    tables = {s: pd.DataFrame(conds).fillna(False).astype(bool).where(valid, False) for s, conds in out.items()}
    weak = (c < year) & (ma[60] < year)
    return tables, weak.fillna(False)

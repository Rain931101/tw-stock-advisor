"""技術指標：均線、量比、轉折點、箱型、角度、格局、波段目標價。

所有函式都吃日 K DataFrame（index=date，欄位 open/high/low/close/volume，volume 單位：張）。
"""
from dataclasses import dataclass
from datetime import datetime, time

import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 20, 60, 120, 240)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in MA_WINDOWS:
        df[f"ma{n}"] = df["close"].rolling(n, min_periods=n).mean()
    df["vma5"] = df["volume"].rolling(5).mean()
    df["vma20"] = df["volume"].rolling(20).mean()
    df["chg"] = df["close"].pct_change()
    return df


# ---------------- 量比（講義：量比 = 估計量／昨量） ----------------
# 台股盤中累積成交量的典型分布（經驗值），用來把「目前累積量」換算成「估計全日量」
_VOL_CURVE = [
    (time(9, 0), 0.0), (time(9, 5), 0.09), (time(9, 30), 0.26), (time(10, 0), 0.39),
    (time(11, 0), 0.55), (time(12, 0), 0.68), (time(13, 0), 0.83), (time(13, 25), 0.93), (time(13, 30), 1.0),
]


def day_fraction(t: time) -> float:
    secs = lambda x: x.hour * 3600 + x.minute * 60 + x.second  # noqa: E731
    s = secs(t)
    if s <= secs(_VOL_CURVE[0][0]):
        return 0.0
    for (t0, f0), (t1, f1) in zip(_VOL_CURVE, _VOL_CURVE[1:]):
        if s <= secs(t1):
            return f0 + (f1 - f0) * (s - secs(t0)) / (secs(t1) - secs(t0))
    return 1.0


def estimated_volume(cum_volume: float, at: datetime | time) -> float:
    t = at.time() if isinstance(at, datetime) else at
    frac = max(day_fraction(t), 0.03)
    return cum_volume / frac


def volume_ratio(cum_volume: float, yesterday_volume: float, at: datetime | time) -> float | None:
    if not yesterday_volume:
        return None
    return estimated_volume(cum_volume, at) / yesterday_volume


# ---------------- 轉折點與格局 ----------------
def swing_points(df: pd.DataFrame, k: int = 5) -> tuple[list[tuple[pd.Timestamp, float]], list[tuple[pd.Timestamp, float]]]:
    """左右各 k 根內的最高 / 最低點視為轉折高 / 低。"""
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    for i in range(k, len(df) - k):
        if h[i] == max(h[i - k : i + k + 1]):
            highs.append((df.index[i], float(h[i])))
        if l[i] == min(l[i - k : i + k + 1]):
            lows.append((df.index[i], float(l[i])))
    return highs, lows


def structure(df: pd.DataFrame, k: int = 5) -> str:
    """講義〈格局轉換順序要對〉：
    多轉空要先看到「低過低」；空轉多要先看到「高過高」。
    回傳 uptrend / downtrend / turning_up / turning_down / range
    """
    highs, lows = swing_points(df, k)
    if len(highs) < 2 or len(lows) < 2:
        return "range"
    tol = 0.005
    (h1, h2), (l1, l2) = (highs[-2][1], highs[-1][1]), (lows[-2][1], lows[-1][1])
    last_close = float(df["close"].iloc[-1])
    hh, lh = h2 > h1 * (1 + tol), h2 < h1 * (1 - tol)
    hl, ll = l2 > l1 * (1 + tol), l2 < l1 * (1 - tol)
    if last_close < l2 * (1 - tol) and lh:
        ll = True  # 目前價已跌破最近低點 = 低過低
    if hh and hl:
        return "uptrend"
    if lh and ll:
        return "downtrend"
    if hh and not ll:
        return "turning_up"  # 先看到高過高，才猜可能反轉向上
    if ll and not hh:
        return "turning_down"  # 先看到低過低，才猜可能做頭
    return "range"


# ---------------- 箱型 / 整理 ----------------
@dataclass
class Box:
    start: pd.Timestamp
    end: pd.Timestamp
    top: float
    bottom: float
    days: int

    @property
    def height(self) -> float:
        return self.top - self.bottom


def find_box(df: pd.DataFrame, min_days: int = 20, max_ratio: float = 1.25, exclude_last: int = 1) -> Box | None:
    """往回找最長的一段「最高/最低 ≤ max_ratio」的整理區（不含最近 exclude_last 根）。

    箱頂以收盤價的高位（重疊最多處）近似，避免被上影線干擾——講義：上影線不管、重疊最多畫哪裡。
    """
    body = df.iloc[: len(df) - exclude_last] if exclude_last else df
    if len(body) < min_days:
        return None
    top_s = body[["open", "close"]].max(axis=1)
    bot_s = body["low"]
    best = None
    for days in range(min_days, min(len(body), 250) + 1):
        win_top = top_s.iloc[-days:]
        win_bot = bot_s.iloc[-days:]
        top = float(win_top.quantile(0.95))
        bottom = float(win_bot.quantile(0.03))
        if bottom <= 0 or top / bottom > max_ratio:
            break
        best = Box(body.index[-days], body.index[-1], top, bottom, days)
    return best


# ---------------- 角度（對數座標） ----------------
def trend_angle(df: pd.DataFrame, days: int = 20) -> float | None:
    """以對數價格線性回歸斜率換算「角度」。

    講義的角度需看圖表比例；這裡固定以「每日 0.5% 漲幅 = 45 度」為基準：
    ‑30~30 度 ≈ 日均漲跌 < 0.29%（震盪盤）、30~60 度 ≈ 0.29%~0.87%（趨勢盤）、>60 度（急拉盤）。
    """
    if len(df) < days:
        return None
    y = np.log(df["close"].iloc[-days:].values)
    x = np.arange(days)
    slope = np.polyfit(x, y, 1)[0]
    return float(np.degrees(np.arctan(slope * 100 / 0.5)))


def angle_regime(angle: float | None) -> str:
    if angle is None:
        return "unknown"
    if angle >= 60:
        return "急拉盤"
    if angle >= 30:
        return "趨勢盤"
    if angle > -30:
        return "震盪盤"
    return "下跌盤"


# ---------------- 位階 ----------------
def position_level(row: pd.Series) -> str:
    """強位階：站在季線、年線之上且季線上揚；弱位階：在年線之下。"""
    close = row["close"]
    if pd.isna(row.get("ma240")) or pd.isna(row.get("ma60")):
        return "未知"
    if close > row["ma60"] and close > row["ma240"] and row["ma60"] > row["ma240"]:
        return "強位階"
    if close < row["ma240"] and row["ma60"] < row["ma240"]:
        return "弱位階"
    return "中立位階"


# ---------------- 波段目標價（講義第四章） ----------------
@dataclass
class WaveTarget:
    rally_start: float  # 起漲點
    resistance: float  # 最關鍵壓力
    box_bottom: float  # 震盪底（關鍵支撐）
    target: float  # 1:1 複製上漲空間
    healthy_pullback: float  # 健康回檔價

    def extend(self, n: int = 1) -> float:
        """突破後可無限 1:1 往上延伸。"""
        return self.target + (self.resistance - self.rally_start) * n


def wave_target(df: pd.DataFrame, box: Box | None = None) -> WaveTarget | None:
    """1. 先找一段明顯急漲段，再找一段明顯的橫盤震盪
    2. 從最明顯上漲的位置抓起漲點、往上抓最關鍵壓力（=上漲段高點）
    3. 目標價 = 震盪底 + (壓力 − 起漲點)
    4. 健康回檔價 = 目標價 − (壓力 − 震盪底)
    """
    box = box or find_box(df, min_days=10, max_ratio=1.20)
    if box is None:
        return None
    before = df.loc[: box.start].iloc[:-1].tail(60)
    if len(before) < 10:
        return None
    rally_start = float(before["low"].min())
    resistance = max(box.top, float(df.loc[box.start : box.end, "high"].quantile(0.9)))
    height = resistance - rally_start
    if height <= 0 or rally_start <= 0 or resistance / rally_start < 1.15:  # 沒有明顯急漲段
        return None
    target = box.bottom + height
    healthy = target - (resistance - box.bottom)
    return WaveTarget(rally_start, resistance, box.bottom, target, healthy)


def m_top_target(neckline: float, top: float) -> float:
    """M 頭等幅：跌破頸線後的空方目標 = 頸線 − (頭 − 頸線)。講義：看空不做空，抵達後左側抓反彈。"""
    return neckline - (top - neckline)


# ---------------- 其他 ----------------
def is_garbage_time(df: pd.DataFrame, lookback: int = 20) -> bool:
    """講義〈垃圾時間〉：1. 創高壓回後破低拉回 2. 破低拉回後創高壓回 → 高機率繼續震盪，綁手不做。"""
    if len(df) < lookback + 20:
        return False
    win = df.iloc[-lookback:]
    prior = df.iloc[-lookback - 20 : -lookback]
    hi_idx = win["high"].idxmax()
    lo_idx = win["low"].idxmin()
    made_high = win["high"].max() > prior["high"].max()
    broke_low = win["low"].min() < prior["low"].min()
    if not (made_high and broke_low):
        return False
    last = float(df["close"].iloc[-1])
    # 目前價位卡在兩個極端的中間
    return win["low"].min() * 1.03 < last < win["high"].max() * 0.97 and hi_idx != lo_idx


def slow_bleed(df: pd.DataFrame, days: int = 10) -> bool:
    """量縮緩跌：最近 days 天收盤持續走低、低點不斷破、量能低於 20 日均量——不要做左側，容易接刀。"""
    if len(df) < days + 20:
        return False
    win = df.iloc[-days:]
    falling = win["close"].iloc[-1] < win["close"].iloc[0] * 0.97
    small_vol = win["volume"].mean() < df["vma20"].iloc[-days - 1] * 0.8
    no_crash = win["chg"].min() > -0.05  # 緩跌而不是急跌
    lower_lows = win["low"].iloc[-1] <= win["low"].iloc[: days // 2].min()
    return bool(falling and small_vol and no_crash and lower_lows)


def ma_converged(row: pd.Series, tol: float = 0.06) -> bool:
    """短中長期均線靠攏。"""
    vals = [row.get(f"ma{n}") for n in (5, 20, 60)]
    if any(v is None or pd.isna(v) for v in vals):
        return False
    return (max(vals) - min(vals)) / row["close"] < tol


def bullish_alignment(row: pd.Series) -> bool:
    vals = [row.get(f"ma{n}") for n in (5, 10, 20, 60)]
    if any(v is None or pd.isna(v) for v in vals):
        return False
    return vals[0] > vals[1] > vals[2] > vals[3]


def above_all_ma(row: pd.Series, windows=(5, 10, 20, 60)) -> bool:
    return all(not pd.isna(row.get(f"ma{n}")) and row["close"] > row[f"ma{n}"] for n in windows)

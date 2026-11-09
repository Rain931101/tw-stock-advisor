"""進場型態（setups）：把《可以用的技術分析》的實戰規則寫成可判斷的條件。

每個 setup 回傳 Signal，包含：觸發與否、理由、進場區間、停損價（進場要素消失的位置）、
目標價、停利／減碼規則（給盤中監控機器判讀）。

df 的最後一根可以是「盤中合成 K 棒」（用即時價組出來），搭配 now 參數套用盤中時間規則：
  9:00~9:05 垃圾時間不進場；9:30~10:00 主力第二波發動；12:00 後確認季線撐不撐；
  12:30 後還沒站回五日線就提前出場。
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, time

import pandas as pd

from app.rules import indicators as ind
from app.rules.risk import round_tick

SETUP_NAMES = {
    "breakout": "量增突破（整理後帶量過前高）",
    "ma60_reclaim": "季線站回（跌破季線後迅速站回）",
    "ma5_pullback": "強勢股回測五日線",
    "ma20_ride": "沿月線趨勢股回測月線",
    "rebound_ma5": "破線反彈短打（年線之上站回五日線）",
    "climax_bottom": "回檔爆量猜底（第二根延續）",
}


@dataclass
class Signal:
    setup: str
    name: str
    triggered: bool
    score: float = 0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # 尚未符合的條件
    entry_low: float | None = None
    entry_high: float | None = None
    stop: float | None = None
    stop_reason: str = ""
    targets: list[dict] = field(default_factory=list)
    healthy_pullback: float | None = None
    exit_rules: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Context:
    df: pd.DataFrame
    row: pd.Series
    prev: pd.Series
    now: datetime | None
    intraday: bool
    angle: float | None
    regime: str
    level: str
    struct: str
    box: ind.Box | None
    wave: ind.WaveTarget | None
    garbage: bool
    bleed: bool

    @property
    def t(self) -> time | None:
        return self.now.time() if self.now else None

    def after(self, hh: int, mm: int) -> bool:
        return (not self.intraday) or (self.t is not None and self.t >= time(hh, mm))


def build_context(df: pd.DataFrame, now: datetime | None = None, intraday: bool = False) -> Context | None:
    if len(df) < 70:
        return None
    df = ind.add_indicators(df)
    row, prev = df.iloc[-1], df.iloc[-2]
    angle = ind.trend_angle(df, 20)
    box = ind.find_box(df, min_days=20, max_ratio=1.25, exclude_last=1)
    return Context(
        df=df, row=row, prev=prev, now=now, intraday=intraday,
        angle=angle, regime=ind.angle_regime(angle), level=ind.position_level(row),
        struct=ind.structure(df.iloc[:-1]), box=box, wave=ind.wave_target(df),
        garbage=ind.is_garbage_time(df), bleed=ind.slow_bleed(df),
    )


def _vol_ratio(ctx: Context) -> float | None:
    """量比：盤中用估計量，盤後用實際量 / 昨量。"""
    vol, yvol = ctx.row["volume"], ctx.prev["volume"]
    if ctx.intraday and ctx.now and not ctx.row.get("vol_est"):
        return ind.volume_ratio(vol, yvol, ctx.now)   # 盤中累計量 → 推估全日量
    return vol / yvol if yvol else None


def _targets_above(ctx: Context, price: float) -> list[dict]:
    """多方目標：頸線／前高、月線、季線、半年線、波段目標價（取高於現價者，由近到遠）。"""
    cands = []
    highs, _ = ind.swing_points(ctx.df.iloc[:-1], 5)
    for _, h in reversed(highs[-6:]):
        if h > price * 1.02:
            cands.append({"price": h, "label": "頸線／前高壓力"})
            break
    for n, label in ((20, "月線"), (60, "季線"), (120, "半年線")):
        v = ctx.row.get(f"ma{n}")
        if v is not None and not pd.isna(v) and v > price * 1.02:
            cands.append({"price": float(v), "label": label})
    if ctx.wave and ctx.wave.target > price * 1.02:
        cands.append({"price": ctx.wave.target, "label": "波段目標價（1:1 等幅）"})
    cands.sort(key=lambda x: x["price"])
    seen, out = set(), []
    for c in cands:
        p = round_tick(c["price"])
        if p not in seen:
            seen.add(p)
            out.append({"price": p, "label": c["label"]})
    return out[:4]


def _common_warnings(ctx: Context) -> list[str]:
    w = []
    if ctx.garbage:
        w.append("垃圾時間：創高壓回後破低（或破低後創高），高機率持續震盪")
    if ctx.bleed:
        w.append("量縮緩跌：不要輕易左側承接，容易接刀")
    if ctx.level == "弱位階":
        w.append("弱位階：便宜≠風險小，越弱的位階資金越不愛")
    if ctx.struct == "downtrend":
        w.append("格局：低過低、高不過高，空方格局")
    if ctx.row["close"] / ctx.row["ma5"] - 1 > 0.08:
        w.append("與五日線乖離過大（>8%），容易兩天吃 10% 以上回檔，等回測五日線")
    if ctx.intraday and ctx.t and ctx.t < time(9, 5):
        w.append("開盤前五分鐘是垃圾時間，綁手觀察")
    return w


def _ma5_trailing(action="sell_all", label="收盤跌破五日線出場（移動停利）"):
    return {"type": "close_below_ma", "ma": 5, "action": action, "label": label, "intraday_after": "12:30"}


# ======================= 各型態 =======================
def setup_breakout(ctx: Context) -> Signal:
    r, s = ctx.row, Signal("breakout", SETUP_NAMES["breakout"], False)
    box = ctx.box
    if box is None or box.days < 40:
        s.missing = ["先要有 40 天以上的整理區間（箱型）"]
        return s
    vr = _vol_ratio(ctx) or 0
    conds = {
        f"整理 {box.days} 天（箱頂 {box.top:.2f}／箱底 {box.bottom:.2f}）": True,
        "帶量突破前高（收盤站上箱頂）": r["close"] > box.top and ctx.prev["close"] <= box.top * 1.02,
        f"量增（量比 {vr:.2f} ≥ 1.2）": vr >= 1.2,
        "穩定放量（5 日均量 > 20 日均量）": r["vma5"] > r["vma20"],
        "站上所有短中期均線": ind.above_all_ma(r, (5, 10, 20, 60)),
        "短中長期均線靠攏": ind.ma_converged(ctx.prev, 0.08),
    }
    s.reasons = [k for k, v in conds.items() if v]
    s.missing = [k for k, v in conds.items() if not v]
    if box.days >= 120:
        s.reasons.append("整理半年以上：水平力道 = 垂直力道，發動力道強")
    if r["close"] >= ctx.df["high"].iloc[-250:].max() * 0.999:
        s.reasons.append("創一年新高：左側無壓力、籌碼單純（擁抱創高股）")
    s.triggered = all(conds.values()) and ctx.after(9, 30)
    s.entry_low = round_tick(box.top)
    s.entry_high = round_tick(min(r["close"] * 1.03, r["ma5"] * 1.06))
    s.stop = round_tick(min(box.top * 0.97, r["ma10"]))
    s.stop_reason = "跌回箱型內（假突破），進場要素消失"
    wave_like = ind.WaveTarget(box.bottom - box.height, box.top, box.bottom, box.top + box.height, box.top)
    s.targets = [{"price": round_tick(box.top + box.height), "label": "箱型 1:1 等幅目標"},
                 {"price": round_tick(box.top + 2 * box.height), "label": "突破後再延伸 1:1"}]
    s.healthy_pullback = round_tick(wave_like.healthy_pullback)
    s.exit_rules = [
        {"type": "price_below", "level": s.stop, "action": "sell_all", "label": "跌回箱內／破停損價"},
        _ma5_trailing("sell_half", "收盤跌破五日線先減碼一半（爆量黑K沒破五日線不用多想）"),
        {"type": "target", "level": s.targets[0]["price"], "action": "watch_pullback", "label": "目標價滿足，盤中壓回轉弱即可提前出場"},
        {"type": "ma_below", "ma": 20, "action": "sell_all", "label": "跌破月線，突破失敗出場"},
    ]
    s.score = 60 + min(box.days, 250) / 10 + (10 if vr > 1.5 else 0)
    return s


def setup_ma60_reclaim(ctx: Context) -> Signal:
    df, r = ctx.df, ctx.row
    s = Signal("ma60_reclaim", SETUP_NAMES["ma60_reclaim"], False)
    recent = df.iloc[-6:-1]
    broke = (recent["close"] < recent["ma60"]).any()
    ma60_up = r["ma60"] >= df["ma60"].iloc[-21]
    conds = {
        "近 5 日內曾跌破季線": bool(broke),
        "今日強勢站回季線": r["close"] > r["ma60"],
        "站上五日線": r["close"] > r["ma5"],
        "季線仍上揚（強勢拉回而非做頭）": bool(ma60_up),
        "位於年線之上": not pd.isna(r["ma240"]) and r["close"] > r["ma240"],
    }
    s.reasons = [k for k, v in conds.items() if v]
    s.missing = [k for k, v in conds.items() if not v]
    # 盤中：打到季線要觀察 12:00 後有無撐住
    s.triggered = all(conds.values()) and ctx.after(12, 0)
    key_low = float(df["low"].iloc[-15:].min())
    s.stop = round_tick(key_low * 0.995)
    s.stop_reason = "跌破這波回檔的關鍵低點，行情容易轉弱或震盪"
    s.entry_low = round_tick(r["ma60"])
    s.entry_high = round_tick(r["close"] * 1.02)
    s.targets = _targets_above(ctx, r["close"])
    s.exit_rules = [
        {"type": "ma_below", "ma": 60, "action": "sell_half", "label": "再次跌破季線先砍 50%"},
        {"type": "price_below", "level": s.stop, "action": "sell_all", "label": "跌破關鍵價全部出場"},
        {"type": "stall_ma", "ma": 60, "days": 5, "action": "sell_all", "label": "進場後一直黏在季線、沒有轉強就出場"},
    ]
    if s.targets:
        s.exit_rules.append({"type": "target", "level": s.targets[0]["price"], "action": "watch_pullback",
                             "label": f"抵達第一目標（{s.targets[0]['label']}）"})
    s.score = 70 + (10 if ind.bullish_alignment(r) else 0)
    return s


def setup_ma5_pullback(ctx: Context) -> Signal:
    df, r = ctx.df, ctx.row
    s = Signal("ma5_pullback", SETUP_NAMES["ma5_pullback"], False)
    touched = r["low"] <= r["ma5"] * 1.015
    new_high_20 = df["high"].iloc[-20:].max() >= df["high"].iloc[-120:].max() * 0.999
    conds = {
        "站在所有均線之上（5 日線思維轉積極）": ind.above_all_ma(r, (10, 20, 60)),
        "均線多頭排列": ind.bullish_alignment(ctx.prev),
        "近 20 日創波段新高": bool(new_high_20),
        f"角度 {ctx.angle:.0f}° 屬趨勢／急拉盤": ctx.angle is not None and ctx.angle >= 30,
        "回測五日線": bool(touched),
        "五日線收腳有撐": r["close"] >= r["ma5"],
    }
    s.reasons = [k for k, v in conds.items() if v]
    s.missing = [k for k, v in conds.items() if not v]
    # 9:30 前的壓回多是洗盤，9:30 後再判斷
    s.triggered = all(conds.values()) and ctx.after(9, 30)
    s.entry_low = round_tick(r["ma5"] * 0.995)
    s.entry_high = round_tick(r["ma5"] * 1.02)
    s.stop = round_tick(min(r["ma5"], r["low"]) * 0.985)
    s.stop_reason = "停損守五日線：日 K 收破五日線即進場要素消失"
    s.targets = _targets_above(ctx, r["close"]) or [{"price": round_tick(r["close"] * 1.1), "label": "前波漲幅延伸"}]
    s.exit_rules = [
        _ma5_trailing("sell_all", "收盤跌破五日線出場；不想承擔隔日風險則 12:30 後仍未站回就先出"),
        {"type": "price_below", "level": s.stop, "action": "sell_all", "label": "跌破停損價"},
        {"type": "target", "level": s.targets[0]["price"], "action": "watch_pullback", "label": "目標滿足，盤中壓回即提前出場"},
    ]
    s.score = 65 + (10 if ctx.regime == "趨勢盤" else 0)
    return s


def setup_ma20_ride(ctx: Context) -> Signal:
    df, r = ctx.df, ctx.row
    s = Signal("ma20_ride", SETUP_NAMES["ma20_ride"], False)
    last60 = df.iloc[-60:]
    ride_ratio = float((last60["close"] > last60["ma20"] * 0.99).mean())
    a60 = ind.trend_angle(df, 60)
    conds = {
        f"近 60 日 {ride_ratio:.0%} 時間沿月線之上": ride_ratio >= 0.85,
        f"60 日角度 {a60 or 0:.0f}° 落在 30~60 度（走最遠最穩）": a60 is not None and 25 <= a60 <= 65,
        "回測月線附近": r["low"] <= r["ma20"] * 1.02,
        "收在月線之上": r["close"] >= r["ma20"],
        "季線上揚": r["ma60"] > df["ma60"].iloc[-21],
    }
    s.reasons = [k for k, v in conds.items() if v]
    s.missing = [k for k, v in conds.items() if not v]
    s.triggered = all(conds.values()) and ctx.after(12, 0)
    s.entry_low = round_tick(r["ma20"] * 0.99)
    s.entry_high = round_tick(r["ma20"] * 1.03)
    s.stop = round_tick(r["ma60"] * 0.99)
    s.stop_reason = "確定基本面後不要驗牌直接蓋牌，季線跌破再出場"
    s.targets = _targets_above(ctx, r["close"]) or [{"price": round_tick(df["high"].iloc[-60:].max()), "label": "前高"}]
    s.exit_rules = [
        {"type": "ma_below", "ma": 60, "action": "sell_all", "label": "跌破季線出場（整理完站回月線再進）"},
        {"type": "close_below_ma", "ma": 20, "action": "notify", "label": "跌破月線：觀察短線是否快速拉回（強勁訊號）"},
    ]
    s.score = 72
    return s


def setup_rebound_ma5(ctx: Context) -> Signal:
    df, r, p = ctx.df, ctx.row, ctx.prev
    s = Signal("rebound_ma5", SETUP_NAMES["rebound_ma5"], False)
    conds = {
        "在年線之上做單": not pd.isna(r["ma240"]) and r["close"] > r["ma240"],
        "前一段明顯修正（曾跌破月線）": bool((df["close"].iloc[-10:-1] < df["ma20"].iloc[-10:-1]).any()),
        "今日站上五日線（昨日在五日線下）": r["close"] > r["ma5"] and p["close"] <= p["ma5"],
    }
    s.reasons = [k for k, v in conds.items() if v]
    s.missing = [k for k, v in conds.items() if not v]
    s.stop = round_tick(float(p["low"]))
    s.stop_reason = "站穩五日線的前一天低點"
    s.targets = _targets_above(ctx, r["close"])
    s.entry_low = round_tick(r["ma5"])
    s.entry_high = round_tick(r["close"] * 1.015)
    room_ok = False
    if s.targets and r["close"] > s.stop:
        rr = (s.targets[0]["price"] - r["close"]) / (r["close"] - s.stop)
        room_ok = rr >= 1.5
        (s.reasons if room_ok else s.warnings).append(
            f"操作空間：到第一目標的報酬／風險 = {rr:.1f}" + ("" if room_ok else "（空間不夠理想）"))
    s.triggered = all(conds.values()) and room_ok and ctx.after(12, 0)
    s.exit_rules = [
        {"type": "price_below", "level": s.stop, "action": "sell_all", "label": "跌破站穩五日線前一天低點"},
        _ma5_trailing("sell_all", "日 K 收破五日線提前出場（五日線防守＋移動保利）"),
    ]
    for t in s.targets:
        if "頸線" in t["label"] or "季線" in t["label"]:
            s.exit_rules.append({"type": "target", "level": t["price"], "action": "sell_half",
                                 "label": f"抵達{t['label']}（頸線、季線壓力最大），積極出場"})
            break
    s.score = 55
    s.warnings.append("此策略吃慣性反彈，站上所有均線與頸線前都當短線操作")
    return s


def setup_climax_bottom(ctx: Context) -> Signal:
    df, r, p = ctx.df, ctx.row, ctx.prev
    s = Signal("climax_bottom", SETUP_NAMES["climax_bottom"], False)
    pp_vma = df["vma20"].iloc[-3]
    climax = p["volume"] >= 2.5 * pp_vma and p["chg"] <= -0.04
    conds = {
        "前一日回檔爆量（≥2.5 倍均量、跌幅 ≥4%）": bool(climax),
        "第二根延續：收紅且守住爆量 K 低點": r["close"] > p["close"] and r["low"] >= p["low"],
        "在季線之上或左側有支撐": r["close"] > r["ma60"] or (ctx.box is not None and p["low"] >= ctx.box.bottom * 0.98),
    }
    s.reasons = [k for k, v in conds.items() if v]
    s.missing = [k for k, v in conds.items() if not v]
    s.triggered = all(conds.values()) and ctx.after(12, 0)
    s.stop = round_tick(float(p["low"]) * 0.995)
    s.stop_reason = "爆量底不一定是底，停損守爆量 K 棒最低"
    s.entry_low = round_tick(float(p["close"]))
    s.entry_high = round_tick(float(r["close"]) * 1.01)
    s.targets = _targets_above(ctx, r["close"])
    s.exit_rules = [
        {"type": "price_below", "level": s.stop, "action": "sell_all", "label": "跌破爆量 K 低點"},
        {"type": "close_below_ma", "ma": 5, "action": "sell_all", "label": "未能站上／跌破五日線離場", "intraday_after": "13:00"},
    ]
    if s.targets:
        s.exit_rules.append({"type": "target", "level": s.targets[0]["price"], "action": "watch_pullback",
                             "label": "目標價滿足，盤中壓回就是弱勢可提前出場"})
    s.warnings.append("爆量反轉需要快速轉強，只要有敗象就出場；不要在第二次觸碰時左側買進")
    s.score = 50
    return s


SETUPS = [setup_breakout, setup_ma60_reclaim, setup_ma5_pullback, setup_ma20_ride, setup_rebound_ma5, setup_climax_bottom]


def evaluate(df: pd.DataFrame, now: datetime | None = None, intraday: bool = False) -> dict:
    """回傳 {context: {...}, signals: [Signal...]}"""
    ctx = build_context(df, now, intraday)
    if ctx is None:
        return {"context": None, "signals": []}
    warnings = _common_warnings(ctx)
    sigs = []
    for fn in SETUPS:
        try:
            sig = fn(ctx)
        except (KeyError, TypeError, ValueError):
            continue
        sig.warnings = warnings + sig.warnings
        # 垃圾時間、空方格局、量縮緩跌時一律綁手
        if ctx.garbage or ctx.struct == "downtrend" or (ctx.bleed and sig.setup != "climax_bottom"):
            sig.triggered = False
        if intraday and ctx.t and ctx.t < time(9, 5):
            sig.triggered = False
        sig.score += {"強位階": 10, "中立位階": 0, "弱位階": -15}.get(ctx.level, 0)
        close = float(ctx.row["close"])
        if sig.triggered and (sig.stop is None or sig.stop >= close):
            sig.triggered = False
            sig.warnings.append("停損價不在現價之下，進場要素不成立")
        if not sig.triggered and not sig.missing:
            # 條件都到位卻沒觸發：被時間規則或共通過濾擋下（垃圾時間、空方格局、量縮緩跌、操作空間不足…）
            sig.missing = [w for w in sig.warnings if "乖離" not in w][:2] or ["盤中尚未到判斷時間"]
        if not sig.triggered:
            # 型態沒成立就沒有「進場要素」，也就沒有停損／目標可言，不提供假設值以免誤用
            sig.stop = sig.entry_low = sig.entry_high = sig.healthy_pullback = None
            sig.stop_reason = ""
            sig.targets, sig.exit_rules = [], []
        sigs.append(sig)
    r = ctx.row
    context = {
        "close": float(r["close"]),
        "ma": {n: (None if pd.isna(r[f"ma{n}"]) else round(float(r[f"ma{n}"]), 2)) for n in ind.MA_WINDOWS},
        "angle": None if ctx.angle is None else round(ctx.angle, 1),
        "regime": ctx.regime,
        "level": ctx.level,
        "structure": ctx.struct,
        "bullish_alignment": ind.bullish_alignment(r),
        "volume_ratio": None if _vol_ratio(ctx) is None else round(_vol_ratio(ctx), 2),
        "box": None if ctx.box is None else {"top": round(ctx.box.top, 2), "bottom": round(ctx.box.bottom, 2), "days": ctx.box.days},
        "wave": None if ctx.wave is None else {k: round(v, 2) for k, v in ctx.wave.__dict__.items()},
        "warnings": warnings,
    }
    return {"context": context, "signals": sigs}


def synthetic_bar(df: pd.DataFrame, quote: dict, day: str, now: datetime | None = None) -> pd.DataFrame:
    """用即時報價組出今日 K 棒，接到歷史資料後面。

    now：盤中時間。給了就把「目前累計成交量」依開盤經過時間推估成全日量（例如 10:00 只過了約 1/4.5），
    均量、量比、爆量等條件才不會因為盤中量還沒出完而誤判。
    """
    price = quote.get("price")
    if price is None:
        return df
    vol = quote.get("volume") or 0
    if now is not None and vol:
        vol = ind.estimated_volume(vol, now)
    bar = pd.DataFrame(
        [{
            "open": quote.get("open") or price,
            "high": max(quote.get("high") or price, price),
            "low": min(quote.get("low") or price, price),
            "close": price,
            "volume": vol,
            "value": 0,
            "vol_est": 1.0 if now is not None else 0.0,
        }],
        index=pd.DatetimeIndex([pd.Timestamp(day)]),
    )
    base = df[df.index < pd.Timestamp(day)]
    return pd.concat([base, bar])


# ======================= 盤中預備訊號（訊號成立前的預告） =======================
# 講義的盤中判斷時間：9:30 前的壓回多是洗盤；季線、月線、反彈類要 12:00 後確認有撐
GATE = {"breakout": (9, 30), "ma5_pullback": (9, 30), "ma60_reclaim": (12, 0), "ma20_ride": (12, 0),
        "rebound_ma5": (12, 0), "climax_bottom": (12, 0)}
_STEPS = [s / 1000 for s in (5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30)]   # 現價 ±0.5% ～ ±3%


def pre_signals(hist: pd.DataFrame, quote: dict, day: str, now: datetime, max_missing: int = 1) -> list[dict]:
    """盤中訊號成立「之前」的預告：
      time：條件都到位，只差講義的判斷時間（9:30／12:00）→ 時間到若維持即成立
      price：只差 1 個條件，且價格到某個位置就會成立 → 回傳觸發價與距離
    垃圾時間、空方格局、量縮緩跌不預告（盤中訊號本來就會被擋下）。"""
    price = quote.get("price")
    if not price:
        return []
    base = synthetic_bar(hist, quote, day, now=now)
    ctx = build_context(base, now, True)
    if ctx is None or ctx.garbage or ctx.struct == "downtrend" or (ctx.t and ctx.t < time(9, 5)):
        return []
    out = []
    for fn in SETUPS:
        try:
            s = fn(ctx)
        except (KeyError, TypeError, ValueError):
            continue
        if s.triggered or (ctx.bleed and s.setup != "climax_bottom"):
            continue
        gate = GATE.get(s.setup, (12, 0))
        info = {"setup": s.setup, "name": s.name, "met": s.reasons, "missing": s.missing,
                "gate": f"{gate[0]}:{gate[1]:02d}", "price": price}
        at = now if ctx.after(*gate) else now.replace(hour=gate[0], minute=gate[1], second=0)
        if not s.missing:
            if not ctx.after(*gate):
                res = _verify(hist, quote, day, at, s.setup)
                if res is not None:
                    out.append({**info, "kind": "time", "_res": res})
            continue
        if len(s.missing) > max_missing or any("量" in m for m in s.missing):
            continue   # 量能條件無法靠價格達成
        for step in _STEPS:
            p = round_tick(price * (1 + step))
            q2 = {**quote, "price": p, "high": max(quote.get("high") or p, p), "low": min(quote.get("low") or p, p)}
            ctx2 = build_context(synthetic_bar(hist, q2, day, now=now), now, True)
            try:
                s2 = fn(ctx2) if ctx2 else None
            except (KeyError, TypeError, ValueError):
                s2 = None
            if s2 is not None and not s2.missing:
                res = _verify(hist, q2, day, at, s.setup)   # 用講義完整的盤中判斷（含停損、操作空間、時間）確認真的會成立
                if res is not None:
                    out.append({**info, "kind": "price", "trigger": p, "dist": round(step * 100, 1), "_res": res})
                break
    return out


def _verify(hist: pd.DataFrame, quote: dict, day: str, at: datetime, setup: str) -> dict | None:
    res = evaluate(synthetic_bar(hist, quote, day, now=at), now=at, intraday=True)
    return res if any(sg.setup == setup and sg.triggered for sg in res["signals"]) else None

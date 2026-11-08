"""步驟 1～2：還原權值價 ＋ 台股專屬特徵工程。

原則：第 t 天的特徵只用 t 日收盤（含）以前就能知道的資料，避免偷看未來。
"""
import numpy as np
import pandas as pd

from app import db
from app.rules import features as F

LIMIT = 0.10        # 台股漲跌幅 10%
LIMIT_TOL = 0.095   # 漲跌停價會向下取到升降單位，比例略小於 10%，用 9.5% 判斷較穩


# ======================= 1. 還原權值價 =======================
def adjust_prices(df: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """向後還原：除權息日之前的價格乘上（參考價 ÷ 除權息前收盤），消除除權息造成的假跳空。

    df：單一股票日 K（index=date，含 open/high/low/close）
    divs：該股票的除權息（date, prev_close, ref_price）
    回傳加上 adj_open/adj_high/adj_low/adj_close 與 ex_day（除權息日旗標）的新表。
    """
    out = df.copy()
    factor = pd.Series(1.0, index=df.index)
    for _, d in divs.sort_values("date", ascending=False).iterrows():
        ex = pd.Timestamp(d["date"])
        if not (d["prev_close"] and d["ref_price"]) or ex <= df.index[0] or ex > df.index[-1]:
            continue
        ratio = d["ref_price"] / d["prev_close"]
        if 0.3 < ratio < 1.0:  # 合理範圍外（例如減資、資料錯誤）不還原
            factor[factor.index < ex] *= ratio
    for k in ("open", "high", "low", "close"):
        out[f"adj_{k}"] = df[k] * factor
    out["ex_day"] = df.index.isin(pd.to_datetime(divs["date"])) if len(divs) else False
    return out


# ======================= 2. 價格與波動度 =======================
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """真實波幅 ATR（Wilder 平滑，等同 TA-Lib ATR）。使用還原價，除權息缺口不會被當成波動。"""
    h, l, c = df["adj_high"], df["adj_low"], df["adj_close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["adj_close"], df["volume"]
    logc = np.log(c)
    f = pd.DataFrame(index=df.index)
    # 對數報酬率（還原價）
    for n in (1, 5, 10, 20, 60):
        f[f"logret_{n}"] = logc.diff(n)
    f["atr14"] = atr(df, 14)
    f["atr_pct"] = f["atr14"] / c                      # 波動度（佔股價比例），跨股票可比較
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = 4 * sd / mid                       # 布林通道寬度 (上軌−下軌)/中軌
    f["bb_pos"] = (c - (mid - 2 * sd)) / (4 * sd)      # 在通道中的位置 0~1
    f["bb_width_chg"] = f["bb_width"] / f["bb_width"].shift(5) - 1   # 通道開口（壓縮後擴張）
    for n in (5, 20, 60):
        ma = c.rolling(n).mean()
        f[f"gap_ma{n}"] = c / ma - 1                   # 乖離率（講義：五日線乖離 > 8% 易回檔）
    f["ma20_slope"] = c.rolling(20).mean().pct_change(5)
    f["vol_ratio"] = v / v.rolling(20).mean()
    f["vol_trend"] = v.rolling(5).mean() / v.rolling(20).mean()
    f["high_dist_60"] = c / df["adj_high"].rolling(60).max() - 1  # 距 60 日高點
    f["chg"] = c.pct_change()
    prev = df["close"].shift(1)                        # 漲跌停用「未還原」價判斷（除權息日以參考價為基準，已由下方排除）
    f["limit_up_close"] = (df["close"] / prev - 1 >= LIMIT_TOL) & (df["close"] >= df["high"])
    f["limit_down_close"] = (df["close"] / prev - 1 <= -LIMIT_TOL) & (df["close"] <= df["low"])
    f.loc[df["ex_day"].astype(bool), ["limit_up_close", "limit_down_close"]] = False
    return f


# ======================= 3. 籌碼面 =======================
def _streak(s: pd.Series) -> pd.Series:
    """連續買超天數（正）／連續賣超天數（負）。"""
    sign = np.sign(s.fillna(0))
    grp = (sign != sign.shift()).cumsum()
    return sign * sign.groupby(grp).cumcount().add(1)


def chip_features(df: pd.DataFrame, chips: pd.DataFrame, margin: pd.DataFrame, broker: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    v = df["volume"].replace(0, np.nan)  # 張
    ch = chips.reindex(df.index)
    for k, name in (("foreign_net", "foreign"), ("trust_net", "trust"), ("dealer_net", "dealer")):
        s = ch[k] if k in ch else pd.Series(np.nan, index=df.index)
        f[f"{name}_streak"] = _streak(s).where(s.notna())          # 連續買超（賣超）天數
        f[f"{name}_part"] = s / v                                   # 當日法人參與度：買超張數 ÷ 成交量
        f[f"{name}_part5"] = s.rolling(5).sum() / v.rolling(5).sum()
    if "total_net" in ch:
        f["insti_part"] = ch["total_net"] / v
        f["insti_part20"] = ch["total_net"].rolling(20).sum() / v.rolling(20).sum()
    # 散戶指標：融資、融券
    m = margin.reindex(df.index)
    if len(margin):
        f["margin_chg"] = m["margin_bal"] / m["margin_prev"].replace(0, np.nan) - 1
        f["short_chg"] = m["short_bal"] / m["short_prev"].replace(0, np.nan) - 1
        f["margin_chg5"] = m["margin_bal"] / m["margin_bal"].shift(5) - 1
        f["short_margin_ratio"] = m["short_bal"] / m["margin_bal"].replace(0, np.nan)   # 券資比
        f["margin_use"] = m["margin_bal"] / v.rolling(20).mean()                        # 融資餘額 ÷ 20 日均量
        f["both_up"] = ((f["margin_chg"] > 0) & (f["short_chg"] > 0)).astype(float)      # 資券同增
        # 融資銳減（斷頭訊號）：單日融資減少 > 3% 且股價下跌
        f["margin_crash"] = ((f["margin_chg"] < -0.03) & (df["close"].pct_change() < 0)).astype(float)
    else:
        for k in ("margin_chg", "short_chg", "margin_chg5", "short_margin_ratio", "margin_use", "both_up", "margin_crash"):
            f[k] = np.nan
    # 主力集中度：(買超前 10 大分點 − 賣超前 10 大分點) ÷ 總成交量；沒有分點資料時為 NaN（LightGBM 會自動處理缺值）
    b = broker.reindex(df.index)
    net = (b["top_buy"] - b["top_sell"]) if len(broker) else pd.Series(np.nan, index=df.index)
    for n in (1, 3, 5):
        f[f"conc_{n}"] = net.rolling(n).sum() / v.rolling(n).sum()
    return f


# ======================= 4. 大盤體制 =======================
def market_regime(index_df: pd.DataFrame) -> pd.DataFrame:
    """加權指數 20MA／60MA：指數在任一條均線之下 → market_trend = 0（空頭體制，過濾多頭訊號）。"""
    c = index_df["close"]
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    out = pd.DataFrame(index=index_df.index)
    out["market_trend"] = ((c > ma20) & (c > ma60)).astype(float).where(ma60.notna())
    out["taiex_gap20"] = c / ma20 - 1
    out["taiex_gap60"] = c / ma60 - 1
    out["taiex_ret5"] = np.log(c).diff(5)
    out["taiex_ret20"] = np.log(c).diff(20)
    return out


# ======================= 5. 講義特徵 =======================
def handout_features(df: pd.DataFrame) -> pd.DataFrame:
    """把講義的條件（28 個是／否特徵）與六種進場型態的「條件符合比例」一起丟給模型。"""
    raw = df[["open", "high", "low", "close", "volume"]].copy()
    raw[["open", "high", "low", "close"]] = df[["adj_open", "adj_high", "adj_low", "adj_close"]].values
    f = F.stock_features(raw).drop(columns="_valid").astype(float).add_prefix("h_")
    tables, weak = F.setup_conditions(raw)
    for s, t in tables.items():
        f[f"setup_{s}"] = t.mean(axis=1)   # 例如 5/6 = 0.83
    f["weak_level"] = weak.astype(float)
    return f


# ======================= 組合 =======================
def load_inputs(since: str) -> dict:
    """一次讀出全市場資料，回傳各表（依股票分組）。"""
    def grouped(sql, params=()):
        rows = db.query(sql, params)
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        df["date"] = pd.to_datetime(df["date"])
        return {sid: g.drop(columns="stock_id").set_index("date") for sid, g in df.groupby("stock_id")}

    idx = pd.DataFrame(db.query("SELECT date, close FROM index_daily WHERE code='TAIEX' ORDER BY date"))
    if not idx.empty:
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.set_index("date")
    divs = pd.DataFrame(db.query("SELECT stock_id, date, prev_close, ref_price FROM dividends"))
    return {
        "chips": grouped("SELECT * FROM chips_daily WHERE date>=?", (since,)),
        "margin": grouped("SELECT * FROM margin_daily WHERE date>=?", (since,)),
        "broker": grouped("SELECT * FROM broker_daily WHERE date>=?", (since,)),
        "divs": {sid: g for sid, g in divs.groupby("stock_id")} if not divs.empty else {},
        "regime": market_regime(idx) if not idx.empty else pd.DataFrame(),
    }


def build_features(sid: str, g: pd.DataFrame, inputs: dict, industry_feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """單一股票的完整特徵表（含還原價，供標記使用）。"""
    empty = pd.DataFrame(columns=["date", "prev_close", "ref_price"])
    adj = adjust_prices(g, inputs["divs"].get(sid, empty))
    f = pd.concat([
        price_features(adj),
        chip_features(adj, inputs["chips"].get(sid, pd.DataFrame()), inputs["margin"].get(sid, pd.DataFrame()),
                      inputs["broker"].get(sid, pd.DataFrame())),
        handout_features(adj),
    ], axis=1)
    if len(inputs["regime"]):
        f = f.join(inputs["regime"].reindex(f.index).ffill(limit=3))
    if industry_feats is not None:
        f = f.join(industry_feats.reindex(f.index))
    return f.join(adj[["adj_open", "adj_high", "adj_low", "adj_close", "close", "high", "low", "open", "ex_day"]])


# 給模型的欄位：排除價格本身與標記用欄位（模型只看相對、可跨股票比較的數值）
EXCLUDE = {"adj_open", "adj_high", "adj_low", "adj_close", "close", "high", "low", "open", "ex_day", "atr14", "chg",
           # 大盤欄位同一天所有股票都一樣：只有一年資料時模型會變成「猜大盤」而過度擬合，
           # 所以大盤只當濾網（market_trend = 0 不做多），不給模型學
           "market_trend", "taiex_gap20", "taiex_gap60", "taiex_ret5", "taiex_ret20", "h_market_bull"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and not c.startswith("_")]

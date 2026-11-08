"""出場策略模擬：初始停損 → 保本 → 移動停利（不設固定停利），向量化計算。

目標（使用者要求）：勝率越高越好、小賺沒關係，但「絕對不能大賠」。

每一筆交易：
  進場：訊號日 t 的隔日開盤（訊號日收盤鎖漲停、進場日一字漲停 → 買不到，不計）
  初始停損：max(進場價 − sl_atr × ATR, 進場價 × (1 − sl_cap))   ← sl_cap 是硬上限，保證停損距離不超過例如 5%
  保本：最高價曾達 進場價 + be_atr × ATR → 停損上移到 進場價 × (1 + be_buffer)（扣完成本仍小賺）
  移動停利：停損 = 進場後最高價 − trail_atr × ATR（只上不下）
  五日線：ma5_mode = "after_be" 時，保本後收盤跌破五日線出場（講義：五日線移動停利）
  時間：持有 horizon 天仍未出場 → 收盤出場

停損價每天以「前一天收盤後」已知的最高價計算（盤中不會偷看當天高點）。
跳空開在停損之下 → 以開盤價成交（這就是可能的大賠來源，所以統計會列出最差 1%／5% 的虧損）。
出場當天一字跌停 → 賣不掉，順延到下一個沒鎖跌停的交易日開盤。
"""
from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

COST = 0.001425 * 0.3 * 2 + 0.003
LIMIT_TOL = 0.095


@dataclass(frozen=True)
class Policy:
    sl_atr: float = 1.5          # 初始停損：幾倍 ATR
    sl_cap: float = 0.05         # 初始停損最多幾 %（硬上限，防大賠）
    be_atr: float | None = 1.0   # 漲到幾倍 ATR 後保本；None = 不保本
    be_buffer: float = 0.005     # 保本價 = 成本 × (1 + 0.5%)，扣手續費與稅後仍小賺
    trail_atr: float | None = 2.0  # 移動停利：最高價 − 幾倍 ATR；None = 不移動
    ma5_mode: str = "none"       # "none" / "after_be"
    horizon: int = 20            # 最長持有天數

    def to_dict(self):
        return asdict(self)

    def describe(self) -> str:
        parts = [f"初始停損 {self.sl_atr} ATR（最多 −{self.sl_cap:.0%}）"]
        if self.be_atr is not None:
            parts.append(f"漲 {self.be_atr} ATR 後停損移到成本 +{self.be_buffer:.1%}（保本）")
        if self.trail_atr is not None:
            parts.append(f"移動停利：最高價 − {self.trail_atr} ATR")
        if self.ma5_mode == "after_be":
            parts.append("保本後收盤跌破五日線出場")
        parts.append(f"最長 {self.horizon} 個交易日")
        return "；".join(parts)


def simulate(o, h, l, c, ma5, atr, raw_c, raw_h, raw_l, ex_day, events: np.ndarray, p: Policy) -> dict:
    """單一股票：對 events（訊號日索引）模擬交易。陣列皆為還原價（raw_* 為未還原價，判斷漲跌停用）。

    回傳 {idx, ret, win, days, reason, mae}：只含成交且有完整未來資料的事件。
    """
    n, H = len(c), p.horizon
    pad = H + 4                                  # 多留幾天處理跌停順延
    if n < pad + 2 or not len(events):
        return {k: np.array([]) for k in ("idx", "ret", "win", "days", "reason", "mae")}
    raw_prev = np.r_[np.nan, raw_c[:-1]]
    locked_up = (raw_l / raw_prev - 1 >= LIMIT_TOL) & ~ex_day
    locked_dn = (raw_h / raw_prev - 1 <= -LIMIT_TOL) & ~ex_day
    lu_close = (raw_c / raw_prev - 1 >= LIMIT_TOL) & (raw_c >= raw_h) & ~ex_day

    ev = events[(events + 1 + pad <= n - 1)]
    ev = ev[~np.isnan(atr[ev]) & (atr[ev] > 0) & ~lu_close[ev] & ~locked_up[ev + 1]]
    if not len(ev):
        return {k: np.array([]) for k in ("idx", "ret", "win", "days", "reason", "mae")}
    e = ev + 1
    W = lambda a: sliding_window_view(a, pad)[e]            # (事件數, pad)
    O, Hh, L, C, M5, LD = W(o), W(h), W(l), W(c), W(ma5), W(locked_dn)
    entry = O[:, 0]
    A = atr[ev]
    init = np.maximum(entry - p.sl_atr * A, entry * (1 - p.sl_cap))
    hh = np.maximum.accumulate(Hh, axis=1)
    hh_prev = np.concatenate([entry[:, None], hh[:, :-1]], axis=1)
    hh_prev = np.maximum(hh_prev, entry[:, None])
    stop = np.repeat(init[:, None], pad, axis=1)
    be_on = np.zeros_like(stop, dtype=bool)
    if p.be_atr is not None:
        be_on = hh_prev >= (entry + p.be_atr * A)[:, None]
        stop = np.where(be_on, np.maximum(stop, (entry * (1 + p.be_buffer))[:, None]), stop)
    if p.trail_atr is not None:
        stop = np.maximum(stop, hh_prev - (p.trail_atr * A)[:, None])
    days = np.arange(pad)
    in_h = days < H
    hit_stop = (L <= stop) & in_h
    hit_ma5 = np.zeros_like(hit_stop)
    if p.ma5_mode == "after_be":
        hit_ma5 = (C < M5) & be_on & in_h & ~hit_stop
    first = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), pad)
    k_stop, k_ma5 = first(hit_stop), first(hit_ma5)
    k = np.minimum(np.minimum(k_stop, k_ma5), H - 1)
    reason = np.where(k == k_stop, "stop", np.where(k == k_ma5, "ma5", "time"))
    rows = np.arange(len(ev))
    s_k = stop[rows, k]
    px = np.where(reason == "stop", np.where(O[rows, k] <= s_k, O[rows, k], s_k), C[rows, k])
    px = np.where((reason == "stop") & (k == 0), s_k, px)    # 進場當天開盤即進場價，不會跳空低於停損
    # 出場日一字跌停：賣不掉，順延到下一個未鎖跌停日的開盤
    locked = LD[rows, k]
    if locked.any():
        for i in np.where(locked)[0]:
            j = k[i] + 1
            while j < pad - 1 and LD[i, j]:
                j += 1
            px[i], k[i] = O[i, j], j
    ret = px / entry - 1 - COST
    mae = (np.minimum.accumulate(L, axis=1)[rows, k] / entry - 1)   # 持有期間最大不利幅度
    return {"idx": ev, "ret": ret, "win": ret > 0, "days": k + 1, "reason": reason, "mae": mae}


def stats(ret: np.ndarray, big_loss: float = 0.05) -> dict:
    """勝率、期望報酬與「大賠」指標。big_loss：虧損超過多少算大賠（預設 5%）。"""
    if not len(ret):
        return {"n": 0}
    r = np.asarray(ret, float)
    loss = r[r <= 0]
    win = r[r > 0]
    q = np.sort(r)
    worst1 = q[: max(1, len(q) // 100)].mean()
    return {
        "n": int(len(r)), "rate": round(float((r > 0).mean()), 4), "exp": round(float(r.mean()) * 100, 3),
        "avg_win": round(float(win.mean()) * 100, 2) if len(win) else None,
        "avg_loss": round(float(loss.mean()) * 100, 2) if len(loss) else None,
        "p5": round(float(np.percentile(r, 5)) * 100, 2),            # 最差 5% 的門檻
        "worst1": round(float(worst1) * 100, 2),                       # 最差 1% 的平均虧損
        "big_loss_rate": round(float((r < -big_loss).mean()), 4),      # 虧損超過 big_loss 的比例
        "max_loss": round(float(r.min()) * 100, 2),
    }

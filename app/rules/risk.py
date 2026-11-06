"""資金控管與停損觀念（技術分析講義〈停損〉）。

- 停損始終貫徹一個觀念：進場要素消失就出場。
- 同時設定「關鍵停損價」與「可承受虧損」：如果進場價會讓虧損超過可承受範圍就不進場。
  例：資金 100 萬、單筆最大虧損 1 萬、每筆 10 萬 → 進場點與停損點距離不能超過 10%。
"""
import math
from dataclasses import asdict, dataclass

BUY_FEE = 0.001425
SELL_TAX = 0.003


@dataclass
class Sizing:
    ok: bool
    max_entry: float  # 在可承受虧損內最高可進場價
    shares: int  # 建議股數（以「股」計，1 張 = 1000 股，可零股）
    lots: float
    cost: float
    loss_at_stop: float
    stop_pct: float
    reason: str

    def to_dict(self):
        return asdict(self)


def max_stop_pct(per_trade_capital: float, max_loss: float) -> float:
    return max_loss / per_trade_capital


def size_position(entry: float, stop: float, per_trade_capital: float, max_loss: float,
                  fee_discount: float = 0.3) -> Sizing:
    pct = max_stop_pct(per_trade_capital, max_loss)
    max_entry = stop / (1 - pct)
    if entry <= stop:
        return Sizing(False, round(max_entry, 2), 0, 0, 0, 0, 0, "進場價低於停損價")
    risk_per_share = entry - stop + entry * BUY_FEE * fee_discount + stop * (BUY_FEE * fee_discount + SELL_TAX)
    by_capital = per_trade_capital / entry
    by_risk = max_loss / risk_per_share
    shares = int(min(by_capital, by_risk))
    shares = shares // 1000 * 1000 if shares >= 1000 else shares  # 足夠時湊整張，不足用零股
    loss = shares * risk_per_share
    stop_pct = (entry - stop) / entry
    ok = entry <= max_entry * 1.0001 and shares > 0
    reason = "符合風控" if ok else f"進場價 {entry:.2f} 高於可承受虧損的最高進場價 {max_entry:.2f}，不進場"
    return Sizing(ok, round(max_entry, 2), shares, round(shares / 1000, 3), round(shares * entry),
                  round(loss), round(stop_pct * 100, 2), reason)


def round_tick(price: float) -> float:
    """依台股升降單位四捨五入。"""
    if price < 10:
        tick = 0.01
    elif price < 50:
        tick = 0.05
    elif price < 100:
        tick = 0.1
    elif price < 500:
        tick = 0.5
    elif price < 1000:
        tick = 1
    else:
        tick = 5
    return round(math.floor(price / tick + 0.5) * tick, 2)

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
    """進場點到停損點的最大距離。現股最多虧掉整個部位，因此最大虧損大於部位資金時以部位資金為上限。"""
    if per_trade_capital <= 0:
        return 0.0
    return min(max_loss, per_trade_capital) / per_trade_capital * 0.95 if max_loss >= per_trade_capital else max_loss / per_trade_capital


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


# ======================= 融資（槓桿） =======================
@dataclass
class MarginSizing:
    ok: bool
    shares: int
    lots: float
    ratio: float  # 融資成數
    self_funded: float  # 自備款
    loan: float  # 融資金額
    leverage: float  # 部位市值 / 自備款
    call_price: float  # 維持率跌到追繳線的股價
    interest_est: float  # 預估持有期間利息
    loss_at_stop: float  # 含利息
    reason: str

    def to_dict(self):
        return asdict(self)


def margin_call_price(buy_price: float, ratio: float, maintenance: float = 1.30) -> float:
    """融資維持率 = 股票市值 / 融資金額；跌到 maintenance 以下會被追繳。
    追繳價 = maintenance × 融資成數 × 買進價（上市 6 成 → 約買進價的 78%）。"""
    return buy_price * ratio * maintenance


def maintenance_ratio(price: float, buy_price: float, ratio: float) -> float:
    return price / (buy_price * ratio)


def margin_interest(loan: float, annual_rate: float, days: int) -> float:
    return loan * annual_rate * days / 365


def size_margin(entry: float, stop: float, per_trade_capital: float, max_loss: float, ratio: float,
                annual_rate: float, maintenance: float = 1.30, hold_days: int = 20, fee_discount: float = 0.3) -> MarginSizing:
    """融資部位：每筆部位資金視為「自備款」，可買到 1/(1−成數) 倍市值；
    但單筆最大虧損不變——虧損按整個部位市值計算，所以槓桿只在停損距離夠近時才放大部位。"""
    call = margin_call_price(entry, ratio, maintenance)
    if entry <= stop:
        return MarginSizing(False, 0, 0, ratio, 0, 0, 0, round(call, 2), 0, 0, "進場價低於停損價")
    interest_ps = entry * ratio * annual_rate * hold_days / 365
    risk_ps = entry - stop + entry * BUY_FEE * fee_discount + stop * (BUY_FEE * fee_discount + SELL_TAX) + interest_ps
    by_capital = per_trade_capital / (entry * (1 - ratio))
    by_risk = max_loss / risk_ps
    shares = int(min(by_capital, by_risk))
    shares = shares // 1000 * 1000 if shares >= 1000 else 0  # 融資不能買零股
    loan = shares * entry * ratio
    self_funded = shares * entry - loan
    reason, ok = "符合風控", shares > 0
    if call >= stop:
        ok, reason = False, f"追繳價 {call:.2f} 高於停損價 {stop:.2f}：還沒停損就會先被追繳，不適合融資"
    elif shares == 0:
        ok, reason = False, "融資須整張買進，可承受虧損內買不到 1 張"
    return MarginSizing(ok, shares, shares / 1000, ratio, round(self_funded), round(loan),
                        round(shares * entry / self_funded, 2) if self_funded else 0, round(call, 2),
                        round(shares * interest_ps), round(shares * risk_ps), reason)


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

"""交易成本、損益與損益平衡點（對齊券商帳上算法，例如元大證券）。

- 手續費 = 成交金額 × 0.1425% × 折數，無條件捨去；整股最低 20 元、零股最低 1 元
- 證交稅：股票 0.3%、ETF 0.1%（賣出時收）
- 成本 = 成交金額 + 買進手續費
- 未實現損益 = 市值 − 預估賣出手續費 − 證交稅 − 成本；報酬率 = 未實現損益 ÷ 成本
- 券商多採「月退」：帳上先收原價手續費、折讓月底退回，所以帳上的未實現損益以原價手續費估算
"""
import math

FEE_RATE = 0.001425
STOCK_TAX = 0.003
ETF_TAX = 0.001


def is_etf(stock_id: str) -> bool:
    return stock_id.startswith("00")


def tax_rate(stock_id: str) -> float:
    return ETF_TAX if is_etf(stock_id) else STOCK_TAX


def fee(amount: float, discount: float = 1.0, shares: int = 1000) -> int:
    if amount <= 0:
        return 0
    minimum = 20 if shares >= 1000 and shares % 1000 == 0 else 1
    return max(minimum, math.floor(amount * FEE_RATE * discount))


def buy_cost(price: float, shares: int, discount: float = 1.0) -> int:
    amount = price * shares
    return math.floor(amount) + fee(amount, discount, shares)


def sell_proceeds(stock_id: str, price: float, shares: int, discount: float = 1.0) -> dict:
    amount = price * shares
    f = fee(amount, discount, shares)
    t = math.floor(amount * tax_rate(stock_id))
    return {"amount": amount, "fee": f, "tax": t, "net": amount - f - t}


def unrealized(stock_id: str, cost: float, price: float, shares: int, discount: float = 1.0) -> dict:
    s = sell_proceeds(stock_id, price, shares, discount)
    pnl = s["net"] - cost
    return {"market_value": round(s["amount"]), "sell_fee": s["fee"], "sell_tax": s["tax"],
            "pnl": round(pnl), "pnl_pct": round(pnl / cost * 100, 2) if cost else None}


def tick(price: float, etf: bool = False) -> float:
    if etf:
        return 0.01 if price < 50 else 0.05
    for lim, t in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1)):
        if price < lim:
            return t
    return 5


def break_even(stock_id: str, cost: float, shares: int, discount: float = 1.0) -> dict:
    """賣出淨額剛好等於成本的價格；實際掛單要用往上取到升降單位的價格。"""
    if shares <= 0:
        return {"exact": None, "tick_price": None}
    exact = cost / (shares * (1 - FEE_RATE * discount - tax_rate(stock_id)))
    t = tick(exact, is_etf(stock_id))
    p = math.ceil(round(exact / t, 6)) * t
    while sell_proceeds(stock_id, p, shares, discount)["net"] < cost:  # 處理捨去與最低手續費
        p += t
    return {"exact": round(exact, 4), "tick_price": round(p, 2)}

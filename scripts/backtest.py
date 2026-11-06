"""純技術面歷史回測：驗證進出場規則（未套用基本面，因歷史財報需時間點對齊）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 讓 scripts/ 可直接執行
from collections import defaultdict
from datetime import timedelta

import pandas as pd

from app import db
from app.rules import technical
from app.services import portfolio

days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 80
dates = [r["date"] for r in db.query("SELECT DISTINCT date FROM prices ORDER BY date")]
allp = db.load_all_prices(dates[0])
allp["date"] = pd.to_datetime(allp["date"])
groups = {s: g.set_index("date").drop(columns="stock_id") for s, g in allp.groupby("stock_id")}
liquid = {s for s, g in groups.items() if len(g) >= 200 and g["volume"].iloc[-60:].mean() >= 1000 and g["close"].iloc[-1] >= 10}

trades = []
scan_dates = dates[-days_back:-10]
for d in scan_dates[::2]:  # 每兩天掃一次加速
    ts = pd.Timestamp(d)
    for sid in liquid:
        g = groups[sid]
        hist = g[g.index <= ts]
        if len(hist) < 200 or hist.index[-1] != ts:
            continue
        res = technical.evaluate(hist.tail(300))
        for sig in res["signals"]:
            if not sig.triggered or not sig.stop:
                continue
            price = float(hist["close"].iloc[-1])
            rr = [(t["price"] - price) / (price - sig.stop) for t in sig.targets] if price > sig.stop else []
            if not rr or max(rr) < 1.5:
                continue
            fut = g[g.index > ts]
            if len(fut) < 2:
                continue
            entry = float(fut["open"].iloc[0])
            if entry <= sig.stop:
                continue
            plan = sig.to_dict()
            pos = {"id": 0, "buy_price": entry, "buy_date": fut.index[0].date().isoformat(), "remaining": 1000, "plan": plan}
            remaining, pnl, exit_reason, held = 1.0, 0.0, "持有到期(30日)", 0
            for i in range(len(fut[:30])):
                day = fut.index[i]
                window = g[g.index <= day].tail(300)
                hits = portfolio.evaluate_exits(pos, window, None, intraday=False)
                held = i + 1
                close = float(window["close"].iloc[-1])
                low = float(window["low"].iloc[-1])
                for h in hits:
                    px = min(h["level"], float(window["open"].iloc[-1])) if h["type"] == "price_below" and low <= h["level"] else close
                    if h["action"] == "sell_all":
                        pnl += remaining * (px / entry - 1)
                        remaining = 0
                        exit_reason = h["label"][:12]
                        break
                    if h["action"] == "sell_half" and remaining == 1.0:
                        pnl += 0.5 * (px / entry - 1)
                        remaining = 0.5
                if remaining == 0:
                    break
            if remaining:
                pnl += remaining * (float(fut["close"].iloc[min(held, len(fut)) - 1]) / entry - 1)
            r = pnl / ((entry - sig.stop) / entry)
            trades.append({"date": d, "sid": sid, "setup": sig.setup, "ret": pnl * 100, "R": r, "held": held, "exit": exit_reason})

df = pd.DataFrame(trades)
if df.empty:
    print("no trades"); sys.exit()
df["ret_net"] = df["ret"] - 0.585  # 手續費 3 折＋證交稅
print(f"期間 {scan_dates[0]} ~ {scan_dates[-1]}，共 {len(df)} 筆")
summary = df.groupby("setup").agg(筆數=("ret", "size"), 勝率=("ret_net", lambda x: f"{(x > 0).mean():.0%}"),
                                  平均報酬=("ret_net", lambda x: round(x.mean(), 2)), 平均R=("R", lambda x: round(x.mean(), 2)),
                                  平均持有=("held", "mean"))
print(summary.to_string())
print("\n整體：勝率 %.0f%%，平均淨報酬 %.2f%%，平均 R %.2f" % ((df.ret_net > 0).mean() * 100, df.ret_net.mean(), df.R.mean()))
print("\n出場原因：\n", df["exit"].value_counts().head(8).to_string())

"""步驟 3：三欄式標記法（Triple-Barrier Method）＋ 台股交易規則防禦。

對每個訊號日 t：
  進場：t+1 日開盤（t 日收盤後才知道訊號，不能用 t 日收盤價成交，否則就是偷看）
  上邊界（停利）：進場價 + up × ATR(t)
  下邊界（停損）：進場價 − down × ATR(t)
  時間邊界：持有 horizon 個交易日，到期以收盤出場

先碰上邊界 → 1；先碰下邊界 → 0；先到時間邊界 → 依扣成本後報酬正負決定。
同一根 K 棒同時碰到上下邊界（看不到盤中先後）→ 保守算碰到停損。

交易規則防禦（防止回測作弊）：
  - 訊號日收盤鎖漲停：當天排隊買不到，該筆不計入（no_fill）
  - 進場日一字漲停（最低價 = 漲停價）：開盤買不到，不計入
  - 跳空：開盤直接越過停損／停利，以開盤價成交，而不是理想的邊界價
  - 跌停鎖死（最高價 = 跌停價）：當天賣不掉，順延到下一個沒有鎖跌停的交易日開盤賣出
  - 價格一律使用還原權值價，除權息缺口不會被誤判成停損
"""
import numpy as np
import pandas as pd

from app.ml.features import LIMIT_TOL

COST = 0.001425 * 0.3 * 2 + 0.003   # 手續費 3 折 × 買賣兩次 ＋ 證交稅（與勝率模型相同）


def triple_barrier(f: pd.DataFrame, up: float = 2.0, down: float = 1.5, horizon: int = 10,
                   skip_limit_up_signal: bool = True) -> pd.DataFrame:
    """f：build_features 的輸出（需含 adj_*、close/high/low、atr14、ex_day、limit_up_close）。

    回傳同 index 的 DataFrame：
      y        1 獲利 / 0 虧損 / NaN（未來資料不足或買不到）
      ret      扣成本後的實際報酬（含跳空、跌停賣不掉的真實結果）
      barrier  'up' / 'down' / 'time'
      days     持有天數
      no_fill  因漲停買不到而不計的樣本
    """
    o, h, l, c = (f[f"adj_{k}"].to_numpy(float) for k in ("open", "high", "low", "close"))
    raw_c, raw_h, raw_l = f["close"].to_numpy(float), f["high"].to_numpy(float), f["low"].to_numpy(float)
    raw_prev = np.r_[np.nan, raw_c[:-1]]
    ex = f["ex_day"].to_numpy(bool)
    # 鎖漲停：整天最低價就是漲停價；鎖跌停：整天最高價就是跌停價（除權息日以參考價為基準，略過判斷）
    locked_up = (raw_l / raw_prev - 1 >= LIMIT_TOL) & ~ex
    locked_down = (raw_h / raw_prev - 1 <= -LIMIT_TOL) & ~ex
    atr = f["atr14"].to_numpy(float)
    lu_close = f["limit_up_close"].to_numpy(bool)
    n = len(f)
    y = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    days = np.full(n, np.nan)
    barrier = np.full(n, None, dtype=object)
    no_fill = np.zeros(n, dtype=bool)

    for t in range(n - 1):
        if np.isnan(atr[t]) or atr[t] <= 0:
            continue
        if skip_limit_up_signal and lu_close[t]:
            no_fill[t] = True                      # 訊號日收盤鎖漲停：排不到
            continue
        e = t + 1
        if locked_up[e]:
            no_fill[t] = True                      # 進場日一字漲停：買不到
            continue
        entry = o[e]
        if not entry or np.isnan(entry):
            continue
        tp, sl = entry + up * atr[t], entry - down * atr[t]
        last = e + horizon - 1
        if last >= n:
            break                                  # 未來資料不足：不標記（不能用「目前為止」的結果）
        exit_px, kind, k_exit = None, None, None
        pending = False                            # 已觸發出場但跌停鎖死，等下一天
        for k in range(e, last + 1):
            if pending:
                if not locked_down[k]:
                    exit_px, k_exit = o[k], k
                    break
                continue
            if l[k] <= sl:                          # 停損優先（保守）
                kind = "down"
                if locked_down[k]:
                    pending = True                 # 跌停鎖死賣不掉，順延
                    continue
                exit_px, k_exit = min(o[k], sl) if k > e else sl, k   # 跳空跌破 → 開盤價成交
                break
            if h[k] >= tp:
                kind = "up"
                exit_px, k_exit = (max(o[k], tp) if k > e else tp), k  # 跳空越過停利 → 開盤價成交
                break
        if exit_px is None and not pending:
            kind = "time"
            if locked_down[last]:
                pending = True
            else:
                exit_px, k_exit = c[last], last
        if pending and exit_px is None:            # 一路鎖跌停到時間邊界之後：繼續往後找能賣的日子
            for k in range(last + 1, n):
                if not locked_down[k]:
                    exit_px, k_exit = o[k], k
                    break
            if exit_px is None:
                continue
        r = exit_px / entry - 1 - COST
        ret[t], days[t], barrier[t] = r, k_exit - e + 1, kind
        y[t] = 1.0 if kind == "up" else 0.0 if kind == "down" else float(r > 0)
    return pd.DataFrame({"y": y, "ret": ret, "barrier": barrier, "days": days, "no_fill": no_fill}, index=f.index)

"""FinMind API v4 客戶端（財報、月營收、法人買賣超、融資券、期權籌碼、匯率）。

免費會員每小時 600 次，這裡用令牌桶節流並把結果快取在 SQLite。
token 從環境變數 FINMIND_TOKEN 讀取。
"""
import threading
import time
from collections import deque

import httpx

from app import db
from app.config import get_settings

API = "https://api.finmindtrade.com/api/v4/data"


class _Bucket:
    """滑動視窗：一小時內最多 per_hour 次，未滿時不等待（允許突發查詢）。"""

    def __init__(self, per_hour: int):
        self.limit = per_hour
        self.lock = threading.Lock()
        self.calls: deque[float] = deque()

    def take(self):
        """背景工作只用到 85% 額度，保留 15% 給網頁上即時查詢（個股分析），避免查詢卡住。"""
        cap = int(self.limit * 0.85) if getattr(_ctx, "background", False) else self.limit
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > 3600:
                    self.calls.popleft()
                if len(self.calls) < cap:
                    self.calls.append(now)
                    return
                wait = 3600 - (now - self.calls[0]) + 0.1
            time.sleep(min(wait, 30))

    def used(self) -> int:
        with self.lock:
            now = time.time()
            return sum(1 for t in self.calls if now - t <= 3600)


_bucket = _Bucket(550 if get_settings().finmind_token else 280)  # 有 token 600/時、無 token 300/時，預留餘裕


_ctx = threading.local()


class background:
    """with finmind.background(): ... 內的查詢屬於背景工作（較低優先）。"""

    def __enter__(self):
        _ctx.background = True

    def __exit__(self, *exc):
        _ctx.background = False


class QuotaExceeded(RuntimeError):
    pass


def fetch(dataset: str, data_id: str | None = None, start: str = "2024-01-01", end: str | None = None,
          cache_hours: float | None = None) -> list[dict]:
    key = f"fm:{dataset}:{data_id}:{start}:{end}"
    if cache_hours:
        hit = db.cache_get(key, cache_hours)
        if hit is not None:
            return hit
    params = {"dataset": dataset, "start_date": start}
    if data_id:
        params["data_id"] = data_id
    if end:
        params["end_date"] = end
    headers = {}
    token = get_settings().finmind_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    _bucket.take()
    r = httpx.get(API, params=params, headers=headers, timeout=40)
    if r.status_code == 402:
        raise QuotaExceeded("FinMind 額度用完，請稍後再試")
    r.raise_for_status()
    data = r.json().get("data", [])
    if cache_hours:
        db.cache_set(key, data)
    return data

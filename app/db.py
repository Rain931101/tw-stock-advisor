"""SQLite 存取層。"""
import math
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    stock_id TEXT PRIMARY KEY, name TEXT, market TEXT, industry TEXT
);
CREATE TABLE IF NOT EXISTS prices (
    stock_id TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
    volume REAL, value REAL, PRIMARY KEY (stock_id, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
CREATE TABLE IF NOT EXISTS sync_log (
    market TEXT, date TEXT, rows INTEGER, PRIMARY KEY (market, date)
);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, updated_at TEXT, value TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, trade_date TEXT,
    stock_id TEXT, name TEXT, setup TEXT, score REAL, status TEXT,
    plan TEXT, UNIQUE (trade_date, stock_id, setup)
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, rec_id INTEGER, stock_id TEXT, name TEXT,
    setup TEXT, buy_date TEXT, buy_price REAL, shares INTEGER, remaining INTEGER,
    plan TEXT, status TEXT, closed_at TEXT, realized_pnl REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, position_id INTEGER, ts TEXT,
    side TEXT, price REAL, shares INTEGER, reason TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, stock_id TEXT, kind TEXT,
    level TEXT, title TEXT, message TEXT, dedup_key TEXT UNIQUE,
    ref_type TEXT, ref_id INTEGER, sent_line INTEGER DEFAULT 0, read INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_memory = False                  # 測試用記憶體資料庫：只有一條連線，讀取也走主連線
_local = threading.local()       # 每個執行緒自己的唯讀連線


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = get_settings().db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        _migrate(_conn)
    return _conn


def _migrate(c: sqlite3.Connection) -> None:
    cols = {r[1] for r in c.execute("PRAGMA table_info(positions)").fetchall()}
    if "mode" not in cols:
        c.execute("ALTER TABLE positions ADD COLUMN mode TEXT DEFAULT 'cash'")
    if "cost" not in cols:  # 實際總成本（成交金額＋買進手續費），對齊券商帳上
        c.execute("ALTER TABLE positions ADD COLUMN cost REAL")


def reset_for_tests(path: str = ":memory:") -> None:
    global _conn, _memory
    _memory = True
    _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.executescript(SCHEMA)
    _migrate(_conn)


@contextmanager
def tx():
    with _lock:
        c = conn()
        c.execute("BEGIN")
        _local.in_tx = getattr(_local, "in_tx", 0) + 1
        try:
            yield c
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            _local.in_tx -= 1


@contextmanager
def reading():
    """讀取用連線。

    WAL 模式允許多個讀者同時讀，所以大量讀取（全市場日 K、模型特徵）用每個執行緒自己的唯讀連線，
    不佔用主連線的鎖，背景掃描讀資料時其他頁面（持股、通知）不會被卡住。
    記憶體資料庫（測試）或交易進行中（要讀到自己還沒提交的寫入）仍走主連線。
    """
    conn()
    if _memory or getattr(_local, "in_tx", 0):
        with _lock:
            yield _conn
        return
    r = getattr(_local, "reader", None)
    if r is None:
        r = sqlite3.connect(f"file:{get_settings().db_path}?mode=ro", uri=True, check_same_thread=False)
        r.row_factory = sqlite3.Row
        _local.reader = r
    yield r


def query(sql: str, params=()) -> list[dict]:
    if not sql.lstrip()[:6].upper() == "SELECT":
        with _lock:
            return [dict(r) for r in conn().execute(sql, params).fetchall()]
    with reading() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def one(sql: str, params=()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params=()) -> int:
    with _lock:
        cur = conn().execute(sql, params)
        return cur.lastrowid


def executemany(sql: str, rows: list) -> None:
    if not rows:
        return
    with tx() as c:
        c.executemany(sql, rows)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---- 快取 ----
def cache_get(key: str, max_age_hours: float | None = None):
    row = one("SELECT updated_at, value FROM cache WHERE key=?", (key,))
    if not row:
        return None
    if max_age_hours is not None:
        age = (datetime.now() - datetime.fromisoformat(row["updated_at"])).total_seconds() / 3600
        if age > max_age_hours:
            return None
    return json.loads(row["value"])


def cache_set(key: str, value) -> None:
    execute(
        "INSERT OR REPLACE INTO cache(key, updated_at, value) VALUES (?,?,?)",
        (key, now(), dumps(value)),
    )


# ---- 設定 ----
def get_setting(key: str, default=None):
    row = one("SELECT value FROM settings WHERE key=?", (key,))
    return json.loads(row["value"]) if row else default


def set_setting(key: str, value) -> None:
    execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)", (key, json.dumps(value)))


# ---- 價格 ----
def upsert_prices(rows: list[tuple]) -> None:
    with tx() as c:
        c.executemany(
            "INSERT OR REPLACE INTO prices(stock_id,date,open,high,low,close,volume,value) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )


BREAK_DOWN, BREAK_UP = 0.75, 1.35   # 一天跳動超過 ±10% 漲跌幅很多 → 分割、面額變更、減資等公司行動


def split_adjust(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """還原分割／減資造成的價格斷層（例如 0050 在 2025-06-18 一拆四：188.65 → 47.57）。

    只調整「斷層之前」的價格（乘上 當天開盤 ÷ 前一日收盤）與成交量（反向），斷層之後的價格維持實際成交價，
    所以現在的股價、持股成本都不受影響，只是均線、型態、圖表不會被假跳空扭曲。
    一般除息不調整（缺口小，講義的技術分析也以實際價格判斷）。by：多檔資料時的股票欄位。
    """
    if df.empty:
        return df
    groups = [(None, df)] if by is None else df.groupby(by, sort=False)
    parts = []
    for _, g in groups:
        c = g["close"].to_numpy(float)
        o = g["open"].to_numpy(float)
        if len(c) < 2:
            parts.append(g)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = c[1:] / c[:-1]
        brk = np.where((ratio < BREAK_DOWN) | (ratio > BREAK_UP))[0] + 1
        if not len(brk):
            parts.append(g)
            continue
        factor = np.ones(len(c))
        for i in brk[::-1]:
            r = o[i] / c[i - 1] if o[i] and c[i - 1] else None
            if r and 0.02 < r < 50:
                factor[:i] *= r
        g = g.copy()
        for k in ("open", "high", "low", "close"):
            g[k] = g[k].to_numpy(float) * factor
        if "volume" in g:
            g["volume"] = g["volume"].to_numpy(float) / factor
        parts.append(g)
    return parts[0] if by is None else pd.concat(parts)


def load_prices(stock_id: str, limit: int = 400) -> pd.DataFrame:
    with reading() as c:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume, value FROM prices WHERE stock_id=? ORDER BY date DESC LIMIT ?",
            c,
            params=(stock_id, limit),
        )
    df = split_adjust(df.iloc[::-1].reset_index(drop=True))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_all_prices(since: str, adjust: bool = True) -> pd.DataFrame:
    with reading() as c:
        df = pd.read_sql_query(
            "SELECT stock_id, date, open, high, low, close, volume, value FROM prices WHERE date>=? ORDER BY stock_id, date",
            c,
            params=(since,),
        )
    return split_adjust(df, by="stock_id") if adjust else df


def _default(o):
    if hasattr(o, "item"):  # numpy 純量
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_default)


def _finite(o):
    """NaN／Infinity 不是合法 JSON，一律轉成 None（例如財報缺值、量能為 0 的比率）。"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def jsonable(obj):
    return _finite(json.loads(dumps(obj)))


# ---- 備份 ----
def backup(keep: int = 7) -> str:
    """用 SQLite 線上備份 API 複製一份（運作中也安全），保留最近 keep 份。"""
    from datetime import date

    src = get_settings().db_path
    folder = src.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"advisor-{date.today().strftime('%Y%m%d')}.db"
    with _lock:
        target = sqlite3.connect(dst)
        conn().backup(target)
        target.close()
    for old in sorted(folder.glob("advisor-*.db"))[:-keep]:
        old.unlink()
    return str(dst)


def backups() -> list[dict]:
    folder = get_settings().db_path.parent / "backups"
    return [{"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
            for p in sorted(folder.glob("advisor-*.db"), reverse=True)] if folder.exists() else []

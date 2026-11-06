"""SQLite 存取層。"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

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


def reset_for_tests(path: str = ":memory:") -> None:
    global _conn
    _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.executescript(SCHEMA)
    _migrate(_conn)


@contextmanager
def tx():
    with _lock:
        c = conn()
        c.execute("BEGIN")
        try:
            yield c
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise


def query(sql: str, params=()) -> list[dict]:
    with _lock:
        return [dict(r) for r in conn().execute(sql, params).fetchall()]


def one(sql: str, params=()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params=()) -> int:
    with _lock:
        cur = conn().execute(sql, params)
        return cur.lastrowid


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


def load_prices(stock_id: str, limit: int = 400) -> pd.DataFrame:
    with _lock:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume, value FROM prices WHERE stock_id=? ORDER BY date DESC LIMIT ?",
            conn(),
            params=(stock_id, limit),
        )
    df = df.iloc[::-1].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_all_prices(since: str) -> pd.DataFrame:
    with _lock:
        return pd.read_sql_query(
            "SELECT stock_id, date, open, high, low, close, volume, value FROM prices WHERE date>=? ORDER BY stock_id, date",
            conn(),
            params=(since,),
        )


def _default(o):
    if hasattr(o, "item"):  # numpy 純量
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_default)


def jsonable(obj):
    return json.loads(dumps(obj))


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

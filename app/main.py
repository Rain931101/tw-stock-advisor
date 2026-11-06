"""FastAPI 入口：API + 靜態網站。"""
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import db, scheduler
from app.config import get_settings
from app.data import twse
from app.rules import indicators as ind
from app.rules import technical
from app.services import company, market, monitor, notifier, portfolio, recommender, sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
WEB = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.conn()
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="台股選股顧問", lifespan=lifespan)


# ---------------- 狀態 / 設定 ----------------
@app.get("/api/status")
def status():
    s = get_settings()
    return {
        "sync": sync.status,
        "latest_date": sync.latest_trade_date(),
        "stocks": db.one("SELECT COUNT(*) AS n FROM stocks")["n"],
        "line_configured": notifier.line_configured(),
        "finmind_token": bool(s.finmind_token),
        "in_session": monitor.in_session(),
        "fundamentals_cached": db.one("SELECT COUNT(*) AS n FROM cache WHERE key LIKE 'fundamental:%'")["n"],
    }


class SettingsIn(BaseModel):
    capital: float | None = None
    per_trade_capital: float | None = None
    max_loss_per_trade: float | None = None
    fee_discount: float | None = None
    min_avg_volume: float | None = None
    min_price: float | None = None


@app.get("/api/settings")
def get_settings_api():
    rs = recommender.risk_settings()
    rs["max_stop_pct"] = round(rs["max_loss_per_trade"] / rs["per_trade_capital"] * 100, 2)
    return rs


@app.post("/api/settings")
def set_settings_api(body: SettingsIn):
    for k, v in body.model_dump(exclude_none=True).items():
        db.set_setting(k, v)
    return get_settings_api()


# ---------------- 大盤 ----------------
_mk_cache: dict = {}
_mk_lock = threading.Lock()


@app.get("/api/market")
def market_api():
    import time
    with _mk_lock:
        if _mk_cache.get("ts", 0) + 30 > time.time():
            return _mk_cache["data"]
        data = db.jsonable(market.overview())
        _mk_cache.update(ts=time.time(), data=data)
        return data


# ---------------- 推薦 ----------------
@app.get("/api/recommendations")
def recs(status: str = "active", limit: int = 50):
    if status == "all":
        rows = db.query("SELECT * FROM recommendations ORDER BY id DESC LIMIT ?", (limit,))
    else:
        rows = db.query("SELECT * FROM recommendations WHERE status=? ORDER BY trade_date DESC, score DESC LIMIT ?", (status, limit))
    for r in rows:
        r["plan"] = json.loads(r["plan"])
    return rows


@app.post("/api/recommendations/{rid}/dismiss")
def dismiss(rid: int):
    db.execute("UPDATE recommendations SET status='dismissed' WHERE id=?", (rid,))
    return {"ok": True}


# ---------------- 個股 ----------------
@app.get("/api/search")
def search(q: str):
    q = q.strip()
    return db.query("SELECT stock_id, name, market, industry FROM stocks WHERE stock_id LIKE ? OR name LIKE ? LIMIT 15",
                    (f"{q}%", f"%{q}%"))


@app.get("/api/stock/{sid}")
def stock(sid: str, fundamentals: bool = True):
    info = db.one("SELECT * FROM stocks WHERE stock_id=?", (sid,))
    if not info:
        raise HTTPException(404, "查無此股票")
    df = db.load_prices(sid, 420)
    out = {"info": info, "quote": None, "technical": None, "fundamental": None, "chips": None, "bars": []}
    if monitor.in_session():
        try:
            out["quote"] = twse.fetch_realtime([(sid, info["market"])]).get(sid)
            if out["quote"]:
                out["quote"]["time"] = f"盤中 {out['quote']['time']}"
        except Exception:
            pass
    elif len(df) >= 2:
        # 盤後 MIS 不含收盤集合競價、晚上還會重置，改用當日正式收盤
        last, prev = df.iloc[-1], df.iloc[-2]
        out["quote"] = {"price": float(last["close"]), "open": float(last["open"]), "high": float(last["high"]),
                        "low": float(last["low"]), "prev_close": float(prev["close"]), "volume": float(last["volume"]),
                        "time": f"{df.index[-1].strftime('%m/%d')} 收盤"}
    if len(df):
        d = ind.add_indicators(df).tail(260)
        bars = d.reset_index()
        bars["date"] = bars["date"].dt.strftime("%Y-%m-%d")
        out["bars"] = json.loads(bars.replace({float("nan"): None}).to_json(orient="records"))
    if len(df) >= 70:
        q = out["quote"]
        intraday = monitor.in_session() and q and q.get("price")
        src = technical.synthetic_bar(df, q, pd.Timestamp.now().date().isoformat()) if intraday else df
        res = technical.evaluate(src, now=pd.Timestamp.now().to_pydatetime() if intraday else None, intraday=bool(intraday))
        out["technical"] = {"context": res["context"], "signals": [s.to_dict() for s in res["signals"]]}
    if fundamentals:
        try:
            out["fundamental"] = company.fundamental_report(sid)
            out["chips"] = company.chips(sid)
        except Exception as e:
            out["fundamental_error"] = str(e)
    return db.jsonable(out)


# ---------------- 持股 ----------------
class BuyIn(BaseModel):
    stock_id: str
    price: float
    shares: int
    rec_id: int | None = None
    buy_date: str | None = None
    stop: float | None = None


class SellIn(BaseModel):
    price: float
    shares: int
    reason: str = "手動賣出"


class StopIn(BaseModel):
    stop: float


@app.post("/api/positions")
def buy(body: BuyIn):
    if body.shares <= 0 or body.price <= 0:
        raise HTTPException(400, "股數與價格必須大於 0")
    pid = portfolio.confirm_buy(body.rec_id, body.stock_id, body.price, body.shares, body.buy_date, body.stop)
    return {"id": pid}


@app.post("/api/positions/{pid}/sell")
def sell(pid: int, body: SellIn):
    try:
        return portfolio.record_sell(pid, body.price, body.shares, body.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/positions/{pid}/stop")
def set_stop(pid: int, body: StopIn):
    portfolio.update_stop(pid, body.stop)
    return {"ok": True}


@app.get("/api/portfolio/health")
def health():
    return db.jsonable(portfolio.health_check())


@app.get("/api/trades")
def trades(limit: int = 100):
    return db.query("SELECT t.*, p.stock_id, p.name FROM trades t JOIN positions p ON p.id=t.position_id ORDER BY t.id DESC LIMIT ?", (limit,))


# ---------------- 通知 ----------------
@app.get("/api/alerts")
def alerts(limit: int = 100, since_id: int = 0):
    return db.query("SELECT * FROM alerts WHERE id>? ORDER BY id DESC LIMIT ?", (since_id, limit))


@app.post("/api/alerts/read")
def alerts_read():
    db.execute("UPDATE alerts SET read=1 WHERE read=0")
    return {"ok": True}


# ---------------- 手動觸發 ----------------
JOBS = {
    "backfill": lambda: sync.backfill(),
    "scan": lambda: recommender.daily_scan(fetch_fundamentals=True),
    "tick": lambda: monitor.intraday_tick(force=True),
    "fundamentals": lambda: recommender.refresh_fundamentals(60),
    "close_review": lambda: monitor.close_review(),
}


@app.post("/api/jobs/{name}")
def run_job(name: str):
    if name == "test_line":
        ok = notifier.push_line(["✅ 台股選股顧問：LINE 通知設定成功"])
        return {"ok": ok, "configured": notifier.line_configured()}
    if name not in JOBS:
        raise HTTPException(404, "unknown job")
    threading.Thread(target=JOBS[name], daemon=True).start()
    return {"started": name}


# ---------------- 網站 ----------------
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")

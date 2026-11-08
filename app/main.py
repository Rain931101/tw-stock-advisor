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
from app.data import chipdata
from app.ml import model as ml_model
from app.services import company, market, monitor, notifier, portfolio, recommender, sync, winrate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
WEB = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    chipdata.ensure_schema()
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
        "fundamentals": recommender.fund_status,
        "db_path": str(s.db_path.resolve()),
        "backups": db.backups(),
    }


class SettingsIn(BaseModel):
    capital: float | None = None
    per_trade_capital: float | None = None
    max_loss_per_trade: float | None = None
    fee_discount: float | None = None
    fee_monthly_rebate: bool | None = None
    min_avg_volume: float | None = None
    min_price: float | None = None
    margin_enabled: bool | None = None
    margin_ratio_twse: float | None = None
    margin_ratio_tpex: float | None = None
    margin_rate: float | None = None
    margin_maintenance: float | None = None
    winrate_buy_exp: float | None = None
    winrate_min_rate: float | None = None
    winrate_avoid_exp: float | None = None
    winrate_target: float | None = None
    winrate_stop: float | None = None
    winrate_hold: float | None = None


@app.get("/api/settings")
def get_settings_api():
    rs = recommender.risk_settings()
    from app.rules.risk import max_stop_pct
    rs["max_stop_pct"] = round(max_stop_pct(rs["per_trade_capital"], rs["max_loss_per_trade"]) * 100, 2)
    wp = winrate.params()
    rs.update(winrate_buy_exp=wp["buy_exp"], winrate_min_rate=wp["min_rate"], winrate_avoid_exp=wp["avoid_exp"], winrate_target=wp["target"],
              winrate_stop=wp["stop"], winrate_hold=wp["hold_days"])
    rs["warnings"] = []
    if rs["max_loss_per_trade"] >= rs["per_trade_capital"]:
        rs["warnings"].append("單筆最大虧損 ≥ 每筆部位資金：等於不設虧損上限。講義範例是部位 10 萬、最大虧損 1 萬（10%）")
    if rs["per_trade_capital"] > rs["capital"]:
        rs["warnings"].append("每筆部位資金大於帳戶資金")
    if rs["margin_enabled"] and rs["per_trade_capital"] < 100_000:
        rs["warnings"].append("每筆部位資金偏小：融資必須整張買，多數股票買不到 1 張")
    return rs


SETTING_RULES = {  # 欄位: (最小, 最大, 中文名稱)
    "capital": (1, 1e11, "帳戶資金"),
    "per_trade_capital": (1, 1e10, "每筆部位資金"),
    "max_loss_per_trade": (1, 1e10, "單筆最大可承受虧損"),
    "fee_discount": (0.01, 1, "手續費折數（0.3 = 3 折）"),
    "min_avg_volume": (0, 1e7, "20 日均量下限"),
    "min_price": (0, 1e5, "最低股價"),
    "margin_ratio_twse": (0, 0.9, "上市融資成數"),
    "margin_ratio_tpex": (0, 0.9, "上櫃融資成數"),
    "margin_rate": (0, 0.3, "融資年利率（0.065 = 6.5%）"),
    "margin_maintenance": (1, 3, "追繳維持率（1.3 = 130%）"),
    "winrate_buy_exp": (0, 20, "推薦：每筆期望報酬至少（%）"),
    "winrate_min_rate": (0, 1, "推薦：勝率至少（0.5 = 50%）"),
    "winrate_avoid_exp": (-20, 20, "不推薦：每筆期望報酬 ≤（%）"),
    "winrate_target": (0.01, 0.5, "勝率定義：停利（0.08 = +8%）"),
    "winrate_stop": (0.01, 0.3, "勝率定義：停損（0.05 = −5%）"),
    "winrate_hold": (3, 120, "勝率定義：最長持有交易日"),
}
WINRATE_DEFINITION_KEYS = {"winrate_target", "winrate_stop", "winrate_hold", "winrate_buy_exp", "winrate_min_rate", "winrate_avoid_exp"}


@app.post("/api/settings")
def set_settings_api(body: SettingsIn):
    data = body.model_dump(exclude_none=True)
    errors = []
    for k, v in data.items():
        if k in SETTING_RULES:
            lo, hi, name = SETTING_RULES[k]
            if not (lo <= v <= hi):
                errors.append(f"{name} 必須介於 {lo:g} 到 {hi:g}")
    if errors:
        raise HTTPException(400, "；".join(errors))
    changed = {k for k, v in data.items() if k in WINRATE_DEFINITION_KEYS and db.get_setting(k) != v}
    with db.tx() as c:  # 一次寫入，全部成功或全部不寫
        for k, v in data.items():
            c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)", (k, db.dumps(v)))
    if changed:  # 勝率定義或門檻改了 → 背景重新訓練
        threading.Thread(target=winrate.train, daemon=True).start()
    if {"per_trade_capital", "max_loss_per_trade", "fee_discount"} & set(data):
        recommender.revalidate_active()  # 風控變嚴 → 不合格的推薦失效
    return get_settings_api()


@app.post("/api/settings/reset")
def reset_settings_api():
    """恢復預設值（.env 或程式預設）。重設前先把目前設定存一份，可以復原。"""
    current = {r["key"]: r["value"] for r in db.query("SELECT key, value FROM settings")}
    if current:
        db.cache_set("settings_before_reset", current)
    db.execute("DELETE FROM settings")
    return {**get_settings_api(), "can_undo": bool(current)}


@app.post("/api/settings/undo-reset")
def undo_reset_api():
    snap = db.cache_get("settings_before_reset")
    if not snap:
        raise HTTPException(404, "沒有可以復原的設定")
    with db.tx() as c:
        for k, v in snap.items():
            c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)", (k, v))
    db.execute("DELETE FROM cache WHERE key='settings_before_reset'")
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
        db.cache_set("market_verdict", {"stance": data["verdict"]["stance"], "sectors": data.get("sectors") or []})
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


@app.get("/api/winrate")
def winrate_api(limit: int = 30):
    m = winrate.model()
    if not m:
        return {"trained": False, "status": winrate.status}
    rules = m["rules"]
    return {"trained": True, "status": winrate.status, "trained_at": m["trained_at"], "params": m["params"],
            "period": m["period"], "samples": m["samples"], "base_rate": m["base_rate"], "base_exp": m.get("base_exp"),
            "best": m.get("best"), "single": m["single"],
            "top": rules[:limit], "buy_rules": [r for r in rules if r["reliable_buy"]][:limit],
            "buy_count": sum(r["reliable_buy"] for r in rules), "avoid_count": sum(r["reliable_avoid"] for r in rules),
            "total": len(rules), "worst": sorted(rules, key=lambda r: r["exp"])[:10]}


@app.get("/api/winrate/all")
def winrate_all():
    return winrate.all_stocks()


@app.get("/api/winrate/all.csv")
def winrate_all_csv():
    import csv
    import io

    from fastapi.responses import Response
    d = winrate.all_stocks()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["代號", "名稱", "產業", "收盤", "漲跌%", "每筆期望報酬%", "勝率%", "判斷", "綜合組合數", "代表組合", "代表組合樣本", "20日均量(張)", "流動性足夠"])
    label = {"buy": "推薦", "avoid": "不推薦", "neutral": "中間"}
    for r in d.get("rows", []):
        w.writerow([r["stock_id"], r["name"], r["industry"], r["close"], r["change_pct"], r["exp"],
                    round(r["rate"] * 100, 1), label.get(r["decision"], ""), r["n_rules"], r["rule"], r["rule_n"],
                    r["avg_volume"], "是" if r["liquid"] else "否"])
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename=expected_return_{d.get('date', '')}.csv"})


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
    sid = sid.strip()
    info = db.one("SELECT * FROM stocks WHERE stock_id=?", (sid,))
    if not info:  # 也接受輸入股票名稱（完全相符優先，其次唯一的部分相符）
        rows = db.query("SELECT * FROM stocks WHERE name=?", (sid,)) or db.query("SELECT * FROM stocks WHERE name LIKE ? LIMIT 2", (f"%{sid}%",))
        if len(rows) == 1:
            info = rows[0]
            sid = info["stock_id"]
    if not info:
        raise HTTPException(404, f"查無此股票：{sid}（請輸入代號，或從下拉選單選擇）")
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
        out["_res"] = res
    if fundamentals:
        try:
            out["fundamental"] = company.fundamental_report(sid)
        except Exception as e:
            out["fundamental_error"] = str(e)
        try:  # 籌碼抓不到（例如 FinMind 額度用完）不影響基本面與結論
            out["chips"] = company.chips(sid)
        except Exception as e:
            out["chips_error"] = str(e)
    res = out.pop("_res", None)
    if res and res["context"]:
        mv = db.cache_get("market_verdict", 1)  # 一小時內的大盤判斷直接沿用，個股頁不必重算
        if mv is None:
            try:
                mk = market_api()
                mv = {"stance": mk["verdict"]["stance"], "sectors": mk.get("sectors") or []}
            except Exception:
                mv = {"stance": "中性", "sectors": db.cache_get("sectors") or []}
        stance, sectors = mv["stance"], mv["sectors"]
        price = (out["quote"] or {}).get("price") or res["context"]["close"]
        fund = out["fundamental"] if not sid.startswith("00") else None
        out["verdict"] = recommender.stock_verdict(info, res, fund, out["chips"], price, stance, sectors)
        try:
            wr = recommender.winrate_for(sid)
        except Exception as e:
            logging.getLogger(__name__).warning("winrate %s: %s", sid, e)
            wr = None
        out["verdict"] = recommender.apply_winrate(out["verdict"], wr, info, price, fund, out["chips"])
        out["verdict"] = recommender.apply_nearmatch(out["verdict"], info, price, fund, out["chips"])
        out["verdict"] = recommender.apply_ml(out["verdict"], info, price, fund, out["chips"])
    return db.jsonable(out)


# ---------------- 持股 ----------------
class BuyIn(BaseModel):
    stock_id: str
    price: float
    shares: int
    rec_id: int | None = None
    buy_date: str | None = None
    mode: str = "cash"
    cost: float | None = None  # 券商帳上成本（含手續費），選填


class SellIn(BaseModel):
    price: float
    shares: int
    reason: str = "手動賣出"


@app.post("/api/positions")
def buy(body: BuyIn):
    if body.shares <= 0 or body.price <= 0:
        raise HTTPException(400, "股數與價格必須大於 0")
    try:
        return portfolio.confirm_buy(body.rec_id, body.stock_id, body.price, body.shares, body.buy_date, body.mode, body.cost)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/positions/{pid}/sell")
def sell(pid: int, body: SellIn):
    try:
        return portfolio.record_sell(pid, body.price, body.shares, body.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))


class CostIn(BaseModel):
    cost: float


@app.post("/api/positions/{pid}/cost")
def set_cost(pid: int, body: CostIn):
    """對帳：填入券商帳上的總成本（含買進手續費）。"""
    p = db.one("SELECT * FROM positions WHERE id=?", (pid,))
    if not p:
        raise HTTPException(404, "持股不存在")
    if body.cost <= 0:
        raise HTTPException(400, "成本必須大於 0")
    # 使用者填的是目前剩餘股數的成本，換算回原始股數存放
    portfolio.set_cost(pid, body.cost * p["shares"] / p["remaining"])
    return {"ok": True}


@app.get("/api/portfolio/health")
def health():
    return db.jsonable(portfolio.health_check())


@app.get("/api/trades")
def trades(limit: int = 100):
    return db.query("SELECT t.*, p.stock_id, p.name FROM trades t JOIN positions p ON p.id=t.position_id ORDER BY t.id DESC LIMIT ?", (limit,))


@app.post("/api/portfolio/reset")
def reset_portfolio():
    """清空持股與交易紀錄。清空前先做資料庫備份，並保留快照可一鍵復原。"""
    positions = db.query("SELECT * FROM positions")
    trades_ = db.query("SELECT * FROM trades")
    if not positions and not trades_:
        return {"cleared": 0, "can_undo": False}
    db.backup()
    db.cache_set("portfolio_before_reset", {"positions": positions, "trades": trades_})
    with db.tx() as c:
        c.execute("DELETE FROM trades")
        c.execute("DELETE FROM positions")
    return {"cleared": len(positions), "trades": len(trades_), "can_undo": True}


@app.post("/api/portfolio/undo-reset")
def undo_reset_portfolio():
    snap = db.cache_get("portfolio_before_reset")
    if not snap:
        raise HTTPException(404, "沒有可以復原的持股資料")
    with db.tx() as c:
        for table in ("positions", "trades"):
            for row in snap[table]:
                cols = ",".join(row)
                c.execute(f"INSERT OR REPLACE INTO {table}({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
    db.execute("DELETE FROM cache WHERE key='portfolio_before_reset'")
    return {"restored": len(snap["positions"]), "trades": len(snap["trades"])}


# ---------------- 通知 ----------------
@app.get("/api/ml")
def ml_api():
    m = ml_model.meta()
    return db.jsonable({"model": m, "status": ml_model.status, "data_status": chipdata.status,
                        "data": {k: db.one(f"SELECT COUNT(DISTINCT date) AS d FROM {k}")["d"] for k in ("chips_daily", "margin_daily")}
                        if db.one("SELECT name FROM sqlite_master WHERE name='chips_daily'") else {}})


@app.get("/api/alerts")
def alerts(limit: int = 100, since_id: int = 0):
    return db.query("""SELECT a.*, r.status AS rec_status FROM alerts a
                       LEFT JOIN recommendations r ON a.ref_type='recommendation' AND r.id=a.ref_id
                       WHERE a.id>? ORDER BY a.id DESC LIMIT ?""", (since_id, limit))


@app.post("/api/alerts/read")
def alerts_read():
    db.execute("UPDATE alerts SET read=1 WHERE read=0")
    return {"ok": True}


# ---------------- 手動觸發 ----------------
JOBS = {
    "backfill": lambda: sync.backfill(),
    "scan": lambda: recommender.daily_scan(fetch_fundamentals=True),
    "tick": lambda: monitor.intraday_tick(force=True),
    "fundamentals": lambda: recommender.refresh_all_fundamentals(),
    "ml_backfill": lambda: chipdata.backfill(),
    "ml_train": lambda: ml_model.train(),
    "close_review": lambda: monitor.close_review(),
    "backup": lambda: db.backup(),
    "train": lambda: winrate.train(),
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


# ---------------- AI 助理 ----------------
class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None
    context: str | None = None


@app.get("/api/chat/status")
def chat_status():
    from app.services import assistant
    s = get_settings()
    key = s.anthropic_api_key
    return {"configured": assistant.configured(), "model": s.ai_model,
            "key_hint": f"sk-ant-…{key[-4:]}" if key else None}  # 只回傳末四碼，完整金鑰不離開伺服器


class ApiKeyIn(BaseModel):
    key: str


def _write_env(name: str, value: str | None) -> None:
    """更新專案 .env 的單一變數（保留其他行）；value=None 代表刪除。"""
    from pathlib import Path
    path = Path(".env")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [ln for ln in lines if not ln.startswith(f"{name}=")]
    if value is not None:
        lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    path.chmod(0o600)


@app.post("/api/settings/ai-key")
async def set_ai_key(body: ApiKeyIn, test: bool = True):
    import os
    import anthropic
    key = body.key.strip()
    if not key.startswith("sk-ant-") or any(c.isspace() for c in key) or len(key) < 30:
        raise HTTPException(400, "金鑰格式不對：Claude API 金鑰是 sk-ant- 開頭的一長串英數字")
    if test:  # 先用一次極小的請求確認金鑰有效，再寫入
        try:
            await anthropic.AsyncAnthropic(api_key=key, max_retries=1).models.retrieve(get_settings().ai_model)
        except anthropic.AuthenticationError:
            raise HTTPException(400, "金鑰無效（驗證失敗），請重新複製一次")
        except anthropic.PermissionDeniedError:
            raise HTTPException(400, "此金鑰沒有使用該模型的權限")
        except anthropic.NotFoundError:
            raise HTTPException(400, f"找不到模型 {get_settings().ai_model}，請確認 AI_MODEL 設定")
        except anthropic.APIConnectionError:
            raise HTTPException(503, "連不上 Claude API，請確認網路後再試")
    _write_env("ANTHROPIC_API_KEY", key)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    get_settings.cache_clear()
    return chat_status()


@app.delete("/api/settings/ai-key")
def delete_ai_key():
    import os
    _write_env("ANTHROPIC_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    get_settings.cache_clear()
    return chat_status()


@app.post("/api/chat")
async def chat_api(body: ChatIn):
    from fastapi.responses import StreamingResponse
    from app.services import assistant
    if not body.message.strip():
        raise HTTPException(400, "請輸入問題")
    return StreamingResponse(assistant.chat(body.conversation_id, body.message[:4000], body.context),
                             media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/chat/{conv_id}/reset")
def chat_reset(conv_id: str):
    from app.services import assistant
    assistant.reset(conv_id)
    return {"ok": True}


# ---------------- 網站 ----------------
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    # 依 css/js 修改時間加上版本號，避免瀏覽器沿用舊版快取造成畫面格式不一致
    from fastapi.responses import HTMLResponse
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for name in ("style.css", "app.js"):
        ver = int((WEB / name).stat().st_mtime)
        html = html.replace(f"/static/{name}\"", f"/static/{name}?v={ver}\"")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

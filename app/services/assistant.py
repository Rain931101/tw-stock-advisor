"""AI 助理：用 Claude 回答與本系統相關的問題。

- 模型透過「工具」查詢本系統的即時資料（大盤、個股分析、持股健檢、推薦、勝率），回答一律以這些資料為依據
- 對話記錄存在伺服器記憶體（append-only，完整保留每回合的 content），重開伺服器即清空
- 金鑰只從環境變數／.env 讀取（ANTHROPIC_API_KEY）
"""
import asyncio
import json
import logging
import uuid
from datetime import date

import anthropic

from app.config import get_settings

log = logging.getLogger(__name__)

MAX_TOOL_CHARS = 30_000  # 單一工具結果上限，避免把整個資料庫塞進對話
MAX_TURNS = 12  # 單次提問最多呼叫工具的回合數
_conversations: dict[str, list] = {}

SYSTEM = """你是「台股選股顧問」網站內建的 AI 助理，使用者是這套系統的擁有者。

這套系統依兩份 Ken 哥講義運作：
1.《可以用的技術分析》：位階（強／中立／弱）、格局（高過高低不過低才是多方）、角度（震盪／趨勢／急拉）、六種進場型態（箱型突破、站回季線、回測五日線、沿月線趨勢、站回五日線反彈、爆量長黑止跌）、垃圾時間與量縮緩跌不做、停損＝進場要素消失、1:1 波段目標價與健康回檔、以可承受虧損反推部位。
2.《從 0 到 1 選出一家好公司》：毛利率、營益率、ROE／ROA、EPS 跟自己比、營收年增與月增、現金流型態、流動／速動比、應收帳款與存貨異常。
系統另外用歷史資料訓練「條件組合 → 每筆期望報酬／勝率」（隔日開盤買、20 個交易日內先到 +8% 停利或 −5% 停損、到期收盤出場、已扣成本），兩期（訓練／驗證）期望都 ≥ 推薦門檻才推薦。

回答規則：
- 一律使用台灣繁體中文與台灣用語（例如：上市、上櫃、融資、停損、停利、張、成交量）。
- 問到大盤、個股、持股、推薦、勝率或設定時，先用工具查本系統的資料，再根據查到的數字回答；不要憑記憶編造股價或財報。
- 使用者只說股票名稱時，用 search_stock 找代號；個股問題用 analyze_stock。
- 說明系統結論時，講清楚依據哪一條講義規則或哪個統計數字，以及還缺什麼條件。
- 資料缺漏（例如基本面還在更新、法人資料額度用完）要直接說明。
- 回答精簡、條列重點，數字附單位；需要時提醒這是規則與統計判斷，不是投資建議，最終決定由使用者自己做。
"""


def _tool(name: str, description: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "eager_input_streaming": True,
        "input_schema": {"type": "object", "properties": props or {}, "required": required or [],
                         "additionalProperties": False},
    }


TOOLS = [
    _tool("get_market_overview", "取得台股大盤現況：加權與櫃買指數、三大法人、期貨與選擇權、台幣匯率、強弱族群、成交值排行，以及系統的大盤判斷（偏多／中性／防守）。"),
    _tool("search_stock", "用股票名稱或代號關鍵字搜尋，回傳代號、名稱、市場、產業。",
          {"query": {"type": "string", "description": "名稱或代號，例如「京元電」或「2449」"}}, ["query"]),
    _tool("analyze_stock", "取得單一個股的完整分析：報價、位階／格局／角度／量比、六種進場型態檢查（含停損、目標價）、基本面健檢、法人籌碼、系統結論（建議買進／觀望／不建議買進）與歷史期望報酬。",
          {"stock_id": {"type": "string", "description": "股票代號，例如 2330；也可給名稱"}}, ["stock_id"]),
    _tool("get_portfolio", "取得使用者目前持股與健檢：成本、損益平衡點、未實現損益、系統停損價、融資維持率、需要停損／減碼的提醒，以及最近的交易紀錄。"),
    _tool("get_recommendations", "取得目前有效的系統推薦清單（進場區、停損、目標價、報酬風險比、理由）。"),
    _tool("get_winrate_ranking", "取得全市場個股的歷史期望報酬排行（依勝率模型），可篩選與排序。",
          {"sort": {"type": "string", "enum": ["exp_desc", "exp_asc", "rate_desc"], "description": "排序方式，預設 exp_desc"},
           "liquid_only": {"type": "boolean", "description": "只看成交量足夠的股票，預設 true"},
           "limit": {"type": "integer", "description": "筆數，1–50，預設 20"},
           "decision": {"type": "string", "enum": ["any", "buy", "avoid", "neutral"], "description": "依模型判斷篩選，預設 any"}}),
    _tool("get_winrate_model", "取得勝率模型摘要：訓練期間、樣本數、全市場基準勝率與期望報酬、通過驗證的推薦條件組合、最差的條件組合。"),
    _tool("get_settings", "取得使用者的風控與系統設定：資金、每筆部位、單筆最大虧損、最大停損距離、手續費折數、融資條件、勝率門檻。"),
]


# ---------------- 工具實作（呼叫網站既有的 API 函式） ----------------
def _main():
    from app import main  # 延遲載入，避免循環 import
    return main


def _stock(stock_id: str) -> dict:
    d = _main().stock(stock_id)
    bars = d.pop("bars", []) or []
    d["recent_bars"] = [{k: b.get(k) for k in ("date", "open", "high", "low", "close", "volume", "ma5", "ma20", "ma60", "ma240")}
                        for b in bars[-10:]]
    return d


def _portfolio() -> dict:
    m = _main()
    return {"health": m.health(), "recent_trades": m.trades(limit=20)}


def _winrate_ranking(sort: str = "exp_desc", liquid_only: bool = True, limit: int = 20, decision: str = "any") -> dict:
    d = _main().winrate_all()
    rows = d.get("rows") or []
    if liquid_only:
        rows = [r for r in rows if r.get("liquid")]
    if decision and decision != "any":
        rows = [r for r in rows if r.get("decision") == decision]
    key = {"exp_desc": lambda r: -(r.get("exp") or -99), "exp_asc": lambda r: r.get("exp") or 99,
           "rate_desc": lambda r: -(r.get("rate") or 0)}.get(sort, lambda r: -(r.get("exp") or -99))
    rows = sorted(rows, key=key)[: max(1, min(int(limit or 20), 50))]
    return {"date": d.get("date"), "params": d.get("params"), "base_exp": d.get("base_exp"),
            "count_after_filter": len(rows), "rows": rows}


def _winrate_model() -> dict:
    d = _main().winrate_api(limit=15)
    return {k: d.get(k) for k in ("trained", "status", "trained_at", "params", "period", "samples", "base_rate", "base_exp",
                                  "buy_rules", "buy_count", "avoid_count", "total", "top", "worst")}


def run_tool(name: str, args: dict):
    m = _main()
    if name == "get_market_overview":
        return m.market_api()
    if name == "search_stock":
        return m.search(str(args["query"]))
    if name == "analyze_stock":
        return _stock(str(args["stock_id"]))
    if name == "get_portfolio":
        return _portfolio()
    if name == "get_recommendations":
        return m.recs()
    if name == "get_winrate_ranking":
        return _winrate_ranking(**args)
    if name == "get_winrate_model":
        return _winrate_model()
    if name == "get_settings":
        return m.get_settings_api()
    raise ValueError(f"未知的工具：{name}")


_REQUIRED = {t["name"]: (t["input_schema"]["required"], t["input_schema"]["properties"]) for t in TOOLS}
_PY_TYPES = {"string": str, "integer": int, "boolean": bool}


def _validate(name: str, args) -> str | None:
    """eager_input_streaming 下 API 不驗證工具輸入，執行前自己檢查。"""
    if name not in _REQUIRED:
        return f"未知的工具：{name}"
    if not isinstance(args, dict):
        return "輸入必須是物件"
    required, props = _REQUIRED[name]
    for k in required:
        if k not in args:
            return f"缺少參數 {k}"
    for k, v in args.items():
        if k not in props:
            return f"不支援的參數 {k}"
        t = _PY_TYPES.get(props[k]["type"])
        if t and not (isinstance(v, t) and not (t is int and isinstance(v, bool))):
            return f"參數 {k} 型別應為 {props[k]['type']}"
        if "enum" in props[k] and v not in props[k]["enum"]:
            return f"參數 {k} 必須是 {props[k]['enum']} 其中之一"
    return None


def _dump(result) -> str:
    from app import db
    text = json.dumps(db.jsonable(result), ensure_ascii=False)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + "…（資料過長已截斷）"
    return text


TOOL_LABELS = {
    "get_market_overview": "查詢大盤", "search_stock": "搜尋股票", "analyze_stock": "分析個股",
    "get_portfolio": "讀取持股健檢", "get_recommendations": "讀取推薦清單", "get_winrate_ranking": "查詢期望報酬排行",
    "get_winrate_model": "讀取勝率模型", "get_settings": "讀取設定",
}


# ---------------- 對話 ----------------
def configured() -> bool:
    return bool(get_settings().anthropic_api_key)


def _client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key, max_retries=2)


def reset(conv_id: str) -> None:
    _conversations.pop(conv_id, None)


def _event(kind: str, **data) -> str:
    return f"data: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\n\n"


async def chat(conv_id: str | None, text: str, context: str | None = None):
    """SSE 產生器：逐字串流回答；工具呼叫時送出進度事件。"""
    conv_id = conv_id if conv_id in _conversations else (conv_id or uuid.uuid4().hex)
    messages = _conversations.setdefault(conv_id, [])
    yield _event("conv", id=conv_id)
    if not configured():
        yield _event("error", message="尚未設定 ANTHROPIC_API_KEY：請在專案的 .env 加上一行 ANTHROPIC_API_KEY=你的金鑰，然後重新啟動伺服器。")
        return

    s = get_settings()
    content = text.strip()
    if context:
        content = f"{content}\n\n（使用者目前在網站的：{context}）"
    if not messages:
        content = f"（今天是 {date.today().isoformat()}，台北時間）\n{content}"
    messages.append({"role": "user", "content": content})

    client = _client()
    json_retries = 0
    turns = 0
    while turns < MAX_TURNS:
        turns += 1
        try:
            async with client.beta.messages.stream(
                model=s.ai_model,
                max_tokens=16000,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
                cache_control={"type": "ephemeral"},
                thinking={"type": "adaptive"},
                output_config={"effort": s.ai_effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                async for ev in stream:
                    if ev.type == "content_block_delta" and ev.delta.type == "text_delta":
                        yield _event("text", text=ev.delta.text)
                response = await stream.get_final_message()
            json_retries = 0
        except ValueError:  # 工具輸入是完全無法解析的 JSON：重送這一回合（有上限）
            json_retries += 1
            if json_retries > 2:
                yield _event("error", message="AI 產生的工具參數格式錯誤，請換個問法再試一次。")
                return
            turns -= 1
            continue
        except anthropic.AuthenticationError:
            yield _event("error", message="ANTHROPIC_API_KEY 無效，請確認 .env 中的金鑰。")
            return
        except anthropic.PermissionDeniedError as e:
            yield _event("error", message=f"金鑰沒有使用此模型的權限：{e.message}")
            return
        except anthropic.RateLimitError:
            yield _event("error", message="AI 服務請求太頻繁，請稍候一分鐘再問。")
            return
        except anthropic.BadRequestError as e:
            log.warning("chat bad request: %s", e.message)
            yield _event("error", message=f"請求被拒絕：{e.message}")
            return
        except anthropic.APIStatusError as e:
            yield _event("error", message=f"AI 服務暫時無法使用（{e.status_code}），請稍後再試。")
            return
        except anthropic.APIConnectionError:
            yield _event("error", message="連不上 AI 服務，請確認網路連線。")
            return

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            yield _event("error", message="這個問題 AI 無法回答，請換個方式詢問。")
            yield _event("done", id=conv_id)
            return
        if response.stop_reason == "pause_turn":
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            yield _event("done", id=conv_id)
            return
        if response.stop_reason == "max_tokens":
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": "輸出被截斷，未執行"} for b in tool_uses]})
            yield _event("error", message="回答太長被截斷，請把問題拆小一點。")
            return

        results = []
        for b in tool_uses:
            err = _validate(b.name, b.input)
            label = TOOL_LABELS.get(b.name, b.name)
            hint = (b.input or {}).get("stock_id") or (b.input or {}).get("query") if isinstance(b.input, dict) else None
            yield _event("tool", name=b.name, label=f"{label}{f'：{hint}' if hint else ''}")
            if err:
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                "content": json.dumps({"INVALID_INPUT": err, "input": b.input}, ensure_ascii=False)})
                continue
            try:
                out = await asyncio.to_thread(run_tool, b.name, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": _dump(out)})
            except Exception as e:  # 工具失敗回給模型，讓它自己說明或改用其他工具
                detail = getattr(e, "detail", None) or str(e)
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": f"查詢失敗：{detail}"})
        messages.append({"role": "user", "content": results})

    yield _event("error", message="查詢步驟太多，已先停止；請把問題問得更具體一點。")

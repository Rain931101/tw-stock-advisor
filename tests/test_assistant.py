"""AI 助理：用假的 Claude 串流測試工具迴圈（不連網、不需金鑰）。"""
import asyncio
import json
from types import SimpleNamespace as NS

from app.services import assistant


class FakeStream:
    def __init__(self, message, texts):
        self.message, self.texts = message, texts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for t in self.texts:
                yield NS(type="content_block_delta", delta=NS(type="text_delta", text=t))
        return gen()

    async def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, turns):
        self.turns, self.calls = list(turns), []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return FakeStream(*self.turns.pop(0))


def _collect(gen):
    async def run():
        return [json.loads(x[6:]) async for x in gen]
    return asyncio.run(run())


def test_tool_loop(monkeypatch):
    tool_msg = NS(stop_reason="tool_use", content=[
        NS(type="tool_use", id="t1", name="search_stock", input={"query": "台積電"}),
        NS(type="tool_use", id="t2", name="get_winrate_ranking", input={"limit": "5"}),  # 型別錯誤，不應執行
    ])
    final = NS(stop_reason="end_turn", content=[NS(type="text", text="台積電代號 2330")])
    fake = FakeClient([(tool_msg, []), (final, ["台積電", "代號 2330"])])
    monkeypatch.setattr(assistant, "_client", lambda: fake)
    monkeypatch.setattr(assistant, "configured", lambda: True)
    called = []
    monkeypatch.setattr(assistant, "run_tool", lambda n, a: called.append(n) or [{"stock_id": "2330", "name": "台積電"}])

    events = _collect(assistant.chat(None, "台積電代號？"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "conv" and kinds[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "text") == "台積電代號 2330"
    assert called == ["search_stock"]
    # 第二次呼叫帶著 assistant 回合與兩個 tool_result（同一則 user 訊息）
    msgs = fake.calls[1]["messages"]
    results = msgs[-1]["content"]
    assert msgs[-2]["role"] == "assistant" and [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True
    assert fake.calls[0]["fallbacks"] == "default" and fake.calls[0]["model"]
    # 對話被保留，下一題沿用同一段歷史
    conv = events[0]["id"]
    assert len(assistant._conversations[conv]) == 4


def test_not_configured(monkeypatch):
    monkeypatch.setattr(assistant, "configured", lambda: False)
    events = _collect(assistant.chat(None, "hi"))
    assert events[-1]["type"] == "error" and "金鑰" in events[-1]["message"]

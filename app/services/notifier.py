"""通知：寫入站內通知，並透過 LINE Messaging API 推播。

LINE 免費方案每月推播則數有限（依方案而定），因此同一輪監控產生的通知會合併成一則訊息送出，
並用 dedup_key 避免重複通知。
"""
import logging
import uuid

import httpx

from app import db
from app.config import get_settings

log = logging.getLogger(__name__)
LINE_PUSH = "https://api.line.me/v2/bot/message/push"
ICON = {"buy": "🟢", "stop": "🔴", "reduce": "🟠", "target": "🎯", "info": "ℹ️", "market": "📊"}


def alert(kind: str, title: str, message: str, dedup_key: str, stock_id: str | None = None,
          level: str = "info", ref_type: str | None = None, ref_id: int | None = None) -> bool:
    """新增通知；同 dedup_key 已存在則忽略。回傳是否為新通知。"""
    before = db.one("SELECT id FROM alerts WHERE dedup_key=?", (dedup_key,))
    if before:
        return False
    db.execute(
        "INSERT OR IGNORE INTO alerts(ts,stock_id,kind,level,title,message,dedup_key,ref_type,ref_id) VALUES (?,?,?,?,?,?,?,?,?)",
        (db.now(), stock_id, kind, level, title, message, dedup_key, ref_type, ref_id),
    )
    return True


def line_configured() -> bool:
    s = get_settings()
    return bool(s.line_channel_access_token and s.line_user_id)


def push_line(texts: list[str]) -> bool:
    s = get_settings()
    if not line_configured() or not texts:
        return False
    msgs = [{"type": "text", "text": t[:4900]} for t in texts[:5]]
    try:
        r = httpx.post(
            LINE_PUSH,
            headers={"Authorization": f"Bearer {s.line_channel_access_token}", "X-Line-Retry-Key": str(uuid.uuid4())},
            json={"to": s.line_user_id, "messages": msgs},
            timeout=15,
        )
        if r.status_code >= 300:
            log.warning("LINE push failed %s %s", r.status_code, r.text[:200])
            return False
        return True
    except httpx.HTTPError as e:
        log.warning("LINE push error %s", e)
        return False


def flush() -> int:
    """把尚未推播的通知合併送出。"""
    pending = db.query("SELECT * FROM alerts WHERE sent_line=0 ORDER BY id LIMIT 30")
    if not pending:
        return 0
    if not line_configured():
        return 0
    lines = []
    for a in pending:
        lines.append(f"{ICON.get(a['kind'], '•')} {a['title']}\n{a['message']}")
    base = get_settings().public_base_url
    text = "\n\n".join(lines) + f"\n\n👉 {base}"
    chunks, cur = [], ""
    for block in text.split("\n\n"):
        if len(cur) + len(block) > 4500:
            chunks.append(cur)
            cur = ""
        cur += block + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    if push_line(chunks):
        ids = ",".join(str(a["id"]) for a in pending)
        db.execute(f"UPDATE alerts SET sent_line=1 WHERE id IN ({ids})")
        return len(pending)
    return 0

"""排程（台北時間）。"""
import logging
import threading
from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler

from app import db
from app.config import get_settings
from app.services import monitor, recommender, sync

log = logging.getLogger(__name__)
_sched: BackgroundScheduler | None = None


def _safe(fn, *a, **kw):
    def run():
        try:
            log.info("job %s start", fn.__name__)
            out = fn(*a, **kw)
            log.info("job %s done: %s", fn.__name__, out)
        except Exception:
            log.exception("job %s failed", fn.__name__)
    return run


def after_close():
    if not monitor.trading_day():
        return "not trading day"
    from datetime import date
    recommender.remind_unconfirmed(date.today().isoformat())
    n = sync.daily_update()
    monitor.close_review()
    return {"synced": n, "scan": recommender.daily_scan(fetch_fundamentals=False)}


def evening_scan():
    if not monitor.trading_day():
        return "not trading day"
    sync.daily_update()
    return recommender.daily_scan(fetch_fundamentals=True)


def start() -> None:
    global _sched
    s = get_settings()
    if not s.scheduler_enabled or _sched:
        return
    _sched = BackgroundScheduler(timezone="Asia/Taipei")
    second = "5" if s.monitor_interval_sec >= 60 else f"*/{max(s.monitor_interval_sec, 20)}"
    _sched.add_job(_safe(monitor.intraday_tick), "cron", day_of_week="mon-fri", hour="9-13", second=second,
                   max_instances=1, coalesce=True, id="intraday")
    _sched.add_job(_safe(after_close), "cron", day_of_week="mon-fri", hour=14, minute=40, id="after_close")
    # 三大法人與融資券約 15:00~21:00 陸續公布
    _sched.add_job(_safe(evening_scan), "cron", day_of_week="mon-fri", hour=21, minute=30, id="evening_scan")
    _sched.add_job(_safe(recommender.refresh_fundamentals, 120), "cron", hour=22, minute=30, id="fund_daily")
    _sched.add_job(_safe(recommender.refresh_fundamentals, 500), "cron", day_of_week="sat", hour=9, id="fund_weekly")
    _sched.add_job(_safe(db.backup), "cron", hour=23, minute=50, id="backup")
    _sched.start()
    if not any(b["file"].endswith(f"{date.today():%Y%m%d}.db") for b in db.backups()):
        threading.Thread(target=_safe(db.backup), daemon=True).start()
    threading.Thread(target=_safe(sync.backfill), daemon=True).start()


def shutdown() -> None:
    if _sched:
        _sched.shutdown(wait=False)

"""排程（台北時間）。"""
import logging
import threading
from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler

from app import db
from app.config import get_settings
from app.services import monitor, recommender, sync, winrate

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
    db.cache_set(f"evening_done:{date.today().isoformat()}", True)
    try:  # 三大法人、融資券（機器學習特徵）
        from app.data import chipdata
        chipdata.sync_today()
    except Exception:
        log.exception("chipdata sync failed")
    try:  # 臺指期日資料（明日大盤預測）
        from app.data import futures
        futures.sync_recent()
        from app.services import forecast
        forecast.predict(force=True)
    except Exception:
        log.exception("futures/forecast failed")
    return recommender.daily_scan(fetch_fundamentals=True)


_catch_lock = threading.Lock()


def catch_up() -> str:
    """電腦睡眠／關機錯過排程時補做：交易日 14:35 後還沒有今天的日 K → 收盤同步＋掃描；
    21:35 後還沒有今天的法人／融資券 → 晚間掃描。每 15 分鐘與啟動時各檢查一次。"""
    from datetime import datetime
    now = datetime.now()
    if now.weekday() >= 5 or now.hour * 60 + now.minute < 14 * 60 + 35:
        return "not yet"
    if not _catch_lock.acquire(blocking=False):
        return "running"
    try:
        today = now.date().isoformat()
        has_prices = db.one("SELECT 1 AS x FROM prices WHERE date=? LIMIT 1", (today,)) is not None
        if not has_prices:
            if not monitor.trading_day(now):
                return "not trading day"
            log.info("catch-up: 補做收盤同步與掃描")
            return f"after_close: {after_close()}"
        if now.hour * 60 + now.minute >= 21 * 60 + 35:
            has_chips = db.one("SELECT 1 AS x FROM ml_sync WHERE kind='margin' AND date=?", (today,)) is not None
            if not has_chips and not db.cache_get(f"evening_done:{today}"):
                log.info("catch-up: 補做晚間掃描")
                out = evening_scan()
                db.cache_set(f"evening_done:{today}", True)
                return f"evening: {out}"
        return "up to date"
    finally:
        _catch_lock.release()


def start() -> None:
    global _sched
    s = get_settings()
    if not s.scheduler_enabled or _sched:
        return
    _sched = BackgroundScheduler(timezone="Asia/Taipei",
                                 job_defaults={"coalesce": True, "misfire_grace_time": 6 * 3600, "max_instances": 1})
    second = "5" if s.monitor_interval_sec >= 60 else f"*/{max(s.monitor_interval_sec, 20)}"
    _sched.add_job(_safe(monitor.intraday_tick), "cron", day_of_week="mon-fri", hour="9-13", second=second,
                   max_instances=1, coalesce=True, misfire_grace_time=30, id="intraday")
    # 盤中預備訊號：每 3 分鐘（比進場判斷慢，獨立排程不拖慢每分鐘的檢查）
    _sched.add_job(_safe(monitor.pre_signal_tick), "cron", day_of_week="mon-fri", hour="9-13", minute="*/3", second=30,
                   max_instances=1, coalesce=True, misfire_grace_time=60, id="pre_signal")
    _sched.add_job(_safe(after_close), "cron", day_of_week="mon-fri", hour=14, minute=40, id="after_close")
    # 三大法人與融資券約 15:00~21:00 陸續公布
    _sched.add_job(_safe(evening_scan), "cron", day_of_week="mon-fri", hour=21, minute=30, id="evening_scan")
    # 全市場基本面：每週六重新整理（7 天內已更新的會跳過）；每天晚上補上漏掉的
    _sched.add_job(_safe(recommender.refresh_all_fundamentals), "cron", hour=22, minute=30, id="fund_daily")
    _sched.add_job(_safe(recommender.refresh_all_fundamentals), "cron", day_of_week="sat", hour=9, id="fund_weekly")
    _sched.add_job(_safe(db.backup), "cron", hour=23, minute=50, id="backup")
    _sched.add_job(_safe(winrate.train), "cron", day_of_week="sat", hour=8, id="winrate_weekly")  # 每週重新訓練勝率
    from app.ml import model as ml_model
    _sched.add_job(_safe(ml_model.train), "cron", day_of_week="sat", hour=8, minute=30, id="ml_weekly")
    from app.services import forecast
    _sched.add_job(_safe(forecast.train), "cron", day_of_week="sat", hour=9, minute=30, id="forecast_weekly")  # 每週重新訓練大盤預測  # 每週重新訓練 AI 模型
    _sched.add_job(_safe(catch_up), "interval", minutes=15, id="catch_up", misfire_grace_time=600)
    _sched.start()
    threading.Thread(target=_safe(catch_up), daemon=True).start()
    if not any(b["file"].endswith(f"{date.today():%Y%m%d}.db") for b in db.backups()):
        threading.Thread(target=_safe(db.backup), daemon=True).start()
    threading.Thread(target=_safe(sync.backfill), daemon=True).start()
    if not winrate.model():
        threading.Thread(target=_safe(winrate.train), daemon=True).start()
    threading.Thread(target=_safe(forecast.predict), daemon=True).start()   # 預先算好明日大盤預測
    if ml_model.meta():  # 預先算好今日 AI 模型預測，個股頁不必等
        threading.Thread(target=_safe(ml_model.predict_today), daemon=True).start()
    if db.get_setting("fundamentals_full_pending", False):  # 上次全市場更新被中斷 → 自動接續
        threading.Thread(target=_safe(recommender.refresh_all_fundamentals), daemon=True).start()


def shutdown() -> None:
    if _sched:
        _sched.shutdown(wait=False)

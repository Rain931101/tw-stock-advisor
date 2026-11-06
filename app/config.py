"""設定：一律從環境變數或 .env 讀取，程式碼內不放任何金鑰。"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 資料
    db_path: Path = Path("data/advisor.db")
    finmind_token: str = ""
    backfill_days: int = 400  # 日曆天，MA240 約需 1 年以上資料

    # LINE Messaging API（LINE Notify 已於 2025/3 停止服務）
    line_channel_access_token: str = ""
    line_user_id: str = ""
    public_base_url: str = "http://localhost:8000"

    # 預設風控（可在網站「設定」頁修改，存於資料庫）
    capital: float = 1_000_000  # 帳戶資金
    per_trade_capital: float = 100_000  # 每筆部位資金
    max_loss_per_trade: float = 10_000  # 單筆最大可承受虧損
    fee_discount: float = 0.3  # 手續費折數（講義建議 3 折）
    fee_monthly_rebate: bool = True  # 券商月退（元大等）：帳上以原價手續費計，折讓月底退回

    # 融資（槓桿）：預設值為台股常見條件，請依自己券商的信用交易條件修改
    margin_enabled: bool = True  # 有信用帳戶才開
    margin_ratio_twse: float = 0.6  # 上市融資成數
    margin_ratio_tpex: float = 0.5  # 上櫃融資成數
    margin_rate: float = 0.065  # 融資年利率
    margin_maintenance: float = 1.30  # 維持率低於此即追繳

    # 選股母體
    min_avg_volume: int = 1000  # 20 日均量（張）
    min_price: float = 10.0

    # 盤中監控
    monitor_interval_sec: int = 60
    scheduler_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()

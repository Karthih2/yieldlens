from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "YieldLens Backend API"
    VERSION: str = "1.0.0"
    MODEL_VERSION: str = "v1.0.0-rf-reduced"
    
    BASE_DIR: Path = BASE_DIR
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts" / "v1"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'yieldlens.db'}"
    
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ]
    
    INSPECTION_CAPACITY_K: int = 30
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    
    # Stability Tier Thresholds
    STABILITY_TIER_HIGH: float = 0.4
    STABILITY_TIER_MEDIUM: float = 0.2
    STABILITY_TIER_LOW: float = 0.0

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

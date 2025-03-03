from pydantic_settings import BaseSettings
from typing import List
from datetime import datetime
import pytz

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5000",
        "http://0.0.0.0:5000",
    ]

    # Environment
    DEBUG: bool = True

    def get_current_time(self) -> str:
        """Returns current UTC time in ISO format"""
        return datetime.now(pytz.UTC).isoformat()

    class Config:
        case_sensitive = True
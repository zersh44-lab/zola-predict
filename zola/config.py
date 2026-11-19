"""Configuration management with secure environment variable loading.

Never prints, logs, or exposes secret values. Reports only whether keys
are present or missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded safely from environment."""

    # API Keys (strictly redacted in __repr__ and string conversions)
    football_api_key: Optional[str] = None
    football_data_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Alternate AI Provider Keys
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ai_provider: str = "auto"  # "gemini", "groq", "openrouter", "ollama", or "auto"

    # Football API Provider
    football_api_provider: str = "api-football"  # "api-football" or "football-data"
    football_api_base_url: Optional[str] = None

    # Database and Cache
    db_path: Path = Path("data/zola_predict.db")
    cache_ttl_minutes: int = 60
    rate_limit_rpm: int = 10  # Free tier default: 10 requests per minute

    def __repr__(self) -> str:
        """Prevent secrets from leaking in repr, logs, or interactive shells."""
        return (
            f"Settings("
            f"football_api_key={'***PRESENT***' if self.football_api_key else 'None'}, "
            f"gemini_api_key={'***PRESENT***' if self.gemini_api_key else 'None'}, "
            f"telegram_bot_token={'***PRESENT***' if self.telegram_bot_token else 'None'}, "
            f"telegram_chat_id={'***PRESENT***' if self.telegram_chat_id else 'None'}, "
            f"groq_api_key={'***PRESENT***' if self.groq_api_key else 'None'}, "
            f"openrouter_api_key={'***PRESENT***' if self.openrouter_api_key else 'None'}, "
            f"ollama_base_url='{self.ollama_base_url}', "
            f"ai_provider='{self.ai_provider}', "
            f"football_api_provider='{self.football_api_provider}', "
            f"football_api_base_url={self.football_api_base_url!r}, "
            f"db_path={self.db_path!r}, "
            f"cache_ttl_minutes={self.cache_ttl_minutes}, "
            f"rate_limit_rpm={self.rate_limit_rpm})"
        )

    def check_variable(self, name: str) -> str:
        """Return 'PRESENT' or 'MISSING' for a given variable name."""
        val = getattr(self, name.lower(), None)
        if val is None:
            val = os.getenv(name)
        return "PRESENT" if bool(val and str(val).strip()) else "MISSING"

    def get_credentials_status(self) -> Dict[str, str]:
        """Return a mapping of variable name to PRESENT or MISSING status."""
        return {
            "FOOTBALL_API_KEY": "PRESENT" if bool(self.football_api_key and self.football_api_key.strip()) else "MISSING",
            "GEMINI_API_KEY": "PRESENT" if bool(self.gemini_api_key and self.gemini_api_key.strip()) else "MISSING",
            "TELEGRAM_BOT_TOKEN": "PRESENT" if bool(self.telegram_bot_token and self.telegram_bot_token.strip()) else "MISSING",
            "TELEGRAM_CHAT_ID": "PRESENT" if bool(self.telegram_chat_id and self.telegram_chat_id.strip()) else "MISSING",
            "GROQ_API_KEY": "PRESENT" if bool(self.groq_api_key and self.groq_api_key.strip()) else "MISSING",
            "OPENROUTER_API_KEY": "PRESENT" if bool(self.openrouter_api_key and self.openrouter_api_key.strip()) else "MISSING",
        }


def load_settings(env_file: Optional[Path | str] = None) -> Settings:
    """Load configuration from .env file and environment variables safely."""
    if env_file is not None:
        target_path = Path(env_file)
        if target_path.exists():
            load_dotenv(dotenv_path=target_path, override=True)
    else:
        # Default search in current working directory and project root
        potential_env = Path(".env")
        if potential_env.exists():
            load_dotenv(dotenv_path=potential_env, override=False)
        else:
            load_dotenv()

    # Read values safely
    raw_football_key = os.getenv("FOOTBALL_API_KEY", "").strip() or None
    raw_football_data_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip() or None
    raw_gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or None
    raw_telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
    raw_telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
    raw_groq_key = os.getenv("GROQ_API_KEY", "").strip() or None
    raw_openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ai_prov = os.getenv("AI_PROVIDER", "auto").strip().lower()

    provider = os.getenv("FOOTBALL_API_PROVIDER", "api-football").strip().lower()
    base_url = os.getenv("FOOTBALL_API_BASE_URL", "").strip() or None

    db_path_str = os.getenv("ZOLA_DB_PATH", "data/zola_predict.db").strip()
    db_path = Path(db_path_str)

    cache_ttl_str = os.getenv("ZOLA_CACHE_TTL_MINUTES", "60").strip()
    try:
        cache_ttl = int(cache_ttl_str)
    except ValueError:
        cache_ttl = 60

    rate_limit_str = os.getenv("ZOLA_RATE_LIMIT_RPM", "10").strip()
    try:
        rate_limit = int(rate_limit_str)
    except ValueError:
        rate_limit = 10

    return Settings(
        football_api_key=raw_football_key,
        football_data_api_key=raw_football_data_key,
        gemini_api_key=raw_gemini_key,
        telegram_bot_token=raw_telegram_token,
        telegram_chat_id=raw_telegram_chat_id,
        groq_api_key=raw_groq_key,
        openrouter_api_key=raw_openrouter_key,
        ollama_base_url=ollama_url,
        ai_provider=ai_prov,
        football_api_provider=provider,
        football_api_base_url=base_url,
        db_path=db_path,
        cache_ttl_minutes=cache_ttl,
        rate_limit_rpm=rate_limit,
    )

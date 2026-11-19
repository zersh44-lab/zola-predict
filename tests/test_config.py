"""Tests for secure configuration and credential handling."""

import os
from pathlib import Path

from zola.config import Settings, load_settings


def test_settings_credential_masking():
    """Ensure API credentials are never exposed in string representation or logs."""
    dummy_secret = "secret_key_12345_football"
    settings = Settings(
        football_api_key=dummy_secret,
        gemini_api_key="secret_gemini_key",
        telegram_bot_token="secret_telegram_token",
    )

    repr_str = repr(settings)
    assert dummy_secret not in repr_str
    assert "secret_gemini_key" not in repr_str
    assert "secret_telegram_token" not in repr_str
    assert "***PRESENT***" in repr_str


def test_settings_status_reporting():
    """Verify check_variable and get_credentials_status report only PRESENT or MISSING."""
    settings = Settings(
        football_api_key="some_key",
        gemini_api_key="",
        telegram_bot_token=None,
    )
    status = settings.get_credentials_status()

    assert status["FOOTBALL_API_KEY"] == "PRESENT"
    assert status["GEMINI_API_KEY"] == "MISSING"
    assert status["TELEGRAM_BOT_TOKEN"] == "MISSING"


def test_load_settings_from_custom_env(tmp_path: Path):
    """Test loading settings from a temporary .env file."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "FOOTBALL_API_KEY=test_provider_key\n"
        "FOOTBALL_API_PROVIDER=football-data\n"
        "ZOLA_RATE_LIMIT_RPM=15\n"
        "ZOLA_CACHE_TTL_MINUTES=45\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file)
    assert settings.football_api_provider == "football-data"
    assert settings.rate_limit_rpm == 15
    assert settings.cache_ttl_minutes == 45
    assert settings.check_variable("FOOTBALL_API_KEY") == "PRESENT"
    # Ensure raw secret is not in string representation
    assert "test_provider_key" not in str(settings)

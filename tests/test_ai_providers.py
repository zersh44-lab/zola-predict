"""Tests for AI research providers and statistical fallback without live API keys."""

from zola.config import Settings
from zola.research.ai import StatisticalFallbackProvider, get_ai_provider


def test_statistical_fallback_provider():
    """Verify statistical fallback generates structured research without external API calls."""
    provider = StatisticalFallbackProvider()
    assert provider.is_available()

    fixture_info = {
        "home_team_name": "Arsenal",
        "away_team_name": "Chelsea",
        "competition_name": "Premier League",
        "utc_date": "2024-12-10 16:00:00",
    }
    mock_predictions = {}
    summary = provider.generate_research_summary(fixture_info, mock_predictions)

    assert "Arsenal vs Chelsea" in summary
    assert "Premier League" in summary
    assert "Consensus Forecast" in summary


def test_ai_provider_factory_fallback():
    """Verify factory returns StatisticalFallbackProvider when no keys are configured."""
    settings = Settings(
        gemini_api_key=None,
        groq_api_key=None,
        ai_provider="auto",
        ollama_base_url="http://invalid-ollama-host:9999",
    )
    provider = get_ai_provider(settings)
    assert provider.name == "statistical_fallback"

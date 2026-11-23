"""Implementation of AI providers for Gemini, Groq, OpenRouter, local Ollama, and statistical fallback."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import requests

from zola.config import Settings
from zola.research.ai.base import BaseAiProvider

logger = logging.getLogger(__name__)


def _serialize_predictions(preds: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {}
    for model_name, markets in preds.items():
        if isinstance(markets, dict):
            serialized[model_name] = {
                m: p.to_dict() if hasattr(p, "to_dict") else p
                for m, p in markets.items()
            }
        else:
            serialized[model_name] = markets.to_dict() if hasattr(markets, "to_dict") else markets
    return serialized


class StatisticalFallbackProvider(BaseAiProvider):
    """Fallback provider when no external AI keys are configured; synthesizes deterministic research."""

    @property
    def name(self) -> str:
        return "statistical_fallback"

    def is_available(self) -> bool:
        return True

    def generate_research_summary(
        self,
        fixture_info: Dict[str, Any],
        predictions: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        home_team = fixture_info.get("home_team_name", "Home Team")
        away_team = fixture_info.get("away_team_name", "Away Team")
        competition = fixture_info.get("competition_name", "League")
        date_str = fixture_info.get("utc_date", "Upcoming")

        ensemble = predictions.get("ensemble", {})
        m_dc = ensemble.get("DOUBLE_CHANCE")
        m_ou15 = ensemble.get("OU15")
        m_ou05 = ensemble.get("OU05")

        # Extract high-confidence report if present
        hc_report = None
        for p in ensemble.values():
            feat = p.get("features", {}) if isinstance(p, dict) else (getattr(p, "features", {}) or {})
            if isinstance(feat, dict) and "high_confidence_report" in feat:
                hc_report = feat["high_confidence_report"]
                break

        lines = [
            f"### Research Brief: {home_team} vs {away_team}",
            f"- **Competition**: {competition} ({date_str})",
            "- **Active Market Policy**: Prioritizing verified high-win-rate markets (`DOUBLE_CHANCE`, `OU15`, `OU05`). High-loss markets (`1X2`, `BTTS`, `OU25`, `OU35`, `DNB`) are strictly excluded.",
        ]

        if hc_report and hc_report.get("is_skipped"):
            lines.append(f"- **High-Confidence Status**: ⚠️ **SKIP THIS MATCH** — {hc_report.get('skip_rationale')}")
        elif hc_report and hc_report.get("picks"):
            lines.append(f"- **High-Confidence Selections ({hc_report.get('total_picks')} of Max 3)**:")
            for pick in hc_report.get("picks", []):
                lines.append(f"  • **{pick.get('outcome_display')}** ({pick.get('probability_pct')} prob, {pick.get('confidence')} Conf, @ {pick.get('estimated_odd')})")
        else:
            if m_dc:
                dc_pick = m_dc.get("predicted_outcome") if isinstance(m_dc, dict) else getattr(m_dc, "predicted_outcome", "N/A")
                dc_prob = m_dc.get("probability", 0) if isinstance(m_dc, dict) else getattr(m_dc, "probability", 0)
                lines.append(f"- **Double Chance Forecast**: **{dc_pick}** ({dc_prob * 100:.1f}%)")
            if m_ou15:
                o15_pick = m_ou15.get("predicted_outcome") if isinstance(m_ou15, dict) else getattr(m_ou15, "predicted_outcome", "N/A")
                o15_prob = m_ou15.get("probability", 0) if isinstance(m_ou15, dict) else getattr(m_ou15, "probability", 0)
                lines.append(f"- **Over/Under 1.5 Goals**: **{o15_pick}** ({o15_prob * 100:.1f}%)")

        return "\n".join(lines)


class GeminiProvider(BaseAiProvider):
    """Google Gemini AI research provider using Google AI REST API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self.model = model

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate_research_summary(
        self,
        fixture_info: Dict[str, Any],
        predictions: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self.is_available():
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self._api_key}"
        headers = {"Content-Type": "application/json"}

        prompt = (
            f"You are the football analyst for Zola Predict.\n"
            f"Fixture: {fixture_info.get('home_team_name')} vs {fixture_info.get('away_team_name')}\n"
            f"Competition: {fixture_info.get('competition_name')}\n"
            f"Statistical Predictions: {json.dumps(_serialize_predictions(predictions))}\n\n"
            "Provide a concise, evidence-based research breakdown highlighting tactical form, consensus probabilities, "
            "and caveats without claiming absolute certainty."
        )

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)
        except Exception as e:
            logger.warning("Gemini API call failed, falling back to statistical summary: %s", e)
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)


class GroqProvider(BaseAiProvider):
    """Groq AI provider using ultra-fast LLM inference."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self.model = model

    @property
    def name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate_research_summary(
        self,
        fixture_info: Dict[str, Any],
        predictions: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self.is_available():
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional football analyst for Zola Predict."},
                {"role": "user", "content": f"Analyze: {fixture_info.get('home_team_name')} vs {fixture_info.get('away_team_name')}."},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=12)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)
        except Exception:
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)


class OllamaProvider(BaseAiProvider):
    """Local Ollama AI provider (100% offline, zero cloud keys)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3:latest"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_research_summary(
        self,
        fixture_info: Dict[str, Any],
        predictions: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self.is_available():
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)

        url = f"{self.base_url}/api/generate"
        prompt = f"Analyze match: {fixture_info.get('home_team_name')} vs {fixture_info.get('away_team_name')}."
        body = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(url, json=body, timeout=20)
            if resp.status_code == 200:
                return resp.json().get("response", "")
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)
        except Exception:
            return StatisticalFallbackProvider().generate_research_summary(fixture_info, predictions)


def get_ai_provider(settings: Settings) -> BaseAiProvider:
    """Factory selecting the appropriate AI provider based on configuration."""
    choice = (settings.ai_provider or "auto").lower().strip()

    if choice == "gemini" and settings.gemini_api_key:
        return GeminiProvider(api_key=settings.gemini_api_key)
    elif choice == "groq" and settings.groq_api_key:
        return GroqProvider(api_key=settings.groq_api_key)
    elif choice == "ollama":
        provider = OllamaProvider(base_url=settings.ollama_base_url)
        if provider.is_available():
            return provider

    # Auto detection
    if settings.gemini_api_key:
        return GeminiProvider(api_key=settings.gemini_api_key)
    if settings.groq_api_key:
        return GroqProvider(api_key=settings.groq_api_key)

    ollama_check = OllamaProvider(base_url=settings.ollama_base_url)
    if ollama_check.is_available():
        return ollama_check

    # Fallback to statistical reasoning
    return StatisticalFallbackProvider()

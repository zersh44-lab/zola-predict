"""Base classes and contracts for AI football research providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAiProvider(ABC):
    """Abstract base class for AI reasoning and football research providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'groq', 'ollama')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider has valid local configuration or endpoint reachable."""
        pass

    @abstractmethod
    def generate_research_summary(
        self,
        fixture_info: Dict[str, Any],
        predictions: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate structured football research summary based on match data."""
        pass

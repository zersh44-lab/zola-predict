"""Data models for Zola Predict SQLite database."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class Competition:
    """Football competition / league entity."""

    external_id: str
    provider: str
    name: str
    code: Optional[str] = None
    country: Optional[str] = None
    emblem_url: Optional[str] = None
    current_season: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)
    updated_at: Optional[str] = None


@dataclass
class Team:
    """Football team / club entity."""

    external_id: str
    provider: str
    name: str
    short_name: Optional[str] = None
    tla: Optional[str] = None
    crest_url: Optional[str] = None
    venue: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)
    updated_at: Optional[str] = None


@dataclass
class Fixture:
    """Match fixture entity."""

    external_id: str
    provider: str
    competition_id: int
    utc_date: str
    status: str
    home_team_id: int
    away_team_id: int
    season: Optional[str] = None
    matchday: Optional[int] = None
    venue: Optional[str] = None
    referee: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)
    updated_at: Optional[str] = None


@dataclass
class MatchResult:
    """Historical or finished match score/result entity."""

    fixture_id: int
    winner: Optional[str] = None
    duration: Optional[str] = "REGULAR"
    full_time_home_score: Optional[int] = None
    full_time_away_score: Optional[int] = None
    half_time_home_score: Optional[int] = None
    half_time_away_score: Optional[int] = None
    extra_time_home_score: Optional[int] = None
    extra_time_away_score: Optional[int] = None
    penalty_home_score: Optional[int] = None
    penalty_away_score: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)
    updated_at: Optional[str] = None


@dataclass
class ResearchSource:
    """Information or data provider source metadata."""

    name: str
    source_type: str  # "API", "WEB", "YOUTUBE", "MANUAL", "CSV"
    status: str = "ACTIVE"
    url: Optional[str] = None
    notes: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)
    updated_at: Optional[str] = None


@dataclass
class ApiCollectionLog:
    """Audit log entry for all external API interactions."""

    provider: str
    endpoint: str
    status_code: int
    response_time_ms: int
    cache_hit: bool = False
    error_message: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CachedRequest:
    """Persisted HTTP request cache entry."""

    cache_key: str
    method: str
    url: str
    status_code: int
    response_body: str
    created_at: str
    expires_at: str
    params_json: Optional[str] = None


@dataclass
class MonitoredSource:
    """A user-selected account, channel, or website to monitor."""

    platform: str
    name: str
    url: str
    status: str = "ACTIVE"
    id: Optional[int] = None
    last_checked_at: Optional[str] = None
    last_error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class SourceItem:
    """A post or page collected from a monitored source."""

    source_id: int
    external_id: str
    url: str
    title: Optional[str] = None
    published_at: Optional[str] = None
    content_text: Optional[str] = None
    content_hash: Optional[str] = None
    id: Optional[int] = None
    collected_at: Optional[str] = None


@dataclass
class Prediction:
    """Model prediction for a specific market on a fixture."""

    fixture_id: int
    model_name: str
    market: str
    predicted_outcome: str
    probability: float
    probabilities_json: str
    confidence: str = "MEDIUM"
    confidence_score: float = 0.5
    data_quality_notes: Optional[str] = None
    features_json: Optional[str] = None
    status: str = "PENDING"
    is_correct: Optional[int] = None
    settlement_status: Optional[str] = "PENDING"
    model_version: Optional[str] = "v1.1.0"
    odds_json: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class HistoricalDataImport:
    """Provenance record for imported historical datasets."""

    source_name: str
    filename: str
    file_hash: str
    season: Optional[str] = None
    competition_code: Optional[str] = None
    rows_imported: int = 0
    notes: Optional[str] = None
    id: Optional[int] = None
    imported_at: Optional[str] = None


@dataclass
class AccuracyMetric:
    """Aggregated accuracy performance metrics per model and market."""

    model_name: str
    market: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy_rate: float = 0.0
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    ece: Optional[float] = None
    push_predictions: int = 0
    void_predictions: int = 0
    id: Optional[int] = None
    last_evaluated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric entry to dictionary for JSON serialization."""
        return asdict(self)

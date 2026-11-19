"""Database package for Zola Predict."""

from zola.database.connection import get_connection, get_db_cursor, init_db
from zola.database.models import (
    AccuracyMetric,
    ApiCollectionLog,
    CachedRequest,
    Competition,
    Fixture,
    HistoricalDataImport,
    MatchResult,
    MonitoredSource,
    Prediction,
    ResearchSource,
    SourceItem,
    Team,
)
from zola.database.repository import FootballRepository

__all__ = [
    "get_connection",
    "get_db_cursor",
    "init_db",
    "FootballRepository",
    "Competition",
    "Team",
    "Fixture",
    "MatchResult",
    "ResearchSource",
    "MonitoredSource",
    "SourceItem",
    "ApiCollectionLog",
    "CachedRequest",
    "Prediction",
    "HistoricalDataImport",
    "AccuracyMetric",
]

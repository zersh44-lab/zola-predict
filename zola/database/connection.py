"""Database connection and schema initialization for SQLite."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS competitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    name TEXT NOT NULL,
    code TEXT,
    country TEXT,
    emblem_url TEXT,
    current_season TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, external_id)
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    name TEXT NOT NULL,
    short_name TEXT,
    tla TEXT,
    crest_url TEXT,
    venue TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, external_id)
);

CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    competition_id INTEGER NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
    season TEXT,
    matchday INTEGER,
    utc_date TIMESTAMP NOT NULL,
    status TEXT NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    away_team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    venue TEXT,
    referee TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, external_id)
);

CREATE TABLE IF NOT EXISTS match_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL UNIQUE REFERENCES fixtures(id) ON DELETE CASCADE,
    winner TEXT,
    duration TEXT DEFAULT 'REGULAR',
    full_time_home_score INTEGER,
    full_time_away_score INTEGER,
    half_time_home_score INTEGER,
    half_time_away_score INTEGER,
    extra_time_home_score INTEGER,
    extra_time_away_score INTEGER,
    penalty_home_score INTEGER,
    penalty_away_score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    url TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitored_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL CHECK(platform IN ('youtube', 'tiktok', 'telegram', 'website')),
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'PAUSED')),
    last_checked_at TIMESTAMP,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES monitored_sources(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    title TEXT,
    url TEXT NOT NULL,
    published_at TIMESTAMP,
    content_text TEXT,
    content_hash TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, external_id)
);

CREATE TABLE IF NOT EXISTS api_collection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS request_cache (
    cache_key TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    url TEXT NOT NULL,
    params_json TEXT,
    status_code INTEGER NOT NULL,
    response_body TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS historical_data_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    season TEXT,
    competition_code TEXT,
    rows_imported INTEGER NOT NULL DEFAULT 0,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL REFERENCES fixtures(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    market TEXT NOT NULL,
    predicted_outcome TEXT NOT NULL,
    probability REAL NOT NULL,
    probabilities_json TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT 'MEDIUM',
    confidence_score REAL DEFAULT 0.5,
    data_quality_notes TEXT,
    features_json TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    is_correct INTEGER,
    settlement_status TEXT DEFAULT 'PENDING',
    model_version TEXT DEFAULT 'v1.1.0',
    odds_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fixture_id, model_name, market)
);

CREATE TABLE IF NOT EXISTS accuracy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    market TEXT NOT NULL,
    total_predictions INTEGER NOT NULL DEFAULT 0,
    correct_predictions INTEGER NOT NULL DEFAULT 0,
    accuracy_rate REAL NOT NULL DEFAULT 0.0,
    brier_score REAL,
    log_loss REAL,
    ece REAL,
    push_predictions INTEGER NOT NULL DEFAULT 0,
    void_predictions INTEGER NOT NULL DEFAULT 0,
    last_evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, market)
);

CREATE TABLE IF NOT EXISTS user_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    item_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(item_type, item_id)
);

CREATE TABLE IF NOT EXISTS prioritization_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telegram_recipients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT UNIQUE NOT NULL,
    username TEXT,
    display_name TEXT,
    type TEXT DEFAULT 'private',
    is_default INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance and lookup
CREATE INDEX IF NOT EXISTS idx_fixtures_competition ON fixtures(competition_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_utc_date ON fixtures(utc_date);
CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(status);
CREATE INDEX IF NOT EXISTS idx_fixtures_home_team ON fixtures(home_team_id);
CREATE INDEX IF NOT EXISTS idx_fixtures_away_team ON fixtures(away_team_id);
CREATE INDEX IF NOT EXISTS idx_api_logs_created ON api_collection_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_monitored_sources_status ON monitored_sources(status);
CREATE INDEX IF NOT EXISTS idx_source_items_published ON source_items(published_at);
CREATE INDEX IF NOT EXISTS idx_request_cache_expires ON request_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON predictions(fixture_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_market ON predictions(market);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_historical_imports_hash ON historical_data_imports(file_hash);
"""


def resolve_db_path(db_path: Union[str, Path] = "data/zola_predict.db", force_reseed: bool = False) -> Path:
    """Resolve database path, seeding into /tmp if running in a serverless environment like Vercel."""
    target = Path(db_path) if not isinstance(db_path, Path) else db_path
    if str(target) == ":memory:":
        return target

    # Check for serverless runtime (Vercel / AWS Lambda)
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        # If path is already inside /tmp, return as is unless force_reseed is requested
        if (str(target).startswith("/tmp") or str(target).startswith("\\tmp")) and not force_reseed:
            return target

        tmp_db = Path("/tmp/zola_predict.db")
        candidates = [
            target,
            Path(__file__).resolve().parent.parent.parent / "data" / "zola_predict.db",
            Path(os.getcwd()) / "data" / "zola_predict.db",
            Path("/var/task/data/zola_predict.db"),
            Path.cwd() / "data" / "zola_predict.db",
        ]
        cand_found = None
        for cand in candidates:
            try:
                if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                    cand_found = cand
                    break
            except Exception:
                continue

        if cand_found:
            should_copy = False
            seed_marker = Path("/tmp/.zola_db_seed_version")
            expected_version = f"{cand_found.stat().st_size}_{cand_found.stat().st_mtime}"

            if force_reseed:
                should_copy = True
            elif not tmp_db.exists() or tmp_db.stat().st_size == 0:
                should_copy = True
            elif not seed_marker.exists() or seed_marker.read_text(encoding="utf-8").strip() != expected_version:
                # New deployment bundle detected
                should_copy = True
            else:
                # Verify tmp_db integrity before using
                try:
                    test_conn = sqlite3.connect(str(tmp_db))
                    check = test_conn.execute("PRAGMA quick_check;").fetchone()
                    test_conn.close()
                    if not check or check[0] != "ok":
                        logger.warning("Corrupted database detected in /tmp. Re-seeding from bundle.")
                        should_copy = True
                except Exception:
                    logger.warning("Error reading /tmp database. Re-seeding from bundle.")
                    should_copy = True

            if should_copy:
                try:
                    import shutil
                    tmp_db.parent.mkdir(parents=True, exist_ok=True)
                    # Safely remove any existing corrupted files and journals
                    for ext in ["", "-wal", "-shm", "-journal"]:
                        f_to_rem = Path(f"{tmp_db}{ext}")
                        try:
                            if f_to_rem.exists():
                                f_to_rem.unlink()
                        except Exception:
                            pass
                    shutil.copy2(cand_found, tmp_db)
                    try:
                        seed_marker.write_text(expected_version, encoding="utf-8")
                    except Exception:
                        pass
                except Exception as e:
                    logger.error("Failed to copy database to /tmp: %s", e)

        # Fallback initialization if database was not found
        if not tmp_db.exists() or tmp_db.stat().st_size == 0:
            try:
                init_db(tmp_db)
            except Exception:
                pass

        if tmp_db.exists():
            return tmp_db

    return target


def get_connection(db_path: Union[str, Path] = "data/zola_predict.db") -> sqlite3.Connection:
    """Create and configure a connection to the SQLite database."""
    target_path = resolve_db_path(db_path)
    if str(target_path) != ":memory:":
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # Read-only filesystem

    try:
        conn = sqlite3.connect(str(target_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            check = conn.execute("PRAGMA quick_check;").fetchone()
            if not check or check[0] != "ok":
                raise sqlite3.DatabaseError("Corrupted database detected in serverless /tmp")
        return conn
    except sqlite3.DatabaseError as de:
        logger.warning("Database connection error (%s). Re-seeding database from bundle.", de)
        target_path = resolve_db_path(db_path, force_reseed=True)
        conn = sqlite3.connect(str(target_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


@contextmanager
def get_db_cursor(db_path: Union[str, Path] = "data/zola_predict.db") -> Generator[sqlite3.Cursor, None, None]:
    """Context manager providing a transactional database cursor."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Union[str, Path] = "data/zola_predict.db") -> None:
    """Initialize SQLite database with required tables, constraints, and indexes."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)

        # Safe schema migrations for prediction intelligence upgrades
        cols_to_add = [
            ("accuracy_metrics", "log_loss", "REAL"),
            ("accuracy_metrics", "ece", "REAL"),
            ("accuracy_metrics", "push_predictions", "INTEGER DEFAULT 0"),
            ("accuracy_metrics", "void_predictions", "INTEGER DEFAULT 0"),
            ("predictions", "settlement_status", "TEXT DEFAULT 'PENDING'"),
            ("predictions", "model_version", "TEXT DEFAULT 'v1.1.0'"),
            ("predictions", "odds_json", "TEXT"),
        ]
        for tbl, col, col_type in cols_to_add:
            try:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_type};")
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        # Seed default research sources if empty
        conn.execute(
            """
            INSERT INTO research_sources (name, source_type, url, status, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                "API-Football",
                "API",
                "https://v3.football.api-sports.io",
                "ACTIVE",
                "Primary live football data provider for competitions, fixtures, and results",
            ),
        )
        conn.execute(
            """
            INSERT INTO research_sources (name, source_type, url, status, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """,
            (
                "football-data.org",
                "API",
                "https://api.football-data.org/v4",
                "ACTIVE",
                "Optional live football data provider; separate key and coverage apply",
            ),
        )
        conn.execute(
            """
            INSERT INTO research_sources (name, source_type, url, status, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """,
            (
                "Historical FootballData CSVs",
                "CSV",
                "https://github.com/jokecamp/FootballData",
                "ACTIVE",
                "Curated historical match results repository for backtesting and statistical models",
            ),
        )
        conn.commit()

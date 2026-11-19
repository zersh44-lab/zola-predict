"""Historical CSV data importer with strict provenance and format validation.

Supports jokecamp/FootballData and Football-Data.co.uk CSV structures.
Ensures historical CSVs are clearly labeled and never treated as live data.
"""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from zola.database.models import (
    Competition,
    Fixture,
    HistoricalDataImport,
    MatchResult,
    Team,
)
from zola.database.repository import FootballRepository

# Common Division Code mappings for Football-Data.co.uk / jokecamp/FootballData
DIVISION_NAMES = {
    "E0": ("Premier League", "England"),
    "E1": ("Championship", "England"),
    "E2": ("League One", "England"),
    "SP1": ("La Liga", "Spain"),
    "D1": ("Bundesliga", "Germany"),
    "I1": ("Serie A", "Italy"),
    "F1": ("Ligue 1", "France"),
}


class HistoricalCsvImporter:
    """Imports historical match data from CSV files with full provenance tracking."""

    def __init__(self, repository: FootballRepository):
        self.repository = repository

    @staticmethod
    def compute_file_hash(content: bytes) -> str:
        """Compute SHA-256 hash of CSV content for provenance tracking."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def parse_date(date_str: str) -> Optional[str]:
        """Parse varied date formats (DD/MM/YYYY, DD/MM/YY, YYYY-MM-DD) to ISO format."""
        date_str = date_str.strip()
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d 15:00:00")
            except ValueError:
                continue
        return None

    def import_csv(
        self,
        file_path_or_content: Union[str, Path, bytes],
        filename: str = "matches.csv",
        source_name: str = "jokecamp/FootballData",
        season: Optional[str] = None,
        default_competition_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate and import historical matches from CSV.

        Guarantees that imported data is tagged with provider 'football-data-csv'
        so it is never confused with live API data.
        """
        if isinstance(file_path_or_content, (str, Path)):
            path = Path(file_path_or_content)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            content_bytes = path.read_bytes()
            filename = path.name
        else:
            content_bytes = file_path_or_content

        file_hash = self.compute_file_hash(content_bytes)

        # Parse CSV text
        try:
            text = content_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1")

        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)

        if not rows:
            return {"rows_imported": 0, "message": "CSV file was empty."}

        # Validate minimum expected columns
        header = [c.strip() for c in (reader.fieldnames or [])]
        required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required.issubset(set(header)):
            raise ValueError(
                f"Invalid CSV structure. Missing required columns: {required - set(header)}. "
                f"Present columns: {header[:8]}..."
            )

        provider_name = "football-data-csv"
        rows_imported = 0
        teams_created = 0

        # Track competitions seen
        comp_id_map: Dict[str, int] = {}

        for r in rows:
            home_name = r.get("HomeTeam", "").strip()
            away_name = r.get("AwayTeam", "").strip()
            fthg_str = r.get("FTHG", "").strip()
            ftag_str = r.get("FTAG", "").strip()

            if not home_name or not away_name or not fthg_str or not ftag_str:
                continue

            try:
                home_goals = int(fthg_str)
                away_goals = int(ftag_str)
            except ValueError:
                continue

            raw_date = r.get("Date", "").strip()
            utc_date = self.parse_date(raw_date) if raw_date else "2023-01-01 15:00:00"

            div_code = r.get("Div", "").strip() or (default_competition_code or "E0")
            if div_code not in comp_id_map:
                comp_name, country = DIVISION_NAMES.get(div_code, (f"Division {div_code}", "International"))
                cid = self.repository.upsert_competition(
                    Competition(
                        external_id=div_code,
                        provider=provider_name,
                        name=comp_name,
                        code=div_code,
                        country=country,
                        current_season=season,
                    )
                )
                comp_id_map[div_code] = cid

            comp_id = comp_id_map[div_code]

            # Upsert Teams
            home_id = self.repository.upsert_team(
                Team(
                    external_id=f"csv_team_{home_name.lower().replace(' ', '_')}",
                    provider=provider_name,
                    name=home_name,
                    short_name=home_name,
                )
            )
            away_id = self.repository.upsert_team(
                Team(
                    external_id=f"csv_team_{away_name.lower().replace(' ', '_')}",
                    provider=provider_name,
                    name=away_name,
                    short_name=away_name,
                )
            )

            # Deterministic unique external ID for historical fixture
            clean_date_tag = utc_date[:10].replace("-", "") if utc_date else "nodate"
            fixture_ext_id = f"csv_{div_code}_{clean_date_tag}_{home_name[:4]}_{away_name[:4]}".lower()

            # Upsert Fixture
            fix_id = self.repository.upsert_fixture(
                Fixture(
                    external_id=fixture_ext_id,
                    provider=provider_name,
                    competition_id=comp_id,
                    utc_date=utc_date or "2023-01-01 15:00:00",
                    status="FINISHED",
                    home_team_id=home_id,
                    away_team_id=away_id,
                    season=season,
                )
            )

            # Winner evaluation
            if home_goals > away_goals:
                winner = "HOME_TEAM"
            elif home_goals == away_goals:
                winner = "DRAW"
            else:
                winner = "AWAY_TEAM"

            # Optional half time goals
            hthg = int(r["HTHG"].strip()) if r.get("HTHG", "").strip().isdigit() else None
            htag = int(r["HTAG"].strip()) if r.get("HTAG", "").strip().isdigit() else None

            # Upsert Match Result
            self.repository.upsert_match_result(
                MatchResult(
                    fixture_id=fix_id,
                    winner=winner,
                    duration="REGULAR",
                    full_time_home_score=home_goals,
                    full_time_away_score=away_goals,
                    half_time_home_score=hthg,
                    half_time_away_score=htag,
                )
            )

            rows_imported += 1

        # Record Provenance
        notes = f"Imported {rows_imported} matches from historical dataset '{filename}'"
        import_rec = HistoricalDataImport(
            source_name=source_name,
            filename=filename,
            file_hash=file_hash,
            season=season,
            competition_code=default_competition_code or "E0",
            rows_imported=rows_imported,
            notes=notes,
        )
        self.repository.record_historical_import(import_rec)

        return {
            "source_name": source_name,
            "filename": filename,
            "file_hash": file_hash[:16] + "...",
            "rows_imported": rows_imported,
            "provider": provider_name,
        }

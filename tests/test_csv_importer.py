"""Tests for Historical FootballData CSV importer and provenance tracking."""

from pathlib import Path
import pytest

from zola.data_sources.csv_importer import HistoricalCsvImporter
from zola.database.repository import FootballRepository

SAMPLE_CSV_CONTENT = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR
E0,12/08/2023,Arsenal,Nott'm Forest,2,1,H,2,0,H
E0,12/08/2023,Bournemouth,West Ham,1,1,D,0,0,D
E0,12/08/2023,Brighton,Luton,4,1,H,1,0,H
E0,12/08/2023,Everton,Fulham,0,1,A,0,0,D
E0,12/08/2023,Sheffield United,Crystal Palace,0,1,A,0,0,D
"""


def test_import_valid_csv(repo: FootballRepository, tmp_path: Path):
    """Test importing historical CSV with provenance and separation from live data."""
    csv_file = tmp_path / "E0_test.csv"
    csv_file.write_text(SAMPLE_CSV_CONTENT, encoding="utf-8")

    importer = HistoricalCsvImporter(repo)
    result = importer.import_csv(
        file_path_or_content=csv_file,
        filename="E0_test.csv",
        source_name="jokecamp/FootballData",
        season="2023-2024",
        default_competition_code="E0",
    )

    assert result["rows_imported"] == 5
    assert result["provider"] == "football-data-csv"

    # Verify provenance recorded in SQLite
    imports = repo.get_historical_imports()
    assert len(imports) == 1
    assert imports[0].source_name == "jokecamp/FootballData"
    assert imports[0].filename == "E0_test.csv"
    assert imports[0].rows_imported == 5

    # Verify matches stored as FINISHED with results
    fixtures = repo.get_fixtures()
    assert len(fixtures) == 5
    for f in fixtures:
        assert f["provider"] == "football-data-csv"
        assert f["status"] == "FINISHED"
        assert f["full_time_home_score"] is not None


def test_import_invalid_csv_structure(repo: FootballRepository, tmp_path: Path):
    """Test importer rejects CSVs missing required columns."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("ColA,ColB,ColC\n1,2,3\n", encoding="utf-8")

    importer = HistoricalCsvImporter(repo)
    with pytest.raises(ValueError) as exc:
        importer.import_csv(bad_csv)

    assert "Missing required columns" in str(exc.value)

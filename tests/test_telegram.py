"""Unit tests for Telegram reporting service without making external network calls."""

import pytest
from datetime import datetime
from zola.database.models import Competition, Team, Fixture
from zola.telegram.reporter import TelegramReportSender
from tests.conftest import MockHttpResponse


def test_telegram_is_configured():
    sender = TelegramReportSender(token="test_token_123")
    assert sender.is_configured() is True

    empty_sender = TelegramReportSender(token="")
    assert empty_sender.is_configured() is False

    none_sender = TelegramReportSender(token=None)
    assert none_sender.is_configured() is False


def test_get_bot_info(monkeypatch):
    sender = TelegramReportSender(token="test_token_123")

    def mock_get(url, timeout=10):
        assert "getMe" in url
        return MockHttpResponse(200, {"ok": True, "result": {"id": 123456, "is_bot": True, "username": "TestBot"}})

    monkeypatch.setattr("requests.get", mock_get)
    info = sender.get_bot_info()
    assert info is not None
    assert info["username"] == "TestBot"


def test_get_recent_chat_ids(monkeypatch):
    sender = TelegramReportSender(token="test_token_123")

    def mock_get(url, timeout=10):
        assert "getUpdates" in url
        return MockHttpResponse(200, {
            "ok": True,
            "result": [
                {
                    "update_id": 100,
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 987654321, "type": "private", "username": "football_fan"}
                    }
                }
            ]
        })

    monkeypatch.setattr("requests.get", mock_get)
    chats = sender.get_recent_chat_ids()
    assert len(chats) == 1
    assert chats[0]["chat_id"] == 987654321
    assert chats[0]["username"] == "football_fan"


def test_send_message_success(monkeypatch):
    sender = TelegramReportSender(token="test_token_123")
    calls = []

    def mock_post(url, json=None, timeout=12):
        calls.append({"url": url, "json": json})
        return MockHttpResponse(200, {"ok": True, "result": {"message_id": 999}})

    monkeypatch.setattr("requests.post", mock_post)
    success = sender.send_message(chat_id=987654321, text="Test prediction match report")
    assert success is True
    assert len(calls) == 1
    assert calls[0]["json"]["chat_id"] == "987654321"


def test_send_message_retry_plain_text_on_parse_error(monkeypatch):
    sender = TelegramReportSender(token="test_token_123")
    calls = []

    def mock_post(url, json=None, timeout=12):
        calls.append(dict(json))
        if len(calls) == 1:
            # First call with parse_mode fails with 400
            return MockHttpResponse(400, {"ok": False, "description": "Bad Request: can't parse entities"}, raw_text="can't parse entities")
        # Second call without parse_mode succeeds
        return MockHttpResponse(200, {"ok": True, "result": {"message_id": 1000}})

    monkeypatch.setattr("requests.post", mock_post)
    success = sender.send_message(chat_id=987654321, text="Unmatched *bold bracket [test")
    assert success is True
    assert len(calls) == 2
    assert "parse_mode" in calls[0]
    assert "parse_mode" not in calls[1]


def test_send_fixture_report(repo, monkeypatch):
    # Setup test fixture in repo
    c = Competition(external_id="PL_TEST", name="Premier League", code="PL", provider="mock")
    c_id = repo.upsert_competition(c)

    t1 = Team(external_id="ARS_TEST", name="Arsenal", short_name="Arsenal", tla="ARS", provider="mock")
    t2 = Team(external_id="CHE_TEST", name="Chelsea", short_name="Chelsea", tla="CHE", provider="mock")
    t1_id = repo.upsert_team(t1)
    t2_id = repo.upsert_team(t2)

    f = Fixture(
        external_id="FIX_1001",
        competition_id=c_id,
        home_team_id=t1_id,
        away_team_id=t2_id,
        season="2024",
        utc_date=datetime(2024, 11, 25, 15, 0, 0),
        status="SCHEDULED",
        provider="mock",
    )
    f_id = repo.upsert_fixture(f)

    sent_payloads = []

    def mock_post(url, json=None, timeout=12):
        sent_payloads.append(json)
        return MockHttpResponse(200, {"ok": True})

    monkeypatch.setattr("requests.post", mock_post)

    sender = TelegramReportSender(token="test_token_123")
    delivered = sender.send_fixture_report(chat_id="12345", fixture_id=f_id, repository=repo)

    assert delivered is True
    assert len(sent_payloads) == 1
    text = sent_payloads[0]["text"]
    assert "ZOLA PREDICT" in text
    assert "Arsenal" in text
    assert "Chelsea" in text
    assert "HIGH-CONFIDENCE REPORT" in text

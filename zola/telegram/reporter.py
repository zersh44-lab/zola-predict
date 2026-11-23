"""Telegram bot delivery service for sending match prediction reports to the user."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from zola.config import Settings
from zola.database.repository import FootballRepository
from zola.prediction.engine import PredictionEngine
from zola.validation.datetime import format_kickoff_ethiopia, format_kickoff_utc, parse_to_utc
from zola.validation.fixture import validate_fixture_for_prediction
from zola.validation.odds import APPROVED_MARKETS, calculate_expected_value

logger = logging.getLogger(__name__)


class TelegramReportSender:
    """Delivers formatted prediction cards to the user via Telegram Bot API."""

    def __init__(self, token: Optional[str] = None):
        self._token = token.strip() if token and token.strip() else None
        self.base_url = f"https://api.telegram.org/bot{self._token}" if self._token else ""

    def is_configured(self) -> bool:
        """Check if Telegram bot token is present without disclosing it."""
        return bool(self._token)

    def get_bot_info(self) -> Optional[Dict[str, Any]]:
        """Verify bot token and return bot identity."""
        if not self.is_configured():
            return None
        try:
            resp = requests.get(f"{self.base_url}/getMe", timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                return resp.json().get("result")
        except Exception as e:
            logger.warning("Failed to contact Telegram API: %s", e)
        return None

    def get_recent_chat_ids(self) -> List[Dict[str, Any]]:
        """Fetch recent interactions to discover the user's chat_id."""
        if not self.is_configured():
            return []
        try:
            resp = requests.get(f"{self.base_url}/getUpdates", timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                updates = resp.json().get("result", [])
                chats = {}
                for u in updates:
                    msg = u.get("message") or u.get("channel_post") or {}
                    chat = msg.get("chat") or {}
                    cid = chat.get("id")
                    if cid and cid not in chats:
                        chats[cid] = {
                            "chat_id": cid,
                            "type": chat.get("type"),
                            "username": chat.get("username") or chat.get("first_name", "User"),
                        }
                return list(chats.values())
        except Exception as e:
            logger.warning("Error fetching Telegram updates: %s", e)
        return []

    def send_message(self, chat_id: int | str, text: str, parse_mode: Optional[str] = "Markdown") -> bool:
        """Send formatted text message to a specific Telegram chat."""
        if not self.is_configured():
            raise ValueError("Telegram Bot Token is not configured.")

        url = f"{self.base_url}/sendMessage"
        payload: Dict[str, Any] = {
            "chat_id": str(chat_id),
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200 and resp.json().get("ok", False):
                return True
            # If markdown entity parsing failed, retry with plain text
            if parse_mode is not None and resp.status_code == 400:
                logger.warning("Telegram parse failed (%s), retrying as plain text", resp.text)
                payload.pop("parse_mode", None)
                retry_resp = requests.post(url, json=payload, timeout=12)
                return retry_resp.status_code == 200 and retry_resp.json().get("ok", False)
            logger.error("Failed to send Telegram message to chat %s: %s", chat_id, resp.text)
            return False
        except Exception as e:
            logger.error("Failed to send Telegram message to chat %s: %s", chat_id, e)
            return False

    def resolve_chat_id(
        self,
        target: str,
        repository: Optional[FootballRepository] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve a user input (numeric chat_id, @username, or username) into a valid Telegram destination.
        
        Returns:
            tuple of (resolved_chat_id, error_message).
        """
        if not target or not str(target).strip():
            return None, "No Telegram user or Chat ID provided."

        raw = str(target).strip()
        clean_user = raw.lstrip("@").strip()

        # 1. Pure integer / signed integer (Chat ID, group/channel ID)
        if raw.lstrip("-").isdigit():
            return raw, None

        # 2. Check stored database recipients first
        if repository:
            rec = repository.find_telegram_recipient(raw)
            if rec and rec.get("chat_id"):
                return str(rec["chat_id"]), None

        # 3. Check live updates from Telegram bot (discovered chats)
        discovered = self.get_recent_chat_ids()
        for chat in discovered:
            u = str(chat.get("username") or "").lower()
            cid = str(chat.get("chat_id"))
            if u == clean_user.lower() or cid == raw:
                if repository:
                    repository.upsert_telegram_recipient(
                        chat_id=cid,
                        username=chat.get("username"),
                        display_name=chat.get("username") or f"Chat {cid}",
                    )
                return cid, None

        # 4. If target starts with '@', it may be a public channel or group username
        if raw.startswith("@"):
            return raw, None

        # 5. User not found - provide actionable guidance
        bot_info = self.get_bot_info()
        bot_tag = f"@{bot_info.get('username')}" if bot_info and bot_info.get("username") else "the bot"
        msg = (
            f"Could not find Telegram user '{raw}'. "
            f"Telegram privacy rules require users to first send /start to {bot_tag} "
            f"before the bot can message them. Alternatively, you can enter the user's numeric Chat ID directly."
        )
        return None, msg

    def send_test_message(self, chat_id: int | str, repository: Optional[FootballRepository] = None) -> bool:
        """Send a test greeting to confirm bot communication with chat_id."""
        resolved_cid, err = self.resolve_chat_id(str(chat_id), repository=repository)
        if not resolved_cid:
            logger.error("Resolve failed: %s", err)
            return False
        msg = (
            "⚽ *Zola Predict — Bot Connected!*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Your Telegram delivery integration is operational.\n"
            "You will receive match research & prediction reports here."
        )
        ok = self.send_message(chat_id=resolved_cid, text=msg)
        if ok and repository:
            repository.touch_telegram_recipient(resolved_cid)
        return ok

    def validate_report_payload(
        self,
        fixture: Dict[str, Any],
        hc_report: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """Validate all required fields before allowing Telegram publication.

        The published report must contain:
        - Correct competition
        - Correct home and away teams
        - Correct match date and kickoff time
        - Correct approved market (OU05, OU15)
        - Correct probability
        - Correct verified odds (> 1.01)
        - Correct positive EV (> 0.0)
        - Correct confidence status
        """
        # 1. Validate fixture
        is_fix_valid, fix_rej = validate_fixture_for_prediction(fixture, enforce_future_kickoff=False)
        if not is_fix_valid:
            return False, f"Publication blocked: {fix_rej}"

        if not hc_report:
            return False, "Publication blocked: Missing high-confidence evaluation report."

        # If report is skipped, skip card is valid to publish
        if hc_report.get("is_skipped"):
            return True, None

        picks = hc_report.get("picks", [])
        if not picks:
            return False, "Publication blocked: Report has no picks and is not marked as skipped."

        for pick in picks:
            market = pick.get("market")
            if market not in APPROVED_MARKETS:
                return False, f"Publication blocked: Unapproved market '{market}' in report picks."

            prob = pick.get("probability")
            if prob is None or not (0.0 < float(prob) < 1.0):
                return False, f"Publication blocked: Invalid probability '{prob}' for {market}."

            m_odd = pick.get("market_odd")
            if m_odd is None or float(m_odd) <= 1.01:
                return False, f"Publication blocked: Missing or invalid verified odds for {market}."

            ev = pick.get("expected_value")
            if ev is None or float(ev) <= 0.0:
                return False, f"Publication blocked: Non-positive expected value ({ev}) for {market}."

        return True, None

    def format_match_report(
        self,
        fixture: Dict[str, Any],
        predictions: Dict[str, Any],
    ) -> str:
        """Format a rich match report card for Telegram adhering strictly to active market allowlist."""
        home_team = fixture.get("home_team_name", "Home Team")
        away_team = fixture.get("away_team_name", "Away Team")
        competition = fixture.get("competition_name", "League")
        raw_utc = fixture.get("utc_date", "")
        try:
            eat_str = format_kickoff_ethiopia(raw_utc)
            utc_str = format_kickoff_utc(raw_utc)[:16].replace("T", " ") + " UTC"
        except Exception:
            utc_str = str(raw_utc)[:16].replace("T", " ") + " UTC"
            eat_str = ""

        ensemble = predictions.get("ensemble", {})

        # Extract high-confidence report if present
        hc_report = None
        for p in ensemble.values():
            feat = p.get("features", {}) if isinstance(p, dict) else (getattr(p, "features", {}) or {})
            if isinstance(feat, dict) and "high_confidence_report" in feat:
                hc_report = feat["high_confidence_report"]
                break

        lines = [
            "⚽ *ZOLA PREDICT — HIGH-CONFIDENCE REPORT*",
            "━━━━━━━━━━━━━━━━━━━━",
            f"🏆 *{competition}*",
            f"⚔️ *{home_team} vs {away_team}*",
            f"📅 `{utc_str}`" + (f" (`{eat_str}`)" if eat_str else ""),
            "",
        ]

        if not hc_report or hc_report.get("is_skipped") or not hc_report.get("picks"):
            skip_msg = (
                hc_report.get("skip_rationale")
                if hc_report and hc_report.get("skip_rationale")
                else "Match does not meet strict eligibility rules (verified odds, positive EV, approved market)."
            )
            lines.extend([
                "⚠️ *NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH.*",
                f"💡 _{skip_msg}_",
                "",
                "🔒 _Market Policy: Only verified high-yield markets (OU05, OU15) with verified pre-match odds and positive EV are eligible._",
                "ℹ️ _Note: Model probability is a statistical estimate and does not represent verified betting value without verified pre-match odds._",
            ])
        else:
            lines.append("🎯 *Selected High-Confidence Options (Max 3):*")
            for pick in hc_report.get("picks", []):
                # Ensure market is strictly in approved allowlist
                if pick.get("market") not in APPROVED_MARKETS:
                    continue
                rank = pick.get("rank", 1)
                outcome = pick.get("outcome_display", "")
                prob = pick.get("probability_pct", "")
                conf = pick.get("confidence", "HIGH")
                m_odd = pick.get("market_odd")
                if not m_odd or float(m_odd) <= 1.01:
                    continue
                odd_str = f"@{float(m_odd):.2f}"
                ev_val = pick.get("expected_value")
                ev_str = f"{float(ev_val)*100:+.1f}%" if ev_val is not None else "N/A"
                val_status = pick.get("betting_value_status", "VERIFIED_POSITIVE_EV")
                rationale = pick.get("rationale", "")
                risks = pick.get("risks", "")
                lines.extend([
                    f"*{rank}. {outcome}*",
                    f"   • Model Probability: `{prob}` | Confidence: `{conf}` (Algorithmic)",
                    f"   • Bookmaker Odd: `{odd_str}` | Expected Value: `{ev_str}` ({val_status})",
                    f"   • Rationale: _{rationale}_",
                    f"   • Key Risk: _{risks}_",
                    "",
                ])

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━",
            "🤖 _Quality Over Quantity • Zola Predict_",
        ])

        return "\n".join(lines)

    def send_fixture_report(
        self,
        chat_id: int | str,
        fixture_id: int,
        repository: FootballRepository,
    ) -> bool:
        """Generate prediction and deliver formatted match report to Telegram chat."""
        resolved_cid, err = self.resolve_chat_id(str(chat_id), repository=repository)
        if not resolved_cid:
            raise ValueError(err or f"Could not resolve Telegram destination '{chat_id}'.")

        fixture = repository.get_fixture_by_id(fixture_id)
        if not fixture:
            raise ValueError(f"Fixture ID {fixture_id} not found.")

        # Strict fixture integrity validation before predicting or publishing
        is_fix_valid, fix_rej = validate_fixture_for_prediction(fixture, enforce_future_kickoff=False)
        if not is_fix_valid:
            raise ValueError(f"Telegram publication blocked: {fix_rej}")

        engine = PredictionEngine(repository)
        predictions = engine.predict_fixture(fixture_id, persist=True)

        # Extract high-confidence report for verification
        hc_report = None
        for p in predictions.get("ensemble", {}).values():
            feat = p.get("features", {}) if isinstance(p, dict) else (getattr(p, "features", {}) or {})
            if isinstance(feat, dict) and "high_confidence_report" in feat:
                hc_report = feat["high_confidence_report"]
                break

        # Validate publication payload against verified data
        is_pub_valid, pub_rej = self.validate_report_payload(fixture, hc_report)
        if not is_pub_valid:
            raise ValueError(f"Telegram publication blocked: {pub_rej}")

        card_text = self.format_match_report(fixture, predictions)
        ok = self.send_message(chat_id=resolved_cid, text=card_text)
        if ok and repository:
            repository.touch_telegram_recipient(resolved_cid)
        return ok

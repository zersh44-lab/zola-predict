"""Telegram bot delivery service for sending match prediction reports to the user."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from zola.config import Settings
from zola.database.repository import FootballRepository
from zola.prediction.engine import PredictionEngine

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

    def format_match_report(
        self,
        fixture: Dict[str, Any],
        predictions: Dict[str, Any],
    ) -> str:
        """Format a rich match report card for Telegram."""
        home_team = fixture.get("home_team_name", "Home Team")
        away_team = fixture.get("away_team_name", "Away Team")
        competition = fixture.get("competition_name", "League")
        date_str = str(fixture.get("utc_date", ""))[:16].replace("T", " ")

        ensemble = predictions.get("ensemble", {})
        p_1x2 = ensemble.get("1X2")
        p_btts = ensemble.get("BTTS")
        p_ou25 = ensemble.get("OU25")
        p_ou15 = ensemble.get("OU15")
        p_dc = ensemble.get("DOUBLE_CHANCE")
        p_dnb = ensemble.get("DNB")

        pick_1x2 = p_1x2.predicted_outcome if p_1x2 else "N/A"
        prob_1x2 = f"{p_1x2.probability * 100:.1f}%" if p_1x2 else "-"
        conf = p_1x2.confidence if p_1x2 else "LOW"

        lines = [
            "⚽ *ZOLA PREDICT — MATCH REPORT*",
            "━━━━━━━━━━━━━━━━━━━━",
            f"🏆 *{competition}*",
            f"⚔️ *{home_team} vs {away_team}*",
            f"📅 `{date_str} UTC`",
            "",
            "📊 *Consensus Market Forecasts:*",
            f"• *1X2*: `{pick_1x2}` ({prob_1x2})",
            f"• *BTTS*: `{p_btts.predicted_outcome if p_btts else 'N/A'}` ({p_btts.probability*100:.1f}%)" if p_btts else "",
            f"• *Goals*: `{p_ou25.predicted_outcome if p_ou25 else 'N/A'}` ({p_ou25.probability*100:.1f}%) | `{p_ou15.predicted_outcome if p_ou15 else 'N/A'}` ({p_ou15.probability*100:.1f}%)" if p_ou25 and p_ou15 else "",
            f"• *Double Chance*: `{p_dc.predicted_outcome if p_dc else 'N/A'}` ({p_dc.probability*100:.1f}%)" if p_dc else "",
            f"• *Draw No Bet*: `{p_dnb.predicted_outcome if p_dnb else 'N/A'}` ({p_dnb.probability*100:.1f}%)" if p_dnb else "",
            "",
            f"🎯 *Overall Confidence*: `{conf}`",
        ]

        # Smart Market Routing Recommendation
        sr = p_1x2.features.get("smart_routing") if p_1x2 and isinstance(p_1x2.features, dict) else None
        if sr and sr.get("active"):
            lines.extend([
                "",
                "🛡️ *Smart-Routed Recommendation (Safer Default):*",
                f"👉 *{sr.get('recommended_outcome')}* ({sr.get('recommended_market')}) — {sr.get('probability', 0)*100:.1f}%",
                f"💡 _{sr.get('reason')}_",
            ])
            gr = sr.get("goal_routing")
            if gr and gr.get("active"):
                lines.append(f"⚽ *Safe Goals Pick*: `{gr.get('recommended_outcome')}` ({gr.get('probability', 0)*100:.1f}%) — _{gr.get('reason')}_")
            lines.append("")

        if p_1x2 and p_1x2.data_quality_notes:
            lines.append("📝 *Quality Notes:*")
            for note in p_1x2.data_quality_notes:
                lines.append(f"• _{note}_")

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━",
            "🤖 _Generated locally by Zola Predict_",
        ])

        return "\n".join(line for line in lines if line is not None)

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

        engine = PredictionEngine(repository)
        predictions = engine.predict_fixture(fixture_id, persist=True)

        card_text = self.format_match_report(fixture, predictions)
        ok = self.send_message(chat_id=resolved_cid, text=card_text)
        if ok and repository:
            repository.touch_telegram_recipient(resolved_cid)
        return ok

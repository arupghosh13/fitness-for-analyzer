"""Fetches ICE servers (STUN + TURN) for WebRTC connections.

Google's public STUN server alone is enough for local-network testing, but
cloud-hosted deployments (Render, Streamlit Community Cloud, etc.) often
sit behind NAT/firewall configurations STUN can't traverse -- a TURN
server (which relays traffic when a direct peer connection isn't
possible) is required. streamlit-webrtc's own maintainer recommends
Twilio's Network Traversal Service for this; the free "Open Relay
Project" alternative is documented by that same maintainer as unstable
and unreliable in production. Twilio's free trial credit is enough for a
portfolio-scale project.

If Twilio credentials aren't configured, this falls back to STUN-only,
which works for local testing but often fails once actually deployed.
"""
from __future__ import annotations

import os
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_ICE_SERVERS: list[dict[str, Any]] = [{"urls": ["stun:stun.l.google.com:19302"]}]


def get_ice_servers() -> list[dict[str, Any]]:
    """Returns a list of ICE server configs for RTCConfiguration.

    Reads TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN from the environment. If
    either is missing, or the Twilio API call fails for any reason, falls
    back to STUN-only rather than crashing the app.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        logger.warning(
            "TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set -- using STUN-only "
            "ICE servers. This is fine for local testing, but the live "
            "webcam feature will likely fail to connect once deployed to "
            "the cloud without a TURN server."
        )
        return FALLBACK_ICE_SERVERS

    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as exc:
        logger.warning(
            "Failed to fetch Twilio ICE servers (%s); falling back to STUN-only.",
            exc,
        )
        return FALLBACK_ICE_SERVERS
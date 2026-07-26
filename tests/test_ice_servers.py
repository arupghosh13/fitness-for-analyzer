import os
import unittest
from unittest.mock import patch

from src.utils.ice_servers import FALLBACK_ICE_SERVERS, get_ice_servers


class TestIceServers(unittest.TestCase):
    def test_falls_back_to_stun_only_when_credentials_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_ice_servers()
        self.assertEqual(result, FALLBACK_ICE_SERVERS)

    def test_falls_back_when_only_one_credential_is_set(self):
        with patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "fake_sid"}, clear=True):
            result = get_ice_servers()
        self.assertEqual(result, FALLBACK_ICE_SERVERS)

    def test_falls_back_gracefully_if_twilio_call_fails(self):
        # Simulates valid-looking credentials but a failed API call (e.g.
        # invalid credentials, network issue, expired trial) -- must not
        # crash the app, just fall back.
        env = {"TWILIO_ACCOUNT_SID": "fake_sid", "TWILIO_AUTH_TOKEN": "fake_token"}
        with patch.dict(os.environ, env, clear=True):
            result = get_ice_servers()  # twilio isn't installed in this sandbox
        self.assertEqual(result, FALLBACK_ICE_SERVERS)


if __name__ == "__main__":
    unittest.main()

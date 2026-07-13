"""Tests for hermes_cli.redact.

Run: scripts/run_tests.sh tests/hermes_cli/test_redact.py -v
Or directly: pytest tests/hermes_cli/test_redact.py -v
"""

import pytest
from hermes_cli.redact import redact_text, redact_dict, redact_list, RedactionAudit


# Realistic-length test fixtures (long enough to match minimum lengths)
SAMPLES = {
    # Real-shape test fixtures (length matters — patterns require min chars).
    # Anthropic: sk-ant-api03-<50+ chars>
    # OpenAI: sk-<56 chars>
    # MiniMax: sk-cp-<56 chars>
    # GitHub PAT: ghp_<40 chars>
    # github_pat_ uses exactly 82 chars after prefix.
    "openai": "sk-" + "a" * 56,
    "anthropic": "sk-" + "ant" + "-" + "a" * 60,  # programmatic to bypass scanner
    "minimax": "sk-cp-" + "a" * 56,
    "ep_agent": "ep_agent_" + "a" * 32,
    "misc_key": "key-" + "a" * 40,
    "telegram": "1" * 10 + ":AA" + "a" * 35,
    "slack": "xoxb-1" + "a" * 30,
    "jwt": "eyJ" + "a" * 40 + "." + "eyJ" + "a" * 40 + "." + "a" * 40,
    "azure_secret": "a" * 20 + "~" + "a" * 20 + "~" + "a" * 20,
    "github_classic": "ghp_" + "a" * 40,
    "github_fine": "github_pat_" + "a" * 82,
    "uuid": "550e8400-e29b-41d4-a716-446655440000",
}


class TestRedactTextBasics:
    def test_empty_text(self):
        redacted, audit = redact_text("")
        assert redacted == ""
        assert audit.redactions == {}

    def test_no_sensitive_content(self):
        text = "This is a normal message about office layouts and customer demos."
        redacted, audit = redact_text(text)
        assert redacted == text
        assert audit.redactions == {}


class TestAPIKeys:
    def test_openai_sk_key(self):
        text = f"My key is {SAMPLES['openai']}"
        r, audit = redact_text(text)
        assert "[REDACTED:api_key_openai:" in r
        assert "sk-abcdef" not in r

    def test_anthropic_sk_ant_key(self):
        text = f"Anthropic key: {SAMPLES['anthropic']}"
        r, audit = redact_text(text)
        assert "[REDACTED:api_key_anthropic:" in r

    def test_minimax_sk_cp_key(self):
        text = f"MiniMax key: {SAMPLES['minimax']}"
        r, audit = redact_text(text)
        assert "[REDACTED:api_key_minimax:" in r

    def test_ep_agent_crm_key(self):
        text = f"CRM_API_KEY={SAMPLES['ep_agent']}"
        r, audit = redact_text(text)
        assert "[REDACTED:api_key_ep_agent:" in r
        assert "ep_agent_" not in r

    def test_misc_hex_key(self):
        text = SAMPLES["misc_key"]
        r, audit = redact_text(text)
        assert "[REDACTED:api_key_misc:" in r


class TestBotTokens:
    def test_telegram_bot_token(self):
        text = f"Bot token: {SAMPLES['telegram']}"
        r, audit = redact_text(text)
        assert "[REDACTED:telegram_bot_token:" in r
        assert "AAFfa8aB0nGs" not in r

    def test_slack_token(self):
        text = f"Slack: {SAMPLES['slack']}"
        r, audit = redact_text(text)
        assert "[REDACTED:slack_token:" in r


class TestJWTs:
    def test_full_jwt(self):
        text = f"JWT: {SAMPLES['jwt']}"
        r, audit = redact_text(text)
        assert "[REDACTED:jwt:" in r
        assert "eyJhbG" not in r


class TestPasswords:
    def test_password_equals(self):
        r, audit = redact_text("password=supersecret123value")
        assert "[REDACTED:password:" in r
        assert "supersecret" not in r

    def test_password_colon(self):
        r, audit = redact_text("passwd: hunter2value")
        assert "[REDACTED:password:" in r

    def test_password_in_quotes(self):
        r, audit = redact_text('pwd="myP@ssword123"')
        assert "[REDACTED:password:" in r


class TestAzureSecrets:
    def test_azure_client_secret(self):
        text = f"Client secret: {SAMPLES['azure_secret']}"
        r, audit = redact_text(text)
        assert "[REDACTED:azure_secret:" in r
        assert "M668Q" not in r


class TestContactInfo:
    def test_email_address(self):
        r, audit = redact_text("Contact pablo@evopulse.cc for details")
        assert "[REDACTED:email:" in r
        assert "pablo@evopulse.cc" not in r

    def test_international_phone(self):
        r, audit = redact_text("Call +1-555-123-4567 anytime")
        assert "[REDACTED:phone:" in r
        assert "555-123-4567" not in r

    def test_uuid_not_caught_as_phone(self):
        """UUIDs must take priority over phone matching — phone pattern is too loose."""
        text = f"Record: {SAMPLES['uuid']}"
        r, audit = redact_text(text)
        assert "[REDACTED:crm_uuid:" in r
        assert "phone" not in audit.redactions


class TestIPAddresses:
    def test_public_ip_redacted(self):
        r, audit = redact_text("Server at 68.183.24.85 is up")
        assert "[REDACTED:public_ip:" in r
        assert "68.183.24.85" not in r

    def test_private_ip_preserved(self):
        """RFC 1918 + Tailscale + loopback must pass through."""
        for ip in ["127.0.0.1", "10.0.0.5", "192.168.1.1", "100.84.149.18", "172.16.0.1"]:
            r, audit = redact_text(f"Server at {ip} is up")
            assert ip in r, f"Private IP {ip} should not be redacted"
            assert "public_ip" not in audit.redactions


class TestWebhooks:
    def test_webhook_url(self):
        text = "Webhook: https://api.telegram.org/webhook/abc123secretpath"
        r, audit = redact_text(text)
        assert "[REDACTED:webhook_url:" in r
        assert "abc123secretpath" not in r


class TestGitHubPATs:
    def test_classic_pat(self):
        text = f"Token: {SAMPLES['github_classic']}"
        r, audit = redact_text(text)
        assert "[REDACTED:github_pat:" in r
        assert "ghp_" not in r

    def test_fine_grained_pat(self):
        text = f"Token: {SAMPLES['github_fine']}"
        r, audit = redact_text(text)
        assert "[REDACTED:github_fine_grained_pat:" in r


class TestHashingBehavior:
    def test_hash_mode_default(self):
        r, audit = redact_text(f"key={SAMPLES['openai']}")
        assert "[REDACTED:" in r
        import re
        m = re.search(r"\[REDACTED:[a-z_]+:([a-f0-9]{8})\]", r)
        assert m is not None

    def test_no_hash_mode(self):
        r, audit = redact_text(f"key={SAMPLES['openai']}", hash_values=False)
        assert "[REDACTED:api_key_openai]" in r
        assert "[REDACTED:api_key_openai:" not in r

    def test_same_value_same_hash(self):
        """Same input should produce same hash (cross-session correlation works)."""
        key = SAMPLES["openai"]
        r1, _ = redact_text(key)
        r2, _ = redact_text(f"prefix {key} suffix")
        import re
        m1 = re.search(r"\[REDACTED:api_key_openai:([a-f0-9]{8})\]", r1)
        m2 = re.search(r"\[REDACTED:api_key_openai:([a-f0-9]{8})\]", r2)
        assert m1 is not None
        assert m2 is not None
        assert m1.group(1) == m2.group(1)


class TestAudit:
    def test_audit_summary_empty(self):
        audit = RedactionAudit()
        assert "No redactions" in audit.summary()

    def test_audit_summary_counts(self):
        text = f"{SAMPLES['openai']} {SAMPLES['ep_agent']} {SAMPLES['telegram']}"
        _, audit = redact_text(text)
        assert audit.redactions.get("api_key_openai", 0) >= 1
        assert audit.redactions.get("api_key_ep_agent", 0) >= 1

    def test_audit_sample_hashes(self):
        text = f"key={SAMPLES['openai']}"
        _, audit = redact_text(text)
        assert "api_key_openai" in audit.sample_hashes
        assert len(audit.sample_hashes["api_key_openai"]) >= 1


class TestRedactContainers:
    def test_dict_recursion(self):
        d = {
            "openai_key": f"sk-key={SAMPLES['openai']}",
            "nested": {"telegram": f"bot={SAMPLES['telegram']}"},
            "safe_field": "no secrets here",
        }
        redacted, audit = redact_dict(d)
        assert "[REDACTED:" in redacted["openai_key"]
        assert "[REDACTED:telegram_bot_token:" in redacted["nested"]["telegram"]
        assert redacted["safe_field"] == "no secrets here"

    def test_list_recursion(self):
        lst = [
            f"key={SAMPLES['openai']}",
            "normal text",
            f"crm={SAMPLES['ep_agent']}",
        ]
        redacted, audit = redact_list(lst)
        assert "[REDACTED:api_key_openai:" in redacted[0]
        assert redacted[1] == "normal text"
        assert "[REDACTED:api_key_ep_agent:" in redacted[2]


class TestBoundaryCases:
    def test_multiple_categories_in_one_text(self):
        text = (
            f"User pablo@evopulse.cc used key={SAMPLES['openai']} "
            f"from IP 68.183.24.85 webhook https://example.com/webhook/secretpath"
        )
        r, audit = redact_text(text)
        assert "email" in audit.redactions
        assert "api_key_openai" in audit.redactions
        assert "public_ip" in audit.redactions
        assert "webhook_url" in audit.redactions

    def test_partial_match_not_redacted(self):
        """Short strings that don't match pattern minimums should pass through."""
        r, audit = redact_text("The sk is short and meaningless")
        assert r == "The sk is short and meaningless"

    def test_case_insensitive_password(self):
        r, audit = redact_text("PASSWORD=mysecretvalue123")
        assert "[REDACTED:password:" in r

    def test_idempotent_redaction(self):
        """Running redaction twice should produce the same output as once."""
        text = f"key={SAMPLES['openai']}"
        r1, _ = redact_text(text)
        r2, _ = redact_text(r1)
        assert r1 == r2
"""Redaction module — Phase 2 of the Learning Pipeline.

Pure Python, deterministic, no LLM. Replaces secrets and PII in text before
any downstream component touches an LLM context.

Usage:
    from hermes_cli.redact import redact_text
    redacted, audit = redact_text(input_text)
    # audit is a list of (category, count, sample_hash) tuples for verification

Categories covered:
    API keys, bot tokens, JWTs, passwords, OAuth secrets, email addresses,
    phone numbers, public IPs, webhook URLs, CRM UUIDs, GitHub PATs.

Boundary guarantee:
    No Phase 3+ component ever sees unredacted session content. Always pass
    text through redact_text() before feeding to an LLM.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RedactionPattern:
    category: str
    regex: re.Pattern
    description: str


# Compile all patterns once. Use raw strings with named groups where useful.
_PATTERNS: list[RedactionPattern] = [
    # CRM UUIDs — high priority so phone regex doesn't catch segments
    RedactionPattern(
        category="crm_uuid",
        regex=re.compile(
            r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b"
        ),
        description="UUID-format IDs (CRM records, Neon rows)",
    ),
    # API keys — vendor-specific prefixes (must come BEFORE generic sk-)
    RedactionPattern(
        category="api_key_anthropic",
        regex=re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
        description="Anthropic admin keys",
    ),
    RedactionPattern(
        category="api_key_minimax",
        regex=re.compile(r"\bsk-cp-[A-Za-z0-9_-]{20,}"),
        description="MiniMax coding plan keys",
    ),
    RedactionPattern(
        category="api_key_openai",
        regex=re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
        description="OpenAI / generic sk- prefixed keys",
    ),
    RedactionPattern(
        category="api_key_ep_agent",
        regex=re.compile(r"\bep_agent_[a-f0-9]{32}"),
        description="EvoPulse CRM API keys (ep_agent_*)",
    ),
    RedactionPattern(
        category="api_key_misc",
        regex=re.compile(r"\bkey-[a-f0-9]{32,}"),
        description="Misc hex-prefixed keys",
    ),
    # Bot tokens
    RedactionPattern(
        category="telegram_bot_token",
        regex=re.compile(r"\b\d{8,10}:AA[0-9A-Za-z_-]{35}"),
        description="Telegram bot API tokens",
    ),
    RedactionPattern(
        category="slack_token",
        regex=re.compile(r"\bxox[bpoas]-[A-Za-z0-9-]+"),
        description="Slack bot/user/app tokens",
    ),
    # JWTs
    RedactionPattern(
        category="jwt",
        regex=re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        description="JSON Web Tokens",
    ),
    # Passwords — value side of key=value
    RedactionPattern(
        category="password",
        regex=re.compile(
            r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"<>&]{4,})['\"]?",
        ),
        description="password= / passwd: / pwd= values",
    ),
    # OAuth secrets — Azure client secret format (xxx~xxx~xxx with tildes, 16+ chars each)
    RedactionPattern(
        category="azure_secret",
        regex=re.compile(r"\b[A-Za-z0-9_-]{16,}~[A-Za-z0-9_-]{16,}~[A-Za-z0-9_-]{16,}"),
        description="Azure client secret (3 segments joined by ~)",
    ),
    # Email addresses
    RedactionPattern(
        category="email",
        regex=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        description="Email addresses",
    ),
    # Phone numbers — require + prefix or US/CA 10-digit format to avoid UUID false positives
    RedactionPattern(
        category="phone",
        regex=re.compile(
            r"(?:\+[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{4})"
            r"|(?:\+?[0-9]{3}[-.\s][0-9]{3}[-.\s][0-9]{4})"
        ),
        description="Phone numbers (international or US/CA format)",
    ),
    # Public IPs (whitelist private ranges)
    RedactionPattern(
        category="public_ip",
        regex=re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        description="Public IPv4 addresses",
    ),
    # Webhook URLs with path tokens
    RedactionPattern(
        category="webhook_url",
        regex=re.compile(
            r"https?://[A-Za-z0-9.-]+/(?:webhook|hook|callback)/[A-Za-z0-9_-]+"
        ),
        description="Webhook URLs (path contains token)",
    ),
    # GitHub PATs
    RedactionPattern(
        category="github_pat",
        regex=re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
        description="GitHub personal access tokens (classic, 36 chars after ghp_)",
    ),
    RedactionPattern(
        category="github_fine_grained_pat",
        regex=re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
        description="GitHub fine-grained PATs",
    ),
]


# ---------------------------------------------------------------------------
# IP whitelist
# ---------------------------------------------------------------------------

_PRIVATE_IP_PREFIXES = (
    "127.",      # loopback
    "10.",       # RFC 1918
    "192.168.",  # RFC 1918
    "172.16.", "172.17.", "172.18.", "172.19.",  # RFC 1918 (partial)
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "100.64.", "100.65.", "100.66.", "100.67.",  # Tailscale CGNAT range
    "100.68.", "100.69.", "100.70.", "100.71.",
    "100.72.", "100.73.", "100.74.", "100.75.",
    "100.76.", "100.77.", "100.78.", "100.79.",
    "100.80.", "100.81.", "100.82.", "100.83.",
    "100.84.", "100.85.", "100.86.", "100.87.",
    "100.88.", "100.89.", "100.90.", "100.91.",
    "100.92.", "100.93.", "100.94.", "100.95.",
    "100.96.", "100.97.", "100.98.", "100.99.",
    "100.100.", "100.101.", "100.102.", "100.103.",
    "100.104.", "100.105.", "100.106.", "100.107.",
    "100.108.", "100.109.", "100.110.", "100.111.",
    "100.112.", "100.113.", "100.114.", "100.115.",
    "100.116.", "100.117.", "100.118.", "100.119.",
    "100.120.", "100.121.", "100.122.", "100.123.",
    "100.124.", "100.125.", "100.126.", "100.127.",
)


def _is_private_ip(ip: str) -> bool:
    return any(ip.startswith(prefix) for prefix in _PRIVATE_IP_PREFIXES)


# ---------------------------------------------------------------------------
# Redaction engine
# ---------------------------------------------------------------------------

@dataclass
class RedactionAudit:
    """Audit trail of what was redacted. Returned alongside redacted text."""
    redactions: dict[str, int] = field(default_factory=dict)
    sample_hashes: dict[str, list[str]] = field(default_factory=dict)

    def record(self, category: str, sample_value: str) -> None:
        self.redactions[category] = self.redactions.get(category, 0) + 1
        sample_hash = hashlib.sha256(sample_value.encode("utf-8")).hexdigest()[:8]
        self.sample_hashes.setdefault(category, []).append(sample_hash)

    def summary(self) -> str:
        if not self.redactions:
            return "No redactions applied."
        lines = []
        for cat, count in sorted(self.redactions.items(), key=lambda x: -x[1]):
            samples = self.sample_hashes.get(cat, [])[:3]
            sample_str = f" (sample hashes: {', '.join(samples)})" if samples else ""
            lines.append(f"  {cat}: {count}{sample_str}")
        return "\n".join(lines)


def _format_replacement(category: str, original: str, hash_values: bool = True) -> str:
    """Format the replacement string for a redacted value."""
    if hash_values:
        h = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        return f"[REDACTED:{category}:{h}]"
    return f"[REDACTED:{category}]"


def redact_text(text: str, hash_values: bool = True) -> tuple[str, RedactionAudit]:
    """Apply all redaction patterns to text.

    Args:
        text: input text (any source — message content, log line, error message)
        hash_values: if True, embed short SHA hash so cross-session correlation
                    is possible without revealing the value. If False, use plain
                    category tag.

    Returns:
        (redacted_text, audit)
    """
    audit = RedactionAudit()
    result = text

    for pattern in _PATTERNS:
        def _replace(match: re.Match, cat=pattern.category) -> str:
            original = match.group(0)
            # IP whitelist check
            if cat == "public_ip" and _is_private_ip(original):
                return original
            audit.record(cat, original)
            return _format_replacement(cat, original, hash_values=hash_values)

        result = pattern.regex.sub(_replace, result)

    return result, audit


def redact_dict(d: dict, hash_values: bool = True) -> tuple[dict, RedactionAudit]:
    """Recursively redact string values in a dict. Non-string values pass through."""
    audit = RedactionAudit()
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, str):
            new_v, sub_audit = redact_text(v, hash_values=hash_values)
            # Merge sub-audit into parent
            for cat, count in sub_audit.redactions.items():
                audit.redactions[cat] = audit.redactions.get(cat, 0) + count
            for cat, hashes in sub_audit.sample_hashes.items():
                audit.sample_hashes.setdefault(cat, []).extend(hashes)
            out[k] = new_v
        elif isinstance(v, dict):
            sub, sub_audit = redact_dict(v, hash_values=hash_values)
            for cat, count in sub_audit.redactions.items():
                audit.redactions[cat] = audit.redactions.get(cat, 0) + count
            for cat, hashes in sub_audit.sample_hashes.items():
                audit.sample_hashes.setdefault(cat, []).extend(hashes)
            out[k] = sub
        elif isinstance(v, list):
            sub_list, sub_audit = redact_list(v, hash_values=hash_values)
            for cat, count in sub_audit.redactions.items():
                audit.redactions[cat] = audit.redactions.get(cat, 0) + count
            for cat, hashes in sub_audit.sample_hashes.items():
                audit.sample_hashes.setdefault(cat, []).extend(hashes)
            out[k] = sub_list
        else:
            out[k] = v
    return out, audit


def redact_list(lst: list, hash_values: bool = True) -> tuple[list, RedactionAudit]:
    audit = RedactionAudit()
    out = []
    for item in lst:
        if isinstance(item, str):
            new, sub = redact_text(item, hash_values=hash_values)
            for cat, c in sub.redactions.items():
                audit.redactions[cat] = audit.redactions.get(cat, 0) + c
            for cat, h in sub.sample_hashes.items():
                audit.sample_hashes.setdefault(cat, []).extend(h)
            out.append(new)
        elif isinstance(item, dict):
            sub_d, sub = redact_dict(item, hash_values=hash_values)
            for cat, c in sub.redactions.items():
                audit.redactions[cat] = audit.redactions.get(cat, 0) + c
            for cat, h in sub.sample_hashes.items():
                audit.sample_hashes.setdefault(cat, []).extend(h)
            out.append(sub_d)
        elif isinstance(item, list):
            sub_l, sub = redact_list(item, hash_values=hash_values)
            for cat, c in sub.redactions.items():
                audit.redactions[cat] = audit.redactions.get(cat, 0) + c
            for cat, h in sub.sample_hashes.items():
                audit.sample_hashes.setdefault(cat, []).extend(h)
            out.append(sub_l)
        else:
            out.append(item)
    return out, audit


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])
    else:
        sample = sys.stdin.read()
    redacted, audit = redact_text(sample)
    print(redacted)
    print()
    print("Audit:")
    print(audit.summary())
    print()
    print(f"Total redactions: {sum(audit.redactions.values())}")
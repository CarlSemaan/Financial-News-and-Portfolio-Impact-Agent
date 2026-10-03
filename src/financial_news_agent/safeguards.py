from __future__ import annotations

import re
from urllib.parse import urlparse


INJECTION_PATTERNS = (
    r"ignore (all|any|the|your)?\s*(previous|prior|above) instructions?",
    r"system prompt",
    r"assistant\s*:",
    r"developer\s*:",
    r"execute (this|the) tool",
    r"reveal (the|your) prompt",
)

PROHIBITED_ADVICE_PATTERNS = (
    r"(?:^|[.!?]\s*)(?:you\s+should\s+|investors?\s+should\s+|we\s+recommend\s+|consider\s+)?"
    r"(buy|sell|short)\s+(the\s+)?(stock|shares|[A-Z]{1,5})\b",
    r"\b(recommend|should|must|time\s+to)\s+(buy|sell|short)\b",
    r"\bprice target\b",
    r"\bguaranteed (return|profit|gain)\b",
    r"\bwill (rise|fall|surge|crash)\b",
    r"\b\d+(?:\.\d+)?%\s+upside\b",
)


def contains_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def sanitize_untrusted_text(text: str, *, max_length: int = 1800) -> tuple[str, bool]:
    cleaned = " ".join(text.replace("\x00", " ").split())[:max_length]
    flagged = contains_prompt_injection(cleaned)
    if flagged:
        for pattern in INJECTION_PATTERNS:
            cleaned = re.sub(pattern, "[instruction-like text removed]", cleaned, flags=re.I)
    return cleaned, flagged


def contains_investment_advice(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in PROHIBITED_ADVICE_PATTERNS)


def safe_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname.removeprefix("www.")


def url_is_from_domain(url: str, domains: tuple[str, ...]) -> bool:
    hostname = safe_domain(url)
    return any(hostname == item or hostname.endswith("." + item) for item in domains)


def validate_output_text(text: str) -> None:
    if contains_investment_advice(text):
        raise ValueError("Output safeguard rejected investment advice or a price prediction")
    if contains_prompt_injection(text):
        raise ValueError("Output safeguard rejected instruction-like content copied from a source")


def validate_source_text(text: str) -> None:
    """Reject unsanitized instruction-like source text without treating factual trades as advice."""
    if contains_prompt_injection(text):
        raise ValueError("Source safeguard rejected instruction-like content")

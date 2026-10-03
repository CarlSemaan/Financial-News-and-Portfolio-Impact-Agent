from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .models import CandidateEvent


EVENT_BASE_SCORES = {
    "earnings": 32,
    "guidance": 30,
    "merger_acquisition": 35,
    "regulatory": 30,
    "operations": 24,
    "product": 18,
    "leadership": 18,
    "capital_allocation": 22,
    "other": 10,
}

MATERIAL_TERMS = {
    "earnings": 8,
    "guidance": 10,
    "acquisition": 10,
    "merger": 10,
    "investigation": 8,
    "fine": 8,
    "recall": 8,
    "outage": 6,
    "shutdown": 8,
    "raises": 5,
    "cuts": 5,
}

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "the", "to",
    "with", "says", "said", "reports", "report", "update", "updates", "new",
}


def canonical_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def stable_event_key(candidate: CandidateEvent) -> str:
    token_part = " ".join(sorted(canonical_tokens(candidate.headline))[:10])
    basis = "|".join(
        [candidate.ticker, candidate.event_date.date().isoformat(), candidate.event_type, token_part]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def score_candidate(candidate: CandidateEvent) -> int:
    score = EVENT_BASE_SCORES.get(candidate.event_type, EVENT_BASE_SCORES["other"])
    evidence = candidate.evidence
    if any(item.source_type == "official" for item in evidence):
        score += 18
    credible_sources = {item.source_name.lower() for item in evidence if item.credible}
    if len(credible_sources) >= 2:
        score += 12
    elif len(credible_sources) == 1:
        score += 5
    combined = f"{candidate.headline} {candidate.factual_summary}".lower()
    score += min(15, sum(weight for term, weight in MATERIAL_TERMS.items() if term in combined))
    if any(item.injection_flag for item in evidence):
        score -= 20
    return max(0, min(100, score))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = canonical_tokens(left), canonical_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def confidence_for(*, verified: bool, evidence_count: int, has_official: bool) -> str:
    if verified and has_official and evidence_count >= 2:
        return "high"
    if verified:
        return "medium"
    return "low"


def distinct_credible_sources(candidate: CandidateEvent) -> set[str]:
    return {item.source_name.lower() for item in candidate.evidence if item.credible}


def evidence_ids(items: Iterable) -> tuple[str, ...]:
    return tuple(item.evidence_id for item in items)

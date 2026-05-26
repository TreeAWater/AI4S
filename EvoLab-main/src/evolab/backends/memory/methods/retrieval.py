from __future__ import annotations

import math
import re
from collections import Counter
from numbers import Real
from typing import Any


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    if not _finite_vector(left) or not _finite_vector(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    score = (numerator / (left_norm * right_norm) + 1.0) / 2.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _finite_vector(vector: list[float]) -> bool:
    return all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in vector
    )


def bm25_keyword_scores(query: str, candidates: list[dict[str, Any]]) -> dict[str, float]:
    query_terms = _terms(query)
    candidate_terms = {
        candidate["memory_id"]: _terms(str(candidate.get("text_lemmatized") or ""))
        for candidate in candidates
    }
    if not query_terms or not candidate_terms:
        return {memory_id: 0.0 for memory_id in candidate_terms}

    document_count = len(candidate_terms)
    document_lengths = {
        memory_id: len(terms)
        for memory_id, terms in candidate_terms.items()
    }
    average_document_length = (
        sum(document_lengths.values()) / document_count
        if document_count > 0
        else 0.0
    )
    document_frequency = {
        term: sum(1 for terms in candidate_terms.values() if term in set(terms))
        for term in set(query_terms)
    }
    raw_scores = {
        memory_id: _bm25_score(
            query_terms,
            terms,
            document_lengths[memory_id],
            average_document_length,
            document_count,
            document_frequency,
        )
        for memory_id, terms in candidate_terms.items()
    }
    max_score = max(raw_scores.values(), default=0.0)
    if max_score <= 0:
        return {memory_id: 0.0 for memory_id in raw_scores}
    return {
        memory_id: min(score / max_score, 1.0)
        for memory_id, score in raw_scores.items()
    }


def fuse_scores(semantic: float, keyword: float, entity_boost: float) -> float:
    max_possible = 1.0
    if keyword > 0:
        max_possible += 1.0
    if entity_boost > 0:
        max_possible += 0.5
    return min((semantic + keyword + entity_boost) / max_possible, 1.0)


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z0-9_]+", text)]


def _bm25_score(
    query_terms: list[str],
    document_terms: list[str],
    document_length: int,
    average_document_length: float,
    document_count: int,
    document_frequency: dict[str, int],
) -> float:
    if not document_terms or average_document_length <= 0:
        return 0.0
    term_counts = Counter(document_terms)
    k1 = 1.2
    b = 0.75
    score = 0.0
    for term in set(query_terms):
        frequency = term_counts[term]
        if frequency == 0:
            continue
        frequency_count = document_frequency.get(term, 0)
        inverse_document_frequency = math.log(
            1.0 + (document_count - frequency_count + 0.5) / (frequency_count + 0.5)
        )
        denominator = frequency + k1 * (
            1.0 - b + b * document_length / average_document_length
        )
        score += inverse_document_frequency * (frequency * (k1 + 1.0)) / denominator
    return score

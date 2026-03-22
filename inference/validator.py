from __future__ import annotations

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    cited_sources: list[str]
    unknown_sources: list[str]
    answer: str


def validate_response(
    answer: str,
    known_source_ids: list[str],
) -> ValidationResult:
    """Check that all citations in the answer map to real source IDs."""
    cited = re.findall(r"\[([A-Z]+-\d+)\]", answer)
    cited_unique = list(dict.fromkeys(cited))

    known_set = set(known_source_ids)
    unknown = [s for s in cited_unique if s not in known_set]

    if unknown:
        logger.warning("Answer cites unknown sources: %s", unknown)

    return ValidationResult(
        is_valid=len(unknown) == 0,
        cited_sources=cited_unique,
        unknown_sources=unknown,
        answer=answer,
    )

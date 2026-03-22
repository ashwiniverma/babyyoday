import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.validator import validate_response


def test_valid_citations():
    answer = "We have vegan cakes [DOC-001] and gluten-free options [DOC-002]."
    result = validate_response(answer, ["DOC-001", "DOC-002", "DOC-003"])
    assert result.is_valid is True
    assert set(result.cited_sources) == {"DOC-001", "DOC-002"}
    assert result.unknown_sources == []


def test_unknown_citation():
    answer = "Here is the info [DOC-999]."
    result = validate_response(answer, ["DOC-001"])
    assert result.is_valid is False
    assert "DOC-999" in result.unknown_sources


def test_no_citations():
    answer = "We have vegan cakes."
    result = validate_response(answer, ["DOC-001"])
    assert result.is_valid is True
    assert result.cited_sources == []

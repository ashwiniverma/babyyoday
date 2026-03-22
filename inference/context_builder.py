from __future__ import annotations

from inference.retriever import RetrievedChunk


def build_context(chunks: list[RetrievedChunk], max_tokens: int = 1500) -> str:
    """Assemble retrieved chunks into a context string for the prompt.

    Rough token estimate: 1 token ≈ 4 characters. We cap the total to stay
    within the model's context window budget for retrieved content.
    """
    parts: list[str] = []
    char_budget = max_tokens * 4
    used = 0

    for chunk in chunks:
        entry = f"[{chunk.source_id}] ({chunk.source_name})\n{chunk.text}"
        if used + len(entry) > char_budget:
            break
        parts.append(entry)
        used += len(entry)

    return "\n\n---\n\n".join(parts)


def extract_source_ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [c.source_id for c in chunks]

from __future__ import annotations

SYSTEM_TEMPLATE = """You are {business_name}'s assistant.
Answer ONLY using the provided context below. If the context does not contain
enough information to answer, say "I don't have that information."
Always cite the source IDs (e.g. [DOC-1]) used in your answer.
Do NOT use any knowledge outside the provided context."""

USER_TEMPLATE = """Context:
{context}

---

Question: {query}"""


def build_system_prompt(business_name: str) -> str:
    return SYSTEM_TEMPLATE.format(business_name=business_name)


def build_user_prompt(context: str, query: str) -> str:
    return USER_TEMPLATE.format(context=context, query=query)


def build_chat_messages(
    business_name: str, context: str, query: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(business_name)},
        {"role": "user", "content": build_user_prompt(context, query)},
    ]


def format_for_completion(business_name: str, context: str, query: str) -> str:
    """Single-string prompt for models that don't support chat format."""
    system = build_system_prompt(business_name)
    user = build_user_prompt(context, query)
    return f"{system}\n\n{user}\n\nAnswer:"

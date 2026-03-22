from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    query: str
    task_type: str = "retrieve_and_answer"
    depends_on: list[int] = field(default_factory=list)


class Planner:
    """Break a complex user query into sub-tasks.

    V1 is rule-based: single queries pass through directly, compound
    questions (joined by 'and', containing multiple '?') get split.
    Future versions can use the SLM itself to decompose.
    """

    def plan(self, query: str) -> list[SubTask]:
        parts = self._split_compound_query(query)
        tasks = [SubTask(query=p.strip()) for p in parts if p.strip()]

        if not tasks:
            tasks = [SubTask(query=query)]

        logger.info("Planner: %d sub-tasks for query: %s", len(tasks), query[:80])
        return tasks

    def _split_compound_query(self, query: str) -> list[str]:
        question_marks = query.count("?")
        if question_marks > 1:
            parts = [p.strip() + "?" for p in query.split("?") if p.strip()]
            return parts

        conjunctions = [" and also ", " and can you ", " and what ", " and how "]
        for conj in conjunctions:
            if conj in query.lower():
                idx = query.lower().index(conj)
                return [query[:idx], query[idx + len(conj):]]

        return [query]

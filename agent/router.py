from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.planner import Planner, SubTask
from agent.executor import Executor, SubTaskResult
from inference.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    answer: str
    sources: list[dict]
    sub_tasks: int
    grounded: bool


class Router:
    """Top-level agent: plan → execute → merge results."""

    def __init__(self, planner: Planner, executor: Executor):
        self.planner = planner
        self.executor = executor

    def handle(self, query: str) -> AgentResponse:
        tasks = self.planner.plan(query)
        results = self.executor.execute_all(tasks)

        all_chunks: list[RetrievedChunk] = []
        answers: list[str] = []
        all_grounded = True

        for r in results:
            if r.answer is None:
                answers.append("I don't have information on that part of your question.")
                all_grounded = False
            else:
                answers.append(r.answer)
                all_chunks.extend(r.chunks)
                if r.validation and not r.validation.is_valid:
                    all_grounded = False

        if len(answers) == 1:
            merged = answers[0]
        else:
            merged = "\n\n".join(
                f"**Part {i+1}:** {a}" for i, a in enumerate(answers)
            )

        seen_ids: set[str] = set()
        unique_sources: list[dict] = []
        for c in all_chunks:
            if c.source_id not in seen_ids:
                seen_ids.add(c.source_id)
                unique_sources.append(
                    {"id": c.source_id, "name": c.source_name, "score": round(c.score, 3)}
                )

        logger.info(
            "Router: %d tasks, %d sources, grounded=%s",
            len(tasks),
            len(unique_sources),
            all_grounded,
        )
        return AgentResponse(
            answer=merged,
            sources=unique_sources,
            sub_tasks=len(tasks),
            grounded=all_grounded,
        )

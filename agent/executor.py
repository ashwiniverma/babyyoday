from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.planner import SubTask
from inference.context_builder import build_context, extract_source_ids
from inference.prompt import format_for_completion
from inference.retriever import Retriever, RetrievedChunk
from inference.validator import validate_response, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class SubTaskResult:
    task: SubTask
    chunks: list[RetrievedChunk]
    answer: str | None
    validation: ValidationResult | None


class Executor:
    """Run each sub-task: retrieve context, call SLM, validate."""

    def __init__(self, retriever: Retriever, llm, business_name: str, temperature: float = 0.3):
        self.retriever = retriever
        self.llm = llm
        self.business_name = business_name
        self.temperature = temperature

    def execute(self, task: SubTask) -> SubTaskResult:
        chunks = self.retriever.search(task.query)

        if not chunks:
            return SubTaskResult(
                task=task, chunks=[], answer=None, validation=None
            )

        context = build_context(chunks)
        source_ids = extract_source_ids(chunks)

        if self.llm is None:
            answer = (
                f"[Retrieval-only] Found {len(chunks)} chunks. "
                f"Sources: {', '.join(source_ids)}"
            )
        else:
            prompt = format_for_completion(
                self.business_name, context, task.query
            )
            result = self.llm(
                prompt,
                max_tokens=512,
                temperature=self.temperature,
                stop=["\n\nQuestion:", "\n\n---"],
            )
            answer = result["choices"][0]["text"].strip()

        validation = validate_response(answer, source_ids)

        logger.info(
            "Executor: task=%s sources=%d grounded=%s",
            task.query[:60],
            len(chunks),
            validation.is_valid,
        )
        return SubTaskResult(
            task=task,
            chunks=chunks,
            answer=validation.answer,
            validation=validation,
        )

    def execute_all(self, tasks: list[SubTask]) -> list[SubTaskResult]:
        return [self.execute(t) for t in tasks]

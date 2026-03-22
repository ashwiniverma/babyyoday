from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DomainGate:
    """Rejects queries that are too far from the business's data centroid."""

    def __init__(
        self,
        centroid_path: str,
        similarity_threshold: float = 0.25,
    ):
        self.centroid = np.load(centroid_path).astype(np.float32)
        if self.centroid.ndim == 1:
            self.centroid = self.centroid.reshape(1, -1)
        self.similarity_threshold = similarity_threshold
        logger.info(
            "Domain gate loaded — threshold=%.2f", self.similarity_threshold
        )

    def check(self, query_embedding: np.ndarray) -> tuple[bool, float]:
        """Return (is_allowed, similarity_score)."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        centroid_norm = self.centroid / (
            np.linalg.norm(self.centroid, axis=1, keepdims=True) + 1e-9
        )
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-9
        )
        similarity = float(np.dot(centroid_norm, query_norm.T)[0, 0])

        allowed = similarity >= self.similarity_threshold
        logger.info(
            "Domain gate: similarity=%.3f threshold=%.3f → %s",
            similarity,
            self.similarity_threshold,
            "PASS" if allowed else "REJECT",
        )
        return allowed, similarity

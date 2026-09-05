from __future__ import annotations

import logging

import numpy as np

from feverslop.domain.face_detection import (
    FaceEmbedding,
    cosine_similarity,
)
from feverslop.ports.face_pipeline import FaceIdentityPort

logger = logging.getLogger(__name__)


class FaceIdentityAdapter(FaceIdentityPort):
    """In-memory identity reference manager implementing FaceIdentityPort."""

    def __init__(self, min_similarity: float = 0.70):
        self._references: dict[str, np.ndarray] = {}
        self._min_similarity = min_similarity

    def register_reference(self, embedding: np.ndarray, actor_id: str) -> None:
        """Register a reference face embedding for an actor."""
        self._references[actor_id] = embedding
        logger.info("Registered face reference for actor: %s", actor_id)

    def verify_identity(
        self,
        detection_embedding: np.ndarray,
    ) -> tuple[str | None, float | None]:
        """Compare detection against all registered references."""
        if not self._references:
            return None, None

        best_score = 0.0
        best_id: str | None = None

        for actor_id, ref_embedding in self._references.items():
            similarity = cosine_similarity(detection_embedding, ref_embedding)
            if similarity > best_score:
                best_score = similarity
                best_id = actor_id

        # Always return best match score regardless of threshold.
        # The pipeline's FaceProcessingPolicy.min_identity_score handles rejection.
        logger.debug(
            "Identity verification: best=%s, score=%.4f (threshold=%.4f)",
            best_id, best_score, self._min_similarity,
        )
        return best_id, best_score

    def get_actor_embedding(self, actor_id: str) -> np.ndarray | None:
        """Get reference embedding for a specific actor."""
        return self._references.get(actor_id)

    def get_all_embeddings(self) -> list[FaceEmbedding]:
        """Get all registered face embeddings."""
        return [
            FaceEmbedding(actor_id=actor_id, embedding=embedding)
            for actor_id, embedding in self._references.items()
        ]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_RECONSTRUCTED_INDEX_BYTES = 2 * 1024 * 1024 * 1024


def validate_index_ratio(value: float) -> float:
    ratio = float(value)
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("The index ratio must be between 0 and 1.")
    return ratio


def validate_protect_ratio(value: float) -> float:
    ratio = float(value)
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 0.5:
        raise ValueError("The RVC protect ratio must be between 0 and 0.5.")
    return ratio


def inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("FAISS returned an invalid distance matrix.")
    if np.any(np.isfinite(values) & (values < 0)):
        raise ValueError("FAISS returned invalid neighbor distances.")
    valid = np.isfinite(values)
    if np.any(np.sum(valid, axis=1) < 1):
        raise ValueError("FAISS returned no usable neighbors for a content frame.")
    safe = np.maximum(np.where(valid, values, 1.0), np.float32(1e-6))
    weights = np.where(
        valid,
        np.square(np.reciprocal(safe), dtype=np.float32),
        0.0,
    ).astype(np.float32, copy=False)
    totals = np.sum(weights, axis=1, keepdims=True, dtype=np.float32)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("FAISS neighbor weights could not be normalized.")
    return weights / totals


@dataclass(frozen=True)
class FaissFeatureIndex:
    index: Any
    vectors: np.ndarray
    path: str
    dimension: int
    vector_count: int
    index_type: str

    @classmethod
    def load(cls, index_path: str, expected_dimension: int) -> "FaissFeatureIndex":
        try:
            import faiss
        except ImportError as error:
            raise ValueError(
                "FAISS is required to use an RVC retrieval index. Install the "
                "engine-python core requirements."
            ) from error

        path = Path(index_path).expanduser().resolve()
        if path.suffix.lower() != ".index" or not path.is_file():
            raise ValueError("The selected retrieval index is not a readable .index file.")
        try:
            index = faiss.read_index(str(path))
        except Exception as error:
            raise ValueError(f"FAISS could not read the selected index: {error}") from error

        dimension = int(index.d)
        vector_count = int(index.ntotal)
        if dimension != expected_dimension:
            raise ValueError(
                "The retrieval index feature dimension does not match the RVC model "
                f"({dimension} != {expected_dimension})."
            )
        if vector_count < 1:
            raise ValueError("The selected retrieval index contains no vectors.")
        reconstructed_bytes = vector_count * dimension * np.dtype(np.float32).itemsize
        if reconstructed_bytes > MAX_RECONSTRUCTED_INDEX_BYTES:
            raise ValueError("The selected retrieval index is too large to load safely.")

        try:
            vectors = np.asarray(index.reconstruct_n(0, vector_count), dtype=np.float32)
        except Exception as error:
            raise ValueError(
                f"FAISS could not reconstruct the selected index vectors: {error}"
            ) from error
        if vectors.shape != (vector_count, dimension):
            raise ValueError(
                f"FAISS reconstructed an unexpected vector shape: {vectors.shape!r}."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("The retrieval index contains non-finite vectors.")
        return cls(
            index=index,
            vectors=np.ascontiguousarray(vectors),
            path=str(path),
            dimension=dimension,
            vector_count=vector_count,
            index_type=type(index).__name__,
        )

    def blend(self, features: np.ndarray, ratio: float) -> np.ndarray:
        index_ratio = validate_index_ratio(ratio)
        source = np.asarray(features, dtype=np.float32)
        if source.ndim != 3 or source.shape[0] != 1:
            raise ValueError(f"RVC content features have an invalid shape: {source.shape!r}.")
        if source.shape[2] != self.dimension:
            raise ValueError(
                "The live content feature dimension does not match the loaded index."
            )
        if index_ratio == 0.0:
            return source

        query = np.ascontiguousarray(source[0])
        neighbor_count = min(8, self.vector_count)
        distances, neighbors = self.index.search(query, neighbor_count)
        if neighbors.shape != distances.shape:
            raise ValueError("FAISS returned invalid neighbor identifiers.")
        valid_neighbors = neighbors >= 0
        distances = np.where(valid_neighbors, distances, np.inf)
        weights = inverse_distance_weights(distances)
        safe_neighbors = np.where(valid_neighbors, neighbors, 0)
        retrieved = np.sum(
            self.vectors[safe_neighbors] * weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        mixed = source[0] * np.float32(1.0 - index_ratio)
        mixed += retrieved * np.float32(index_ratio)
        if not np.isfinite(mixed).all():
            raise ValueError("FAISS retrieval produced non-finite content features.")
        return mixed[None]

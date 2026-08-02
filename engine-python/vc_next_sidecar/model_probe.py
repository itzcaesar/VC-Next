from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import zipfile


SUPPORTED_SUFFIXES = {".pth", ".onnx", ".index"}


def _index_affinity(checkpoint: Path, index: Path) -> int:
    """Rank a sibling FAISS index by how clearly its name belongs to the checkpoint."""
    checkpoint_stem = checkpoint.stem.casefold()
    index_stem = index.stem.casefold()
    score = 0
    if checkpoint_stem in index_stem or index_stem in checkpoint_stem:
        score += 100

    def tokens(value: str) -> set[str]:
        result: set[str] = set()
        for token in re.findall(r"[a-z]+\d*", value.casefold()):
            if token in {"added", "trained", "index", "flat", "nprobe"}:
                continue
            if token.startswith("ivf") or token.startswith("nprobe") or token.isdigit():
                continue
            result.add(token)
        return result

    score += 25 * len(tokens(checkpoint_stem) & tokens(index_stem))
    return score


def inspect_model(model_path: str) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise ValueError("The selected model file does not exist.")
    if not path.is_file():
        raise ValueError("The selected model path is not a file.")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Supported model files are .pth, .onnx, and .index.")

    size = path.stat().st_size
    if size == 0:
        raise ValueError("The selected model file is empty.")

    role = {
        ".pth": "rvc-checkpoint",
        ".onnx": "onnx-model",
        ".index": "faiss-index",
    }[suffix]
    container = "binary"
    if suffix == ".pth" and zipfile.is_zipfile(path):
        container = "pytorch-zip"

    sibling_indexes = []
    if suffix in {".pth", ".onnx"}:
        candidates = sorted(
            (candidate for candidate in path.parent.glob("*.index") if candidate.is_file()),
            key=lambda candidate: (-_index_affinity(path, candidate), candidate.name.casefold()),
        )
        sibling_indexes = [str(candidate) for candidate in candidates[:20]]

    recommended_index = sibling_indexes[0] if sibling_indexes else None
    pairing_note = (
        "A matching sibling .index was found and will be selected by default."
        if recommended_index
        else "No sibling .index was found; the checkpoint can still run without retrieval."
        if suffix == ".pth"
        else "Retrieval indexes are optional for this model format."
    )

    return {
        "path": str(path),
        "name": path.name,
        "extension": suffix,
        "role": role,
        "container": container,
        "sizeBytes": size,
        "siblingIndexes": sibling_indexes,
        "recommendedIndex": recommended_index,
        "packageComplete": suffix != ".pth" or recommended_index is not None,
        "pairingNote": pairing_note,
        "safeInspectionOnly": True,
        "checkpointLoaded": False,
        "warnings": [
            "Checkpoint contents have not been deserialized.",
            "Only load models from a trusted source with known usage rights.",
        ],
    }

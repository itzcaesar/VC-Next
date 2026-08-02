from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .model_probe import inspect_model

MAX_TRUSTED_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024


def summarize_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("The checkpoint root must be a mapping.")

    config = checkpoint.get("config")
    weights = checkpoint.get("weight")
    if not isinstance(config, Sequence) or isinstance(config, (str, bytes)):
        raise ValueError("The checkpoint does not contain an RVC config sequence.")
    if not isinstance(weights, Mapping):
        raise ValueError("The checkpoint does not contain an RVC weight mapping.")

    target_sample_rate = config[-1] if config else None
    if not isinstance(target_sample_rate, int) or target_sample_rate < 8_000:
        raise ValueError("The checkpoint target sample rate is invalid.")

    speaker_count = None
    embedding = weights.get("emb_g.weight")
    shape = getattr(embedding, "shape", None)
    if shape is not None and len(shape) >= 1:
        speaker_count = int(shape[0])

    version = str(checkpoint.get("version", "v1"))
    return {
        "rvcVersion": version,
        "targetSampleRate": target_sample_rate,
        "usesPitch": bool(checkpoint.get("f0", 1)),
        "speakerCount": speaker_count,
        "weightKeyCount": len(weights),
        "configLength": len(config),
    }


def load_weights_only_checkpoint(path: Path) -> Any:
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required for trusted checkpoint inspection.") from error

    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"The restricted checkpoint loader rejected the file: {error}") from error
    return checkpoint


def inspect_trusted_checkpoint(model_path: str) -> dict[str, Any]:
    metadata = inspect_model(model_path)
    if metadata["extension"] != ".pth":
        raise ValueError("Deep checkpoint inspection currently supports only .pth files.")
    if metadata["sizeBytes"] > MAX_TRUSTED_CHECKPOINT_BYTES:
        raise ValueError("The checkpoint exceeds the 2 GiB inspection limit.")

    checkpoint = load_weights_only_checkpoint(Path(metadata["path"]))
    summary = summarize_checkpoint(checkpoint)
    return {
        **metadata,
        **summary,
        "safeInspectionOnly": False,
        "checkpointLoaded": True,
        "loadPolicy": "torch-weights-only",
    }

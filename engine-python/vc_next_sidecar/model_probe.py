from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import zipfile
import json
import math


SUPPORTED_SUFFIXES = {".pth", ".onnx", ".index"}
_PACKAGE_ROOT_NAMES = {
    "model",
    "models",
    "model_dir",
    "voices",
    "voice_models",
    "voice models",
}
_INDEX_DIRECTORY_NAMES = {"index", "indexes"}
_PACKAGE_PARAMS_MAX_BYTES = 1_048_576
_LIVE_SAMPLE_RATE = 48_000


def _package_params_candidates(model: Path) -> tuple[Path, ...]:
    candidates = [model.parent / "params.json"]
    # Some exports place the checkpoint one level below the package metadata.
    if model.parent.parent != model.parent:
        candidates.append(model.parent.parent / "params.json")
    return tuple(candidates)


def read_model_package_params(model_path: str | Path) -> dict[str, Any]:
    """Read bounded, non-neural metadata shipped beside a model.

    w-okada stores the voice's pitch/index/protect/embedder choices in
    ``params.json``. Importing these values keeps a model sounding like its
    original configuration without deserializing checkpoint weights. Invalid
    or oversized metadata is ignored; the model can still be loaded manually.
    """

    model = Path(model_path).expanduser().resolve()
    for candidate in _package_params_candidates(model):
        try:
            if not candidate.is_file() or candidate.stat().st_size > _PACKAGE_PARAMS_MAX_BYTES:
                continue
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _bounded_float(value: Any, minimum: float, maximum: float) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        return None
    return numeric


def _chunk_frames_from_seconds(value: Any) -> int | None:
    seconds = _bounded_float(value, 0.064, 10.0)
    if seconds is None:
        return None
    frames = round(seconds * _LIVE_SAMPLE_RATE / 480) * 480
    return max(480, min(480_000, frames))


def _frame_count(value: Any) -> int | None:
    """Normalize a w-okada-style frame setting without trusting metadata."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    frames = int(numeric)
    if not 480 <= frames <= 480_000:
        return None
    return frames


def model_package_defaults(model_path: str | Path) -> dict[str, Any]:
    """Return safe, normalized settings from a w-okada-style params.json."""

    model = Path(model_path).expanduser().resolve()
    params = read_model_package_params(model)
    defaults: dict[str, Any] = {}
    pitch = _bounded_float(params.get("pitch_shift", params.get("pitchShift")), -50.0, 50.0)
    index_ratio = _bounded_float(params.get("index_ratio", params.get("indexRatio")), 0.0, 1.0)
    protect = _bounded_float(params.get("protect_ratio", params.get("protectRatio")), 0.0, 0.5)
    chunk_frames = _chunk_frames_from_seconds(params.get("chunk_sec", params.get("chunkSeconds")))
    extra_frames = _frame_count(
        params.get(
            "extra_convert_size",
            params.get("extraConvertSize", params.get("extraFrames")),
        )
    )
    embedder = params.get("embedder")
    pitch_estimator = params.get("pitch_estimator", params.get("pitchEstimator"))
    if pitch is not None:
        defaults["pitchShift"] = pitch
    if index_ratio is not None:
        defaults["indexRatio"] = index_ratio
    if protect is not None:
        defaults["protectRatio"] = protect
    if chunk_frames is not None:
        defaults["chunkFrames"] = chunk_frames
    if extra_frames is not None:
        defaults["extraFrames"] = extra_frames
    if isinstance(embedder, str) and embedder.strip():
        defaults["embedder"] = embedder.strip()
    if isinstance(pitch_estimator, str) and pitch_estimator.strip():
        defaults["pitchEstimator"] = pitch_estimator.strip()
    if model.suffix.casefold() in {".pth", ".onnx"}:
        sibling_indexes = _sibling_indexes(model)
        if sibling_indexes:
            defaults["recommendedIndex"] = str(sibling_indexes[0])
    return defaults


def _index_name_affinity(checkpoint: Path, index: Path) -> int:
    """Score only the filename relationship between a checkpoint and an index."""
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
            # RVC exports commonly append shared build metadata to every voice
            # filename (for example ``_v2_40k_e100``).  Those markers must not
            # make an unrelated index look like a companion for an ONNX voice.
            # Keep meaningful names such as ``e-girl`` and ``model`` while
            # ignoring version, epoch, sample-rate and step-count fragments.
            if len(token) == 1 or re.fullmatch(r"v\d+|e\d+|s\d+|f\d+|\d+k", token):
                continue
            result.add(token)
        return result

    score += 25 * len(tokens(checkpoint_stem) & tokens(index_stem))
    return score


def _index_affinity(checkpoint: Path, index: Path) -> int:
    """Rank a sibling FAISS index by how clearly its name belongs to the checkpoint."""
    score = _index_name_affinity(checkpoint, index)
    # Prefer an index beside the selected checkpoint.  w-okada commonly keeps
    # each voice in a numbered model_dir child, but some exported packages put
    # indexes in a nearby ``index`` folder or beside the model directory.  The
    # directory score keeps a nearby unrelated voice from being recommended
    # over an exact local companion when we inspect a larger package.
    if index.parent == checkpoint.parent:
        score += 1_000
    elif (
        index.parent.name.casefold() in _INDEX_DIRECTORY_NAMES
        and index.parent.parent == checkpoint.parent
    ):
        score += 500
    elif index.parent.parent == checkpoint.parent.parent:
        score += 200
    return score


def _index_search_directories(checkpoint: Path) -> list[Path]:
    """Return bounded, conventional directories for paired .index discovery.

    Do not recursively scan an arbitrary Downloads folder: model inspection is
    invoked while the file chooser is open and should stay fast and private.
    We only inspect the selected folder, explicit ``index`` folders, and the
    nearest directory that looks like a w-okada model package.
    """

    parent = checkpoint.parent
    directories: list[Path] = [parent]
    directories.extend(parent / name for name in sorted(_INDEX_DIRECTORY_NAMES))

    if parent.name.casefold() in _PACKAGE_ROOT_NAMES:
        directories.append(parent.parent)

    for ancestor in (parent, *parent.parents):
        if ancestor.name.casefold() not in _PACKAGE_ROOT_NAMES:
            continue
        directories.append(ancestor)
        try:
            directories.extend(
                child for child in ancestor.iterdir() if child.is_dir()
            )
        except OSError:
            # An inaccessible neighboring voice should not prevent importing
            # the selected checkpoint; the load path will report its own error.
            pass
        break

    unique: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        key = str(directory).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(directory)
    return unique


def _sibling_indexes(checkpoint: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for directory in _index_search_directories(checkpoint):
        try:
            if not directory.is_dir():
                continue
            entries = list(directory.iterdir())
        except OSError:
            continue
        for candidate in entries:
            if candidate.suffix.casefold() != ".index" or not candidate.is_file():
                continue
            # Neighboring model_dir slots can contain many indexes. Only
            # surface a cross-folder candidate when its filename shares a
            # meaningful token with the selected checkpoint; the explicit
            # chooser remains available for unusual naming schemes.
            if candidate.parent != checkpoint.parent and _index_name_affinity(checkpoint, candidate) == 0:
                continue
            key = str(candidate).casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda candidate: (-_index_affinity(checkpoint, candidate), candidate.name.casefold()),
    )


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
        sibling_indexes = [str(candidate) for candidate in _sibling_indexes(path)[:20]]

    recommended_index = sibling_indexes[0] if sibling_indexes else None
    pairing_note = (
        "A matching .index was found beside the checkpoint and will be selected by default."
        if recommended_index
        else "No sibling .index was found; the checkpoint can still run without retrieval."
        if suffix == ".pth"
        else "Retrieval indexes are optional for this model format."
    )
    if recommended_index and Path(recommended_index).parent != path.parent:
        pairing_note = (
            "A nearby .index was found in the surrounding model package and will be "
            "selected by default."
        )

    package_defaults = model_package_defaults(path) if suffix in {".pth", ".onnx"} else {}

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
        "modelDefaults": package_defaults,
        "safeInspectionOnly": True,
        "checkpointLoaded": False,
        "warnings": [
            "Checkpoint contents have not been deserialized.",
            "Only load models from a trusted source with known usage rights.",
        ],
    }

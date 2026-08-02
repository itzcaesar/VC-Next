from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from ..checkpoint_probe import (
    MAX_TRUSTED_CHECKPOINT_BYTES,
    load_weights_only_checkpoint,
    summarize_checkpoint,
)
from ..model_probe import inspect_model


@dataclass(frozen=True)
class LoadedGenerator:
    model: Any
    model_path: str
    rvc_version: str
    uses_pitch: bool
    target_sample_rate: int
    feature_channels: int
    speaker_count: int
    device: str
    precision: str
    parameter_count: int
    backend: str = "pytorch"


def _generator_class(rvc_version: str, uses_pitch: bool) -> tuple[type[Any], int]:
    from .infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    if rvc_version == "v2":
        return (
            SynthesizerTrnMs768NSFsid if uses_pitch else SynthesizerTrnMs768NSFsid_nono,
            768,
        )
    if rvc_version == "v1":
        return (
            SynthesizerTrnMs256NSFsid if uses_pitch else SynthesizerTrnMs256NSFsid_nono,
            256,
        )
    raise ValueError(f"Unsupported RVC checkpoint version: {rvc_version!r}.")


def load_generator(
    model_path: str,
    *,
    device: str = "cuda:0",
    use_half: bool = True,
) -> LoadedGenerator:
    metadata = inspect_model(model_path)
    if metadata["extension"] != ".pth":
        raise ValueError("The PyTorch RVC loader requires a .pth checkpoint.")
    if metadata["sizeBytes"] > MAX_TRUSTED_CHECKPOINT_BYTES:
        raise ValueError("The checkpoint exceeds the 2 GiB loading limit.")

    checkpoint = load_weights_only_checkpoint(Path(metadata["path"]))
    summary = summarize_checkpoint(checkpoint)
    weights = checkpoint["weight"]
    config = list(checkpoint["config"])
    speaker_count = summary["speakerCount"]
    if speaker_count is None or speaker_count < 1:
        raise ValueError("The checkpoint speaker embedding is missing or invalid.")
    if len(config) < 3:
        raise ValueError("The RVC config is too short to contain a speaker count.")
    config[-3] = speaker_count

    model_class, feature_channels = _generator_class(
        summary["rvcVersion"], summary["usesPitch"]
    )
    model = model_class(*config, is_half=use_half)
    if hasattr(model, "enc_q"):
        delattr(model, "enc_q")

    inference_weights = {
        key: value for key, value in weights.items() if not key.startswith("enc_q.")
    }
    incompatible = model.load_state_dict(inference_weights, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "The checkpoint does not exactly match the selected RVC architecture "
            f"(missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)})."
        )

    model = model.eval().to(device)
    model = model.half() if use_half else model.float()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return LoadedGenerator(
        model=model,
        model_path=metadata["path"],
        rvc_version=summary["rvcVersion"],
        uses_pitch=summary["usesPitch"],
        target_sample_rate=summary["targetSampleRate"],
        feature_channels=feature_channels,
        speaker_count=speaker_count,
        device=device,
        precision="fp16" if use_half else "fp32",
        parameter_count=parameter_count,
        backend="pytorch",
    )


def _onnx_parameters(model_path: Path) -> dict[str, Any]:
    candidates = (model_path.with_name("params.json"), model_path.parent / "params.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read the ONNX model metadata: {candidate}: {error}") from error
        if isinstance(value, dict):
            return value
    return {}


def _onnx_sample_rate(model_path: Path, params: dict[str, Any]) -> int:
    value = params.get("sample_rate")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("The ONNX model sample_rate metadata is invalid.") from error
    # w-okada exports commonly encode the generator rate in the filename when
    # a params.json file was not copied with the model package.
    match = re.search(r"(?:^|[_-])(32|40|48)k(?:$|[_-])", model_path.stem.casefold())
    if match:
        return int(match.group(1)) * 1_000
    return 40_000


def load_onnx_generator(model_path: str) -> LoadedGenerator:
    """Load the generator contract used by w-okada's exported RVC ONNX models."""

    metadata = inspect_model(model_path)
    if metadata["extension"] != ".onnx":
        raise ValueError("The ONNX RVC loader requires an .onnx model.")
    try:
        # On Windows the CUDA/cuBLAS DLLs shipped with the verified PyTorch
        # wheel are not always on the process DLL search path yet. Importing
        # Torch before creating an ONNX Runtime session registers that path and
        # prevents a silent generator fallback to CPU.
        import torch
        import onnxruntime as ort
    except ImportError as error:
        raise ValueError("PyTorch and ONNX Runtime are required for an exported RVC model.") from error
    del torch

    options = ort.SessionOptions()
    # ONNX Runtime can print a multi-line provider-loading traceback when a
    # CUDA DLL is missing even though it can safely fall back to CPU. The
    # selected provider is reported below; keep the protocol stderr clean.
    options.log_severity_level = 3
    available_providers = set(ort.get_available_providers())
    requested_providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        requested_providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(
        metadata["path"],
        sess_options=options,
        providers=requested_providers,
    )
    input_names = {item.name for item in session.get_inputs()}
    required = {"feats", "p_len", "pitch", "pitchf", "sid"}
    missing = sorted(required - input_names)
    if missing:
        raise ValueError(
            "The ONNX model does not expose the RVC generator inputs: "
            + ", ".join(missing)
        )
    if not session.get_outputs():
        raise ValueError("The ONNX model does not expose an audio output.")

    params = _onnx_parameters(Path(metadata["path"]))
    target_sample_rate = _onnx_sample_rate(Path(metadata["path"]), params)
    if not 8_000 <= target_sample_rate <= 192_000:
        raise ValueError("The ONNX model sample rate must be between 8 kHz and 192 kHz.")
    uses_pitch = bool(params.get("is_f0", True))
    speakers = params.get("speakers")
    if isinstance(speakers, (dict, list, tuple)) and speakers:
        speaker_count = len(speakers)
    else:
        speaker_count = 1
    feature_channels = 768
    feature_input = next(
        (item for item in session.get_inputs() if item.name == "feats"), None
    )
    feature_shape = feature_input.shape if feature_input is not None else None
    if feature_shape and isinstance(feature_shape[-1], int):
        feature_channels = int(feature_shape[-1])
    if feature_channels not in {256, 768}:
        raise ValueError(
            f"The ONNX RVC model uses unsupported feature width {feature_channels}; expected 256 or 768."
        )
    rvc_version = "v1" if feature_channels == 256 else "v2"
    providers = session.get_providers()
    device = "cuda:0" if providers and providers[0] == "CUDAExecutionProvider" else "cpu"
    return LoadedGenerator(
        model=session,
        model_path=metadata["path"],
        rvc_version=rvc_version,
        uses_pitch=uses_pitch,
        target_sample_rate=target_sample_rate,
        feature_channels=feature_channels,
        speaker_count=speaker_count,
        device=device,
        precision="fp32",
        parameter_count=0,
        backend="onnx",
    )

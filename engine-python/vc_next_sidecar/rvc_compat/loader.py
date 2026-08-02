from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    )

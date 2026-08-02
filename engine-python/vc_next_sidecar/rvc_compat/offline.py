from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .loader import LoadedGenerator, load_generator
from .retrieval import (
    FaissFeatureIndex,
    validate_index_ratio,
    validate_protect_ratio,
)


@dataclass(frozen=True)
class OfflineConversionResult:
    input_path: str
    output_path: str
    input_sample_rate: int
    output_sample_rate: int
    input_seconds: float
    output_seconds: float
    content_frames: int
    pitch_frames: int
    converted_frames: int
    voiced_pitch_frames: int
    content_ms: float
    pitch_ms: float
    retrieval_ms: float
    generator_ms: float
    total_processing_ms: float
    generator_headroom_x: float
    output_peak: float
    output_rms: float
    providers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OnnxFeaturePipeline:
    def __init__(self, contentvec_path: str, rmvpe_path: str) -> None:
        import torch  # preload CUDA/cuDNN libraries for ONNX Runtime
        import onnxruntime as ort

        del torch
        options = ort.SessionOptions()
        options.log_severity_level = 3
        requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.content_session = ort.InferenceSession(
            str(Path(contentvec_path).resolve()),
            sess_options=options,
            providers=requested_providers,
        )
        self.pitch_session = ort.InferenceSession(
            str(Path(rmvpe_path).resolve()),
            sess_options=options,
            providers=requested_providers,
        )
        self.providers = self.content_session.get_providers()
        if not self.providers or self.providers[0] != "CUDAExecutionProvider":
            raise ValueError("The ONNX feature pipeline did not activate its CUDA provider.")
        if self.pitch_session.get_providers()[0] != "CUDAExecutionProvider":
            raise ValueError("The ONNX pitch pipeline did not activate its CUDA provider.")

    def extract_content(self, waveform_16k: np.ndarray) -> np.ndarray:
        features = self.content_session.run(
            ["unit12"],
            {"audio": waveform_16k[None].astype(np.float32, copy=False)},
        )[0]
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] != 768:
            raise ValueError(f"ContentVec returned an unexpected shape: {features.shape!r}.")
        if not np.isfinite(features).all():
            raise ValueError("ContentVec returned non-finite features.")
        return features

    def extract_pitch(
        self, waveform_16k: np.ndarray, threshold: float = 0.03
    ) -> np.ndarray:
        pitch = self.pitch_session.run(
            ["pitchf"],
            {
                "waveform": waveform_16k[None].astype(np.float32, copy=False),
                "threshold": np.asarray([threshold], dtype=np.float32),
            },
        )[0]
        pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
        if not np.isfinite(pitch).all():
            raise ValueError("RMVPE returned non-finite pitch values.")
        return pitch


def _load_mono_16k(input_path: str, max_seconds: float | None) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional

    audio, source_rate = sf.read(input_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    waveform = torch.from_numpy(mono)
    if source_rate != 16_000:
        waveform = audio_functional.resample(waveform, source_rate, 16_000)
    if max_seconds is not None:
        waveform = waveform[: round(max_seconds * 16_000)]
    if waveform.numel() < 1_600:
        raise ValueError("At least 100 ms of input audio is required.")
    return waveform.contiguous().numpy(), source_rate


def _coarse_pitch(pitchf: np.ndarray) -> np.ndarray:
    minimum_hz = 50.0
    maximum_hz = 1_100.0
    mel_minimum = 1127.0 * np.log1p(minimum_hz / 700.0)
    mel_maximum = 1127.0 * np.log1p(maximum_hz / 700.0)
    pitch_mel = 1127.0 * np.log1p(pitchf / 700.0)
    voiced = pitch_mel > 0
    pitch_mel[voiced] = (
        (pitch_mel[voiced] - mel_minimum) * 254.0 / (mel_maximum - mel_minimum)
        + 1.0
    )
    return np.rint(np.clip(pitch_mel, 1.0, 255.0)).astype(np.int64)


def _run_generator(
    loaded: LoadedGenerator,
    content: np.ndarray,
    pitchf: np.ndarray,
    *,
    speaker_id: int,
    pitch_shift: float,
    retrieval_index: FaissFeatureIndex | None = None,
    index_ratio: float = 0.0,
    protect_ratio: float = 0.5,
) -> tuple[np.ndarray, int, int, float, float]:
    import torch
    from torch.nn import functional as torch_functional

    if not loaded.uses_pitch:
        raise ValueError("The first offline pipeline currently requires a pitch-enabled RVC model.")
    if speaker_id < 0 or speaker_id >= loaded.speaker_count:
        raise ValueError(
            f"Speaker ID {speaker_id} is outside the checkpoint range 0..{loaded.speaker_count - 1}."
        )

    index_ratio = validate_index_ratio(index_ratio)
    protect_ratio = validate_protect_ratio(protect_ratio)
    if index_ratio > 0.0 and retrieval_index is None:
        raise ValueError("An index ratio above zero requires a loaded retrieval index.")

    source_content = np.asarray(content, dtype=np.float32)
    if source_content.shape[-1] < loaded.feature_channels:
        raise ValueError(
            "The content encoder returned fewer channels than the RVC model requires."
        )
    source_content = np.ascontiguousarray(
        source_content[..., : loaded.feature_channels]
    )
    retrieval_started = perf_counter()
    converted_content = (
        retrieval_index.blend(source_content, index_ratio)
        if retrieval_index is not None and index_ratio > 0.0
        else source_content
    )
    retrieval_ms = (perf_counter() - retrieval_started) * 1_000.0

    dtype = torch.float16 if loaded.precision == "fp16" else torch.float32
    features = torch.from_numpy(converted_content).to(device=loaded.device, dtype=dtype)
    features = torch_functional.interpolate(
        features.permute(0, 2, 1), scale_factor=2
    ).permute(0, 2, 1)
    protected_source = None
    if retrieval_index is not None and index_ratio > 0.0 and protect_ratio < 0.5:
        protected_source = torch.from_numpy(source_content).to(
            device=loaded.device, dtype=dtype
        )
        protected_source = torch_functional.interpolate(
            protected_source.permute(0, 2, 1), scale_factor=2
        ).permute(0, 2, 1)
    shifted_pitch = pitchf * (2.0 ** (pitch_shift / 12.0))
    frame_count = min(features.shape[1], shifted_pitch.shape[0])
    if frame_count < 2:
        raise ValueError("The extracted feature sequence is too short for inference.")
    features = features[:, :frame_count]
    if protected_source is not None:
        protected_source = protected_source[:, :frame_count]
    shifted_pitch = shifted_pitch[-frame_count:]
    pitch = torch.from_numpy(_coarse_pitch(shifted_pitch)).unsqueeze(0).to(loaded.device)
    pitchf_tensor = (
        torch.from_numpy(shifted_pitch)
        .unsqueeze(0)
        .to(device=loaded.device, dtype=dtype)
    )
    if protected_source is not None:
        voiced_weight = torch.where(
            pitchf_tensor > 0,
            torch.ones_like(pitchf_tensor),
            torch.full_like(pitchf_tensor, protect_ratio),
        ).unsqueeze(-1)
        features = features * voiced_weight + protected_source * (1.0 - voiced_weight)
    lengths = torch.tensor([frame_count], device=loaded.device, dtype=torch.long)
    speaker = torch.tensor([speaker_id], device=loaded.device, dtype=torch.long)

    torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        converted = loaded.model.infer(
            features, lengths, pitch, pitchf_tensor, speaker
        )[0][0, 0].float()
    torch.cuda.synchronize()
    generator_ms = (perf_counter() - started) * 1_000.0
    output = torch.clamp(converted, -1.0, 1.0).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("The RVC generator returned non-finite audio.")
    return (
        output,
        int(features.shape[1]),
        int(pitchf_tensor.shape[1]),
        generator_ms,
        retrieval_ms,
    )


def convert_file(
    *,
    input_path: str,
    output_path: str,
    model_path: str,
    contentvec_path: str,
    rmvpe_path: str,
    index_path: str | None = None,
    speaker_id: int = 0,
    pitch_shift: float = 0.0,
    index_ratio: float = 0.0,
    protect_ratio: float = 0.5,
    max_seconds: float | None = None,
) -> OfflineConversionResult:
    import soundfile as sf

    total_started = perf_counter()
    waveform, input_sample_rate = _load_mono_16k(input_path, max_seconds)
    loaded = load_generator(model_path)
    features = OnnxFeaturePipeline(contentvec_path, rmvpe_path)
    retrieval_index = (
        FaissFeatureIndex.load(index_path, loaded.feature_channels)
        if index_path
        else None
    )

    started = perf_counter()
    content = features.extract_content(waveform)
    content_ms = (perf_counter() - started) * 1_000.0

    started = perf_counter()
    pitchf = features.extract_pitch(waveform)
    pitch_ms = (perf_counter() - started) * 1_000.0

    output, content_frames, pitch_frames, generator_ms, retrieval_ms = _run_generator(
        loaded,
        content,
        pitchf,
        speaker_id=speaker_id,
        pitch_shift=pitch_shift,
        retrieval_index=retrieval_index,
        index_ratio=index_ratio,
        protect_ratio=protect_ratio,
    )
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, loaded.target_sample_rate, subtype="PCM_16")
    output_seconds = output.shape[0] / loaded.target_sample_rate
    total_processing_ms = (perf_counter() - total_started) * 1_000.0
    return OfflineConversionResult(
        input_path=str(Path(input_path).resolve()),
        output_path=str(destination),
        input_sample_rate=input_sample_rate,
        output_sample_rate=loaded.target_sample_rate,
        input_seconds=waveform.shape[0] / 16_000.0,
        output_seconds=output_seconds,
        content_frames=content_frames,
        pitch_frames=pitch_frames,
        converted_frames=int(output.shape[0]),
        voiced_pitch_frames=int(np.count_nonzero(pitchf > 0)),
        content_ms=content_ms,
        pitch_ms=pitch_ms,
        retrieval_ms=retrieval_ms,
        generator_ms=generator_ms,
        total_processing_ms=total_processing_ms,
        generator_headroom_x=(output_seconds * 1_000.0 / generator_ms),
        output_peak=float(np.max(np.abs(output))),
        output_rms=float(np.sqrt(np.mean(np.square(output)))),
        providers=features.providers,
    )

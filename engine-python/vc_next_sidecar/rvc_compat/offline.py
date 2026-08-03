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


def _select_content_output(outputs: list[Any], feature_channels: int) -> str:
    """Choose the ContentVec head matching an RVC v1 or v2 generator."""

    if feature_channels not in {256, 768}:
        raise ValueError(f"Unsupported ContentVec feature width: {feature_channels}.")
    preferred_names = {
        256: ("units9", "unit9", "units"),
        768: ("unit12", "unit12s", "units12"),
    }[feature_channels]
    by_name = {str(output.name): output for output in outputs}
    for name in preferred_names:
        if name in by_name:
            return name
    for output in outputs:
        shape = getattr(output, "shape", None)
        if shape and isinstance(shape[-1], int) and int(shape[-1]) == feature_channels:
            return str(output.name)
    available = ", ".join(sorted(by_name)) or "none"
    raise ValueError(
        f"ContentVec does not expose a {feature_channels}-channel output (available: {available})."
    )


class OnnxFeaturePipeline:
    def __init__(self, contentvec_path: str, rmvpe_path: str, *, feature_channels: int = 768) -> None:
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
        self.content_output_name = _select_content_output(
            list(self.content_session.get_outputs()), feature_channels
        )
        self.feature_channels = feature_channels
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
            [self.content_output_name],
            {"audio": waveform_16k[None].astype(np.float32, copy=False)},
        )[0]
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] != self.feature_channels:
            raise ValueError(f"ContentVec returned an unexpected shape: {features.shape!r}.")
        if not np.isfinite(features).all():
            raise ValueError("ContentVec returned non-finite features.")
        return features

    def extract_pitch(
        self,
        waveform_16k: np.ndarray,
        threshold: float = 0.30,
        *,
        silence_front_samples: int = 0,
        output_frames: int | None = None,
    ) -> np.ndarray:
        """Extract F0 with the same front-context rule as w-okada's RMVPE path.

        RVCr2 keeps zero-padded conversion context in the ContentVec input,
        but RMVPE only sees the portion after ``silence_front``.  Its result is
        copied into the tail of the full pitch buffer, leaving the front frames
        at zero.  Applying that distinction here avoids boundary F0 decisions
        from a synthetic zero prefix while preserving the feature geometry.
        """
        waveform = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("RMVPE requires a non-empty waveform.")
        offset = max(0, int(silence_front_samples))
        # RVC pitch frames advance in 160-sample (10 ms) windows.  Upstream
        # floors the front offset to a complete window before trimming.
        offset = min(waveform.size, (offset // 160) * 160)
        target_length = max(160, waveform.size - offset)
        source = waveform[-target_length:]
        pitch = self.pitch_session.run(
            ["pitchf"],
            {
                "waveform": source[None].astype(np.float32, copy=False),
                "threshold": np.asarray([threshold], dtype=np.float32),
            },
        )[0]
        pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
        if not np.isfinite(pitch).all():
            raise ValueError("RMVPE returned non-finite pitch values.")
        frame_count = (
            max(1, waveform.size // 160)
            if output_frames is None
            else max(1, int(output_frames))
        )
        restored = np.zeros(frame_count, dtype=np.float32)
        copy_count = min(frame_count, pitch.size)
        if copy_count:
            restored[-copy_count:] = pitch[-copy_count:]
        return restored


def _load_fairseq_checkpoint_utils() -> Any:
    """Import Fairseq on Python 3.11 without changing the installed package.

    w-okada ships Fairseq 0.12.x, whose configuration dataclasses predate
    Python 3.11's mutable-default validation.  The compatibility shim is
    scoped to the import and is only used when a user explicitly selects a
    Fairseq HuBERT ``.pt`` asset; the normal ONNX path never imports Fairseq.
    """

    import dataclasses

    original_get_field = dataclasses._get_field
    missing = dataclasses.MISSING

    def compatible_get_field(cls: Any, name: str, annotation: Any, kw_only: bool) -> Any:
        default = getattr(cls, name, missing)
        changed = False
        if getattr(cls, "__module__", "").startswith("fairseq"):
            value = default.default if isinstance(default, dataclasses.Field) else default
            if value is not missing and value.__class__.__hash__ is None:
                if isinstance(default, dataclasses.Field):
                    replacement = dataclasses.field(
                        default_factory=value.__class__,
                        init=default.init,
                        repr=default.repr,
                        hash=default.hash,
                        compare=default.compare,
                        metadata=default.metadata,
                        kw_only=default.kw_only,
                    )
                else:
                    replacement = dataclasses.field(default_factory=value.__class__)
                setattr(cls, name, replacement)
                changed = True
        try:
            return original_get_field(cls, name, annotation, kw_only)
        finally:
            if changed:
                setattr(cls, name, default)

    dataclasses._get_field = compatible_get_field
    config_store = None
    original_store = None
    try:
        try:
            from hydra.core.config_store import ConfigStore

            config_store = ConfigStore
            original_store = ConfigStore.store
            # Fairseq's import-time registry is not needed for inference and
            # OmegaConf 2.3 rejects one of its legacy MISSING sentinels.
            ConfigStore.store = lambda self, *args, **kwargs: None
        except ImportError:
            pass
        from fairseq import checkpoint_utils

        return checkpoint_utils
    except ImportError as error:
        raise RuntimeError(
            "Fairseq HuBERT support is not installed. Install the optional "
            "Fairseq compatibility dependency or select a ContentVec .onnx embedder."
        ) from error
    finally:
        dataclasses._get_field = original_get_field
        if config_store is not None and original_store is not None:
            config_store.store = original_store


class FairseqHubertFeaturePipeline:
    """Feature extractor compatible with w-okada's Fairseq HuBERT fallback."""

    def __init__(self, hubert_path: str, rmvpe_path: str, *, feature_channels: int = 768) -> None:
        if feature_channels not in {256, 768}:
            raise ValueError(f"Unsupported Fairseq HuBERT feature width: {feature_channels}.")
        import torch
        import onnxruntime as ort

        checkpoint_utils = _load_fairseq_checkpoint_utils()
        original_load = torch.load

        def trusted_checkpoint_load(*args: Any, **kwargs: Any) -> Any:
            # Fairseq's official HuBERT checkpoint contains task metadata and
            # cannot be loaded with PyTorch's weights-only restriction. This
            # path is reachable only for the explicit user-selected embedder.
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = trusted_checkpoint_load
        try:
            models, _saved_cfg, _task = checkpoint_utils.load_model_ensemble_and_task(
                [str(Path(hubert_path).resolve())],
                suffix="",
            )
        finally:
            torch.load = original_load
        if not models:
            raise ValueError("The Fairseq HuBERT checkpoint did not contain a model.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models[0].eval().to(self.device)
        self.use_half = self.device.type == "cuda"
        if self.use_half:
            self.model = self.model.half()
        self.feature_channels = feature_channels
        self.output_layer = 12 if feature_channels == 768 else 9
        self.use_final_proj = feature_channels == 256

        options = ort.SessionOptions()
        options.log_severity_level = 3
        requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.pitch_session = ort.InferenceSession(
            str(Path(rmvpe_path).resolve()),
            sess_options=options,
            providers=requested_providers,
        )
        self.providers = [
            f"FairseqHuBERT({self.device.type})",
            *self.pitch_session.get_providers(),
        ]
        if self.device.type != "cuda":
            raise ValueError("The Fairseq HuBERT feature pipeline requires CUDA.")
        if self.pitch_session.get_providers()[0] != "CUDAExecutionProvider":
            raise ValueError("The ONNX pitch pipeline did not activate its CUDA provider.")

    def extract_content(self, waveform_16k: np.ndarray) -> np.ndarray:
        import torch

        waveform = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("Fairseq HuBERT requires a non-empty waveform.")
        source = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(source.shape, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_half,
            ):
                logits = self.model.extract_features(
                    source=source,
                    padding_mask=padding_mask,
                    output_layer=self.output_layer,
                )[0]
                features = self.model.final_proj(logits) if self.use_final_proj else logits
        result = features.detach().float().cpu().numpy()
        if result.ndim != 3 or result.shape[2] != self.feature_channels:
            raise ValueError(f"Fairseq HuBERT returned an unexpected shape: {result.shape!r}.")
        if not np.isfinite(result).all():
            raise ValueError("Fairseq HuBERT returned non-finite features.")
        return result

    def extract_pitch(
        self,
        waveform_16k: np.ndarray,
        threshold: float = 0.30,
        *,
        silence_front_samples: int = 0,
        output_frames: int | None = None,
    ) -> np.ndarray:
        """Use the same RMVPE front-context rule as :class:`OnnxFeaturePipeline`."""

        waveform = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("RMVPE requires a non-empty waveform.")
        offset = max(0, int(silence_front_samples))
        offset = min(waveform.size, (offset // 160) * 160)
        target_length = max(160, waveform.size - offset)
        source = waveform[-target_length:]
        pitch = self.pitch_session.run(
            ["pitchf"],
            {
                "waveform": source[None].astype(np.float32, copy=False),
                "threshold": np.asarray([threshold], dtype=np.float32),
            },
        )[0]
        pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
        if not np.isfinite(pitch).all():
            raise ValueError("RMVPE returned non-finite pitch values.")
        frame_count = (
            max(1, waveform.size // 160)
            if output_frames is None
            else max(1, int(output_frames))
        )
        restored = np.zeros(frame_count, dtype=np.float32)
        copy_count = min(frame_count, pitch.size)
        if copy_count:
            restored[-copy_count:] = pitch[-copy_count:]
        return restored


def load_feature_pipeline(
    feature_path: str,
    rmvpe_path: str,
    *,
    feature_channels: int = 768,
) -> OnnxFeaturePipeline | FairseqHubertFeaturePipeline:
    """Construct the feature extractor selected by the asset extension.

    ContentVec ONNX remains the default compatibility path.  A Fairseq
    HuBERT checkpoint is opt-in because it requires the optional legacy
    Fairseq runtime and can be substantially slower to initialize.
    """

    path = Path(feature_path).expanduser().resolve()
    if path.suffix.casefold() in {".pt", ".pth"}:
        return FairseqHubertFeaturePipeline(
            str(path),
            rmvpe_path,
            feature_channels=feature_channels,
        )
    return OnnxFeaturePipeline(
        str(path),
        rmvpe_path,
        feature_channels=feature_channels,
    )


def _load_mono_16k(input_path: str, max_seconds: float | None) -> tuple[np.ndarray, int]:
    import soundfile as sf
    from .resampling import resample_kaiser_fast

    audio, source_rate = sf.read(input_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    waveform = resample_kaiser_fast(mono, source_rate, 16_000)
    if max_seconds is not None:
        waveform = waveform[: round(max_seconds * 16_000)]
    if waveform.size < 1_600:
        raise ValueError("At least 100 ms of input audio is required.")
    return np.ascontiguousarray(waveform, dtype=np.float32), source_rate


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


def _rvc_protect_weights(pitchf_tensor: Any, protect_ratio: float) -> Any:
    """Match w-okada's voiced/unvoiced retrieval protection mask.

    The upstream implementation first promotes every positive F0 value to a
    full retrieval weight, then replaces values below one (including the
    unvoiced zero and any fractional estimator output) with ``protect``.
    Keeping the two comparisons separate is deliberate; a single
    ``pitchf > 0`` branch is subtly different for fractional F0 values.
    """

    import torch

    voiced_weight = torch.where(
        pitchf_tensor > 0,
        torch.ones_like(pitchf_tensor),
        pitchf_tensor,
    )
    voiced_weight = torch.where(
        pitchf_tensor < 1,
        torch.full_like(pitchf_tensor, protect_ratio),
        voiced_weight,
    )
    return voiced_weight.unsqueeze(-1)


def _blend_retrieval_with_silence_front(
    source_content: np.ndarray,
    retrieval_index: FaissFeatureIndex,
    ratio: float,
    silence_front_frames: int,
    front_features: np.ndarray | None = None,
) -> np.ndarray:
    """Apply retrieval while preserving the front context used by w-okada.

    The upstream pipeline omits the zero-padded front frames from the FAISS
    query, then reconstructs that front edge from the pre-retrieval features
    before cropping back to the original feature length. This matters when a
    live window contains the extra conversion context used to prime the
    generator.
    """

    if silence_front_frames <= 0:
        return retrieval_index.blend(source_content, ratio)
    frame_count = source_content.shape[1]
    offset = min(int(silence_front_frames), frame_count - 1)
    retrieved = retrieval_index.blend(source_content[:, offset:, :], ratio)[0]
    if front_features is None:
        # Offline conversion has no preceding live feature buffer.  Retain
        # the historical behavior for that path; the live worker supplies
        # w-okada's rolling pre-retrieval buffer explicitly.
        preserved_front = source_content[0, :offset:2].astype(
            np.float32, copy=False
        )
    else:
        preserved_front = np.asarray(front_features, dtype=np.float32).reshape(
            -1, source_content.shape[2]
        )[:offset:2]
        if preserved_front.shape[0] < (offset + 1) // 2:
            preserved_front = np.concatenate(
                (
                    np.zeros(
                        ((offset + 1) // 2 - preserved_front.shape[0], source_content.shape[2]),
                        dtype=np.float32,
                    ),
                    preserved_front,
                ),
                axis=0,
            )
        preserved_front = preserved_front[: (offset + 1) // 2]
    prefix = np.concatenate(
        (
            np.zeros((offset, source_content.shape[2]), dtype=np.float32),
            preserved_front,
            retrieved,
        ),
        axis=0,
    )
    return np.ascontiguousarray(prefix[-frame_count:][None], dtype=np.float32)


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
    silence_front_frames: int = 0,
    convert_length: int | None = None,
    front_features: np.ndarray | None = None,
    feature_buffer_out: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, int, int, float, float]:
    if loaded.backend == "onnx":
        return _run_onnx_generator(
            loaded,
            content,
            pitchf,
            speaker_id=speaker_id,
            pitch_shift=pitch_shift,
            retrieval_index=retrieval_index,
            index_ratio=index_ratio,
            protect_ratio=protect_ratio,
            silence_front_frames=silence_front_frames,
            convert_length=convert_length,
            front_features=front_features,
            feature_buffer_out=feature_buffer_out,
        )
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
        _blend_retrieval_with_silence_front(
            source_content,
            retrieval_index,
            index_ratio,
            silence_front_frames,
            front_features,
        )
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
        voiced_weight = _rvc_protect_weights(pitchf_tensor, protect_ratio)
        features = features * voiced_weight + protected_source * (1.0 - voiced_weight)
    if feature_buffer_out is not None:
        feature_buffer_out.clear()
        feature_buffer_out.append(
            features.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    lengths = torch.tensor([frame_count], device=loaded.device, dtype=torch.long)
    speaker = torch.tensor([speaker_id], device=loaded.device, dtype=torch.long)

    torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        converted = loaded.model.infer(
            features,
            lengths,
            pitch,
            pitchf_tensor,
            speaker,
            convert_length=convert_length,
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


def _run_onnx_generator(
    loaded: LoadedGenerator,
    content: np.ndarray,
    pitchf: np.ndarray,
    *,
    speaker_id: int,
    pitch_shift: float,
    retrieval_index: FaissFeatureIndex | None = None,
    index_ratio: float = 0.0,
    protect_ratio: float = 0.5,
    silence_front_frames: int = 0,
    convert_length: int | None = None,
    front_features: np.ndarray | None = None,
    feature_buffer_out: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, int, int, float, float]:
    """Run the five-input generator signature emitted by w-okada's ONNX exporter."""

    import torch
    from torch.nn import functional as torch_functional

    if not loaded.uses_pitch:
        raise ValueError("The live ONNX pipeline currently requires a pitch-enabled RVC model.")
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
        raise ValueError("The content encoder returned fewer channels than the ONNX model requires.")
    source_content = np.ascontiguousarray(source_content[..., : loaded.feature_channels])
    retrieval_started = perf_counter()
    converted_content = (
        _blend_retrieval_with_silence_front(
            source_content,
            retrieval_index,
            index_ratio,
            silence_front_frames,
            front_features,
        )
        if retrieval_index is not None and index_ratio > 0.0
        else source_content
    )
    retrieval_ms = (perf_counter() - retrieval_started) * 1_000.0

    features = torch.from_numpy(np.asarray(converted_content, dtype=np.float32))
    features = torch_functional.interpolate(
        features.permute(0, 2, 1), scale_factor=2
    ).permute(0, 2, 1)
    protected_source = None
    if retrieval_index is not None and index_ratio > 0.0 and protect_ratio < 0.5:
        protected_source = torch.from_numpy(source_content)
        protected_source = torch_functional.interpolate(
            protected_source.permute(0, 2, 1), scale_factor=2
        ).permute(0, 2, 1)
    shifted_pitch = np.asarray(pitchf, dtype=np.float32) * (2.0 ** (pitch_shift / 12.0))
    frame_count = min(features.shape[1], shifted_pitch.shape[0])
    if frame_count < 2:
        raise ValueError("The extracted feature sequence is too short for ONNX inference.")
    features = features[:, :frame_count]
    if protected_source is not None:
        protected_source = protected_source[:, :frame_count]
    shifted_pitch = shifted_pitch[-frame_count:]
    pitch = torch.from_numpy(_coarse_pitch(shifted_pitch)).unsqueeze(0)
    pitchf_tensor = torch.from_numpy(shifted_pitch).unsqueeze(0)
    if protected_source is not None:
        voiced_weight = _rvc_protect_weights(pitchf_tensor, protect_ratio)
        features = features * voiced_weight + protected_source * (1.0 - voiced_weight)
    if feature_buffer_out is not None:
        feature_buffer_out.clear()
        feature_buffer_out.append(
            features.detach().cpu().numpy().astype(np.float32, copy=False)
        )

    inputs = {
        "feats": features.contiguous().numpy().astype(np.float32, copy=False),
        "p_len": np.asarray([frame_count], dtype=np.int64),
        "pitch": pitch.contiguous().numpy().astype(np.int64, copy=False),
        "pitchf": pitchf_tensor.contiguous().numpy().astype(np.float32, copy=False),
        "sid": np.asarray([speaker_id], dtype=np.int64),
    }
    started = perf_counter()
    converted = loaded.model.run(None, inputs)[0]
    generator_ms = (perf_counter() - started) * 1_000.0
    output = np.clip(np.asarray(converted, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if not np.isfinite(output).all():
        raise ValueError("The ONNX RVC generator returned non-finite audio.")
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
    f0_threshold: float = 0.30,
    max_seconds: float | None = None,
) -> OfflineConversionResult:
    import soundfile as sf

    total_started = perf_counter()
    waveform, input_sample_rate = _load_mono_16k(input_path, max_seconds)
    loaded = load_generator(model_path)
    features = load_feature_pipeline(
        contentvec_path,
        rmvpe_path,
        feature_channels=loaded.feature_channels,
    )
    retrieval_index = (
        FaissFeatureIndex.load(index_path, loaded.feature_channels)
        if index_path
        else None
    )

    started = perf_counter()
    content = features.extract_content(waveform)
    content_ms = (perf_counter() - started) * 1_000.0

    started = perf_counter()
    pitchf = features.extract_pitch(waveform, f0_threshold)
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

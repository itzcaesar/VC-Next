from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, BinaryIO

import numpy as np

from .framed_protocol import (
    AUDIO_REQUEST,
    AUDIO_RESPONSE,
    ERROR_RESPONSE,
    JSON_REQUEST,
    JSON_RESPONSE,
    PROTOCOL_VERSION,
    SHUTDOWN,
    Frame,
    read_frame,
    write_frame,
)
from .model_probe import model_package_defaults
from .rvc_compat.loader import LoadedGenerator, load_generator, load_onnx_generator
from .rvc_compat.offline import (
    OnnxFeaturePipeline,
    _run_generator,
    load_feature_pipeline,
)
from .rvc_compat.retrieval import (
    FaissFeatureIndex,
    validate_index_ratio,
    validate_protect_ratio,
)
from .rvc_compat.resampling import resample_kaiser_fast
from .streaming import SolaStitcher
from .stream_config import (
    get_stream_profile,
    validate_f0_threshold,
    validate_pitch_shift,
)


LIVE_INPUT_SAMPLE_RATE = 48_000
LIVE_CHUNK_FRAMES = 9_600
LIVE_ANALYSIS_FRAMES = 24_000
LIVE_CROSSFADE_FRAMES = 4_096
LIVE_SOLA_SEARCH_FRAMES = 576
FEATURE_SAMPLE_RATE = 16_000
CALIBRATION_SAMPLES = 3

# A model can produce a small non-zero waveform when the source frame is empty.
# That is useful for a neural decoder's internal calibration, but it is not
# useful on a microphone route: an idle input should stay digitally silent.
# These thresholds are deliberately conservative and are measured on the
# normalized 32-bit float input. Physical interfaces can sit below -70 dBFS,
# while virtual buses such as Voicemeeter commonly expose a continuous idle
# floor around -56 dBFS; the higher floor prevents that bus hiss from waking
# the decoder without touching normal speech levels.
SILENCE_RMS_THRESHOLD = 0.002
SILENCE_PEAK_THRESHOLD = 0.0015
# A quiet syllable can have a lower RMS than a full speech hop.  Treat a
# short, concentrated burst as speech even when its RMS is below the idle
# floor.  Conversely, a single hot sample on an otherwise empty virtual bus
# must not wake the neural decoder and recreate the static problem this gate
# is meant to prevent.
SILENCE_ACTIVITY_RATIO = 0.02
SILENCE_ACTIVITY_MULTIPLIER = 1.5


def discover_feature_models(model_path: str, embedder_hint: str | None = None) -> tuple[str, str]:
    model = Path(model_path).expanduser().resolve()
    default_contentvec_names = (
        "contentvec-f.onnx",
        "contentvec.onnx",
        "contentvec_f.onnx",
        "hubert_base_l12.onnx",
        "hubert_base.onnx",
        "rinna_hubert_base-f.onnx",
        "rinna_hubert_base.onnx",
    )
    fairseq_hubert_names = (
        "hubert_base.pt",
        "hubert_base.pth",
        "hubert_base_l12.pt",
        "hubert_base_l12.pth",
    )
    hint = (embedder_hint or "").casefold()
    if "rinna_hubert" in hint:
        preferred_contentvec_names = (
            "rinna_hubert_base-f.onnx",
            "rinna_hubert_base.onnx",
            *default_contentvec_names,
        )
        contentvec_folders = ("rinna_hubert", "hubert", "contentvec")
    elif "hubert_base_l12" in hint or "contentvec" in hint:
        # w-okada's current RVC packages label this embedder
        # ``hubert_base_l12`` in params.json but resolve it to the canonical
        # modules/contentvec/contentvec-f.onnx asset.  Prefer that path when
        # both ContentVec and Rinna Hubert are present; the latter remains a
        # fallback and can still be selected explicitly by its hint/path.
        preferred_contentvec_names = (
            "contentvec-f.onnx",
            "contentvec.onnx",
            "contentvec_f.onnx",
            *default_contentvec_names,
        )
        contentvec_folders = ("contentvec", "rinna_hubert", "hubert")
    elif "hubert_base_l9" in hint:
        preferred_contentvec_names = (
            "hubert_base_l9fp.onnx",
            "hubert_base_l9.onnx",
            *default_contentvec_names,
        )
        contentvec_folders = ("hubert", "contentvec", "rinna_hubert")
    else:
        preferred_contentvec_names = default_contentvec_names
        contentvec_folders = ("contentvec", "rinna_hubert", "hubert")
    rmvpe_names = (
        "rmvpe_20231006.onnx",
        "rmvpe_onnx.onnx",
        "rmvpe.onnx",
        "rmvpe_2023.onnx",
    )
    searched: list[str] = []
    for root in (model.parent, *model.parents):
        for modules in (root / "modules", root / "main" / "modules"):
            searched.append(str(modules))
            contentvec_candidates = (
                *(modules / folder / name for folder in contentvec_folders for name in preferred_contentvec_names),
                *(modules / name for name in preferred_contentvec_names),
            )
            fairseq_candidates = (
                *(modules / folder / name for folder in ("contentvec", "hubert") for name in fairseq_hubert_names),
                *(modules / name for name in fairseq_hubert_names),
            )
            rmvpe_candidates = (
                *(modules / "rmvpe" / name for name in rmvpe_names),
                *(modules / name for name in rmvpe_names),
            )
            contentvec = next((path for path in contentvec_candidates if path.is_file()), None)
            if contentvec is None:
                contentvec = next((path for path in fairseq_candidates if path.is_file()), None)
            rmvpe = next((path for path in rmvpe_candidates if path.is_file()), None)
            if contentvec and rmvpe:
                return str(contentvec), str(rmvpe)
    missing = []
    if not any(
        Path(candidate).is_file()
        for root in (model.parent, *model.parents)
        for modules in (root / "modules", root / "main" / "modules")
        for candidate in (
            *(modules / folder / name for folder in contentvec_folders for name in preferred_contentvec_names),
            *(modules / name for name in preferred_contentvec_names),
            *(modules / folder / name for folder in ("contentvec", "hubert") for name in fairseq_hubert_names),
            *(modules / name for name in fairseq_hubert_names),
        )
    ):
        missing.append("ContentVec/Fairseq HuBERT feature embedder (.onnx/.pt)")
    if not any(
        Path(candidate).is_file()
        for root in (model.parent, *model.parents)
        for modules in (root / "modules", root / "main" / "modules")
        for candidate in (
            *(modules / "rmvpe" / name for name in rmvpe_names),
            *(modules / name for name in rmvpe_names),
        )
    ):
        missing.append("RMVPE (.onnx)")
    searched_display = "; ".join(dict.fromkeys(searched))
    raise ValueError(
        "Feature embedder and RMVPE assets were not found above the selected model. "
        f"Missing: {', '.join(missing) or 'unknown asset'}. "
        "Import from a w-okada model_dir or choose an explicit feature embedder; "
        f"searched: {searched_display}"
    )


def input_signal_levels(samples: np.ndarray) -> tuple[float, float]:
    """Return RMS and peak levels for one normalized mono input frame."""

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(np.square(finite), dtype=np.float64)))
    peak = float(np.max(np.abs(finite)))
    return rms, peak


def rvc_volume_gain(samples: np.ndarray) -> tuple[float, float]:
    """Return w-okada's input RMS and output gain for one conversion crop.

    w-okada computes ``vol = sqrt(mean(crop**2))`` and scales the generated
    waveform by ``sqrt(vol)``.  Returning both values makes the parity rule
    explicit and lets the live status expose what was applied without forcing
    the audio thread to recalculate the level.
    """

    rms, _ = input_signal_levels(samples)
    return rms, float(np.sqrt(max(rms, 0.0)))


def is_silent_input(
    samples: np.ndarray,
    *,
    rms_threshold: float = SILENCE_RMS_THRESHOLD,
    peak_threshold: float = SILENCE_PEAK_THRESHOLD,
) -> bool:
    """Identify an idle microphone frame before it reaches neural inference.

    RMS is authoritative for a stationary floor, with a small activity-ratio
    escape hatch for quiet speech. Real interfaces can have a low average
    level with occasional isolated peaks; a peak by itself must not wake the
    neural decoder and create audible static on an otherwise idle route.
    Callers can disable this gate for calibration/warm-up measurements.
    """

    rms, peak = input_signal_levels(samples)
    if rms > rms_threshold:
        return False
    values = np.abs(np.asarray(samples, dtype=np.float32).reshape(-1))
    if values.size == 0 or peak <= 0.0:
        return True
    activity_floor = max(float(peak_threshold) * 2.0, rms_threshold * SILENCE_ACTIVITY_MULTIPLIER)
    activity_ratio = float(np.count_nonzero(values >= activity_floor)) / values.size
    return activity_ratio < SILENCE_ACTIVITY_RATIO


class LiveRvcProcessor:
    def __init__(self) -> None:
        self.generator: LoadedGenerator | None = None
        self.features: OnnxFeaturePipeline | None = None
        self.retrieval_index: FaissFeatureIndex | None = None
        self.model_path: str | None = None
        self.contentvec_path: str | None = None
        self.feature_backend = "contentvec-onnx"
        self.rmvpe_path: str | None = None
        self.pitch_shift = 0.0
        self.speaker_id = 0
        self.index_ratio = 0.0
        self.protect_ratio = 0.5
        # Match w-okada's RMVPEOnnxPitchExtractor default threshold.
        self.f0_threshold = 0.30
        self.streaming_preset = "balanced"
        self.warmup_ms = 0.0
        self.process_calls = 0
        self.last_process_ms = 0.0
        self.last_resample_ms = 0.0
        self.last_content_ms = 0.0
        self.last_pitch_ms = 0.0
        self.last_retrieval_ms = 0.0
        self.last_generator_ms = 0.0
        self.last_stitch_ms = 0.0
        self.last_sola_offset_frames = 0
        self.silence_suppressed_calls = 0
        self.last_input_rms = 0.0
        self.last_input_peak = 0.0
        self.max_input_rms = 0.0
        self.max_input_peak = 0.0
        self.last_input_volume = 0.0
        self.last_output_gain = 1.0
        self._silence_active = False
        self.feature_history: np.ndarray | None = None
        # w-okada keeps the post-inference feature tensor around and appends
        # zero feature frames for the next live hop.  Retrieval restores a
        # small prefix from that rolling buffer; using the current ContentVec
        # output there makes the first voiced frame sound different from the
        # reference pipeline.
        self.rvc_feature_buffer: np.ndarray | None = None
        self._pending_resample_ms = 0.0
        self._configure_stream("balanced")

    def _configure_stream(
        self,
        preset: object,
        *,
        chunk_frames: object | None = None,
        extra_frames: object | None = None,
        rvc_version: str | None = None,
    ) -> None:
        profile = get_stream_profile(
            preset,
            chunk_frames=chunk_frames,
            extra_frames=extra_frames,
            rvc_version=rvc_version,
        )
        self.streaming_preset = profile.name
        self.chunk_frames = profile.chunk_frames
        self.extra_frames = profile.extra_frames
        self.analysis_frames = profile.analysis_frames
        self.crossfade_frames = profile.crossfade_frames
        self.sola_search_frames = profile.sola_search_frames
        self.input_history = np.zeros(self.analysis_frames, dtype=np.float32)
        is_v2 = str(
            rvc_version or getattr(self.generator, "rvc_version", "")
        ).casefold() == "v2"
        self.feature_history = (
            np.zeros(
                int(
                    round(
                        self.analysis_frames
                        * FEATURE_SAMPLE_RATE
                        / LIVE_INPUT_SAMPLE_RATE
                    )
                ),
                dtype=np.float32,
            )
            if is_v2
            else None
        )
        self.rvc_feature_buffer = None
        self._pending_resample_ms = 0.0
        self.stitcher = SolaStitcher(
            self.chunk_frames,
            self.crossfade_frames,
            self.sola_search_frames,
        )

    def status(self) -> dict[str, Any]:
        loaded = self.generator
        generator_providers = (
            loaded.model.get_providers()
            if loaded is not None
            and loaded.backend == "onnx"
            and hasattr(loaded.model, "get_providers")
            else []
        )
        return {
            "state": "ready" if loaded is not None else "empty",
            "protocolVersion": PROTOCOL_VERSION,
            "modelPath": self.model_path,
            "contentvecPath": self.contentvec_path,
            "featurePath": self.contentvec_path,
            "featureBackend": self.feature_backend,
            "rmvpePath": self.rmvpe_path,
            "indexPath": self.retrieval_index.path if self.retrieval_index else None,
            "indexLoaded": self.retrieval_index is not None,
            "indexDimension": self.retrieval_index.dimension if self.retrieval_index else None,
            "indexVectorCount": self.retrieval_index.vector_count if self.retrieval_index else 0,
            "indexType": self.retrieval_index.index_type if self.retrieval_index else None,
            "indexNeighbors": 1 if self.retrieval_index else 0,
            "sampleRate": LIVE_INPUT_SAMPLE_RATE,
            "chunkFrames": self.chunk_frames,
            "chunkMilliseconds": self.chunk_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "extraFrames": self.extra_frames,
            "extraMilliseconds": self.extra_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "analysisFrames": self.analysis_frames,
            "analysisMilliseconds": self.analysis_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "crossfadeFrames": self.crossfade_frames,
            "crossfadeMilliseconds": self.crossfade_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "solaSearchFrames": self.sola_search_frames,
            "solaSearchMilliseconds": self.sola_search_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "silenceFrontFrames": self._silence_front_audio_frames(),
            "silenceFrontFeatureFrames": self._silence_front_feature_frames(),
            "generatorConvertFrames": self._generator_convert_length(),
            "streamPrimed": self.stitcher.primed,
            "rvcVersion": loaded.rvc_version if loaded else None,
            "targetSampleRate": loaded.target_sample_rate if loaded else None,
            "speakerCount": loaded.speaker_count if loaded else None,
            "precision": loaded.precision if loaded else None,
            "device": loaded.device if loaded else None,
            "backend": loaded.backend if loaded else None,
            "generatorProviders": generator_providers,
            "pitchShift": self.pitch_shift,
            "speakerId": self.speaker_id,
            "indexRatio": self.index_ratio,
            "protectRatio": self.protect_ratio,
            "f0Method": "RMVPE",
            "f0Threshold": self.f0_threshold,
            "streamingPreset": self.streaming_preset,
            "warmupMs": self.warmup_ms,
            "processCalls": self.process_calls,
            "lastProcessMs": self.last_process_ms,
            "lastResampleMs": self.last_resample_ms,
            "lastContentMs": self.last_content_ms,
            "lastPitchMs": self.last_pitch_ms,
            "lastRetrievalMs": self.last_retrieval_ms,
            "lastGeneratorMs": self.last_generator_ms,
            "lastStitchMs": self.last_stitch_ms,
            "lastSolaOffsetFrames": self.last_sola_offset_frames,
            "silenceSuppressedCalls": self.silence_suppressed_calls,
            "lastInputRms": self.last_input_rms,
            "lastInputPeak": self.last_input_peak,
            "maxInputRms": self.max_input_rms,
            "maxInputPeak": self.max_input_peak,
            "lastInputVolume": self.last_input_volume,
            "lastOutputGain": self.last_output_gain,
            "silenceGateRms": SILENCE_RMS_THRESHOLD,
            "silenceGatePeak": SILENCE_PEAK_THRESHOLD,
            "silenceGateMode": "rms+activity",
            "providers": self.features.providers if self.features else [],
        }

    def load(self, params: dict[str, Any]) -> dict[str, Any]:
        model_path = str(params.get("modelPath", "")).strip()
        if not model_path:
            raise ValueError("A modelPath is required.")

        package_defaults = model_package_defaults(model_path)

        def value_or_default(name: str, fallback: object) -> object:
            value = params.get(name)
            return fallback if value is None else value

        # Validate the inexpensive, user-editable settings before touching CUDA or loading a
        # large checkpoint. When a w-okada params.json is present, its values are the
        # compatibility defaults; explicit UI values still win over them.
        pitch_shift = validate_pitch_shift(
            value_or_default("pitchShift", package_defaults.get("pitchShift", 0.0))
        )
        explicit_index_ratio = params.get("indexRatio")
        index_ratio = validate_index_ratio(
            value_or_default("indexRatio", package_defaults.get("indexRatio", 0.0))
        )
        protect_ratio = validate_protect_ratio(
            value_or_default("protectRatio", package_defaults.get("protectRatio", 0.5))
        )
        f0_threshold = validate_f0_threshold(params.get("f0Threshold", 0.30))
        stream_profile = get_stream_profile(
            params.get("streamingPreset", "balanced"),
            chunk_frames=value_or_default(
                "chunkFrames", package_defaults.get("chunkFrames")
            ),
            extra_frames=value_or_default(
                "extraFrames", package_defaults.get("extraFrames")
            ),
        )
        try:
            speaker_id = int(params.get("speakerId", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("Speaker ID must be an integer.") from error

        contentvec_path = params.get("contentvecPath")
        rmvpe_path = params.get("rmvpePath")
        if not contentvec_path or not rmvpe_path:
            discovered_contentvec, discovered_rmvpe = discover_feature_models(
                model_path,
                package_defaults.get("embedder"),
            )
            contentvec_path = contentvec_path or discovered_contentvec
            rmvpe_path = rmvpe_path or discovered_rmvpe

        contentvec = Path(str(contentvec_path)).expanduser().resolve()
        rmvpe = Path(str(rmvpe_path)).expanduser().resolve()
        if not contentvec.is_file():
            raise ValueError(f"Feature embedder was not found: {contentvec}")
        if not rmvpe.is_file():
            raise ValueError(f"RMVPE model was not found: {rmvpe}")

        index_path = str(
            params.get("indexPath")
            or package_defaults.get("recommendedIndex")
            or ""
        ).strip()
        if index_ratio > 0.0 and not index_path:
            # A w-okada params.json can retain a retrieval ratio after its
            # optional .index file was removed or never shipped.  Treat that
            # metadata as a safe no-retrieval default; an explicit UI ratio is
            # still rejected so a user never thinks retrieval is active when
            # no index was loaded.
            if explicit_index_ratio is None and package_defaults.get("indexRatio") is not None:
                index_ratio = 0.0
            else:
                raise ValueError("An index ratio above zero requires a selected .index file.")

        loaded = (
            load_onnx_generator(model_path)
            if Path(model_path).suffix.lower() == ".onnx"
            else load_generator(model_path)
        )
        features = load_feature_pipeline(
            str(contentvec),
            str(rmvpe),
            feature_channels=loaded.feature_channels,
        )
        feature_backend = (
            "fairseq-hubert"
            if contentvec.suffix.casefold() in {".pt", ".pth"}
            else "contentvec-onnx"
        )
        retrieval_index = (
            FaissFeatureIndex.load(index_path, loaded.feature_channels)
            if index_path
            else None
        )
        if not 0 <= speaker_id < loaded.speaker_count:
            raise ValueError(
                f"Speaker ID {speaker_id} is outside the checkpoint range (0-{loaded.speaker_count - 1})."
            )

        # Build and warm the replacement in isolation. If feature-session creation, index
        # loading, CUDA warm-up, or stream-shape validation fails, the currently loaded voice
        # remains untouched instead of leaving a half-loaded worker behind.
        candidate = LiveRvcProcessor()
        candidate.generator = loaded
        candidate.features = features
        candidate.retrieval_index = retrieval_index
        candidate.model_path = loaded.model_path
        candidate.contentvec_path = str(contentvec)
        candidate.feature_backend = feature_backend
        candidate.rmvpe_path = str(rmvpe)
        candidate.pitch_shift = pitch_shift
        candidate.speaker_id = speaker_id
        candidate.index_ratio = index_ratio
        candidate.protect_ratio = protect_ratio
        candidate.f0_threshold = f0_threshold
        candidate._configure_stream(
            stream_profile.name,
            chunk_frames=stream_profile.chunk_frames,
            extra_frames=stream_profile.extra_frames,
            rvc_version=loaded.rvc_version,
        )
        candidate.reset_stream()

        started = perf_counter()
        candidate._convert_analysis_window(
            np.zeros(candidate.analysis_frames, dtype=np.float32)
        )
        # Warm the exact live request path as well as the analysis window. The
        # native engine feeds Chunk-sized blocks, and the first SOLA/process
        # call can otherwise pay one-time allocator or kernel setup costs
        # after the UI already reports the model as ready.
        candidate.process(
            np.zeros(candidate.chunk_frames, dtype=np.float32),
            record=False,
            suppress_silence=False,
        )
        candidate.warmup_ms = (perf_counter() - started) * 1_000.0
        candidate.reset_stream()
        self.__dict__.update(candidate.__dict__)
        return self.status()

    def unload(self) -> dict[str, Any]:
        self.generator = None
        self.features = None
        self.retrieval_index = None
        self.model_path = None
        self.contentvec_path = None
        self.feature_backend = "contentvec-onnx"
        self.rmvpe_path = None
        self.pitch_shift = 0.0
        self.speaker_id = 0
        self.index_ratio = 0.0
        self.protect_ratio = 0.5
        self.f0_threshold = 0.30
        self._configure_stream("balanced")
        self.reset_stream()
        self.warmup_ms = 0.0
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return self.status()

    def set_settings(self, params: dict[str, Any]) -> dict[str, Any]:
        pitch_shift = validate_pitch_shift(params.get("pitchShift", self.pitch_shift))
        speaker_id = int(params.get("speakerId", self.speaker_id))
        index_ratio = validate_index_ratio(params.get("indexRatio", self.index_ratio))
        protect_ratio = validate_protect_ratio(params.get("protectRatio", self.protect_ratio))
        f0_threshold = validate_f0_threshold(params.get("f0Threshold", self.f0_threshold))
        stream_profile = get_stream_profile(
            params.get("streamingPreset", self.streaming_preset),
            chunk_frames=params.get("chunkFrames"),
            extra_frames=params.get("extraFrames"),
            rvc_version=getattr(self.generator, "rvc_version", None) if self.generator else None,
        )
        if self.generator and not 0 <= speaker_id < self.generator.speaker_count:
            raise ValueError("The requested speaker ID is outside the checkpoint range.")
        if index_ratio > 0.0 and self.retrieval_index is None:
            raise ValueError("An index ratio above zero requires a loaded retrieval index.")
        changed = (
            pitch_shift != self.pitch_shift
            or speaker_id != self.speaker_id
            or index_ratio != self.index_ratio
            or protect_ratio != self.protect_ratio
            or f0_threshold != self.f0_threshold
            or stream_profile.name != self.streaming_preset
            or stream_profile.chunk_frames != self.chunk_frames
            or stream_profile.extra_frames != self.extra_frames
            or stream_profile.analysis_frames != self.analysis_frames
        )
        profile_changed = (
            stream_profile.name != self.streaming_preset
            or stream_profile.chunk_frames != self.chunk_frames
            or stream_profile.extra_frames != self.extra_frames
            or stream_profile.analysis_frames != self.analysis_frames
        )
        self.pitch_shift = pitch_shift
        self.speaker_id = speaker_id
        self.index_ratio = index_ratio
        self.protect_ratio = protect_ratio
        self.f0_threshold = f0_threshold
        if profile_changed:
            self._configure_stream(
                stream_profile.name,
                chunk_frames=stream_profile.chunk_frames,
                extra_frames=stream_profile.extra_frames,
                rvc_version=getattr(self.generator, "rvc_version", None) if self.generator else None,
            )
        if changed:
            self.reset_stream()
        return self.status()

    def calibrate(self) -> dict[str, Any]:
        if self.generator is None or self.features is None:
            raise ValueError("Load an RVC voice before running stream calibration.")

        previous = (
            self.streaming_preset,
            self.chunk_frames,
            self.extra_frames,
            self.analysis_frames,
        )
        measurements: list[dict[str, Any]] = []
        try:
            for preset_name in ("latency", "balanced", "quality"):
                profile = get_stream_profile(preset_name)
                self._configure_stream(
                    profile.name,
                    chunk_frames=profile.chunk_frames,
                    extra_frames=profile.extra_frames,
                    rvc_version=getattr(self.generator, "rvc_version", None) if self.generator else None,
                )
                self.reset_stream()
                # Switching Chunk/Extra can trigger one-time graph, allocator, or CUDA
                # setup costs. Discard that first request so the recommendation reflects
                # the steady-state deadline of the selected profile.
                self._convert_analysis_window(
                    np.zeros(self.analysis_frames, dtype=np.float32)
                )
                process_times: list[float] = []
                for _ in range(CALIBRATION_SAMPLES):
                    started = perf_counter()
                    self.process(
                        np.zeros(self.chunk_frames, dtype=np.float32),
                        record=False,
                        suppress_silence=False,
                    )
                    process_times.append((perf_counter() - started) * 1_000.0)
                ordered_times = sorted(process_times)
                p95_position = (len(ordered_times) - 1) * 0.95
                lower = int(p95_position)
                upper = min(lower + 1, len(ordered_times) - 1)
                p95_process_ms = ordered_times[lower] + (
                    ordered_times[upper] - ordered_times[lower]
                ) * (p95_position - lower)
                max_process_ms = max(ordered_times)
                deadline_ms = self.chunk_frames * 1_000.0 / LIVE_INPUT_SAMPLE_RATE
                measurements.append(
                    {
                        "preset": preset_name,
                        "chunkFrames": self.chunk_frames,
                        "extraFrames": self.extra_frames,
                        "analysisFrames": self.analysis_frames,
                        # Keep processMs as the p95 value for compatibility
                        # with the existing UI/report shape; maxProcessMs
                        # exposes the scheduler-spike guard separately.
                        "processMs": p95_process_ms,
                        "maxProcessMs": max_process_ms,
                        "sampleCount": len(process_times),
                        "deadlineMs": deadline_ms,
                        "headroomMs": deadline_ms - p95_process_ms,
                        "stable": (
                            p95_process_ms <= deadline_ms * 0.8
                            and max_process_ms <= deadline_ms
                        ),
                    }
                )
        finally:
            self._configure_stream(
                previous[0],
                chunk_frames=previous[1],
                extra_frames=previous[2],
                rvc_version=getattr(self.generator, "rvc_version", None) if self.generator else None,
            )
            self.reset_stream()

        stable = [measurement for measurement in measurements if measurement["stable"]]
        candidates = stable or sorted(
            measurements,
            key=lambda measurement: measurement["headroomMs"],
            reverse=True,
        )
        recommended = candidates[0]["preset"] if candidates else previous[0]
        return {
            "sampleRate": LIVE_INPUT_SAMPLE_RATE,
            "recommendedPreset": recommended,
            "restoredPreset": previous[0],
            "profiles": measurements,
            "message": (
                "The recommended preset has enough multi-sample inference headroom."
                if stable
                else "No preset met the headroom target; use the profile with the largest margin and re-test with real speech."
            ),
        }

    def reset_stream(self) -> None:
        self._reset_stream_buffers()
        self._silence_active = False
        self.process_calls = 0
        self.silence_suppressed_calls = 0
        self.last_input_rms = 0.0
        self.last_input_peak = 0.0
        self.last_input_volume = 0.0
        self.last_output_gain = 1.0
        self.last_process_ms = 0.0
        self.last_resample_ms = 0.0
        self.last_content_ms = 0.0
        self.last_pitch_ms = 0.0
        self.last_retrieval_ms = 0.0
        self.last_generator_ms = 0.0
        self.last_stitch_ms = 0.0
        self.last_sola_offset_frames = 0

    def _reset_stream_buffers(self) -> None:
        self.input_history.fill(0.0)
        if self.feature_history is not None:
            self.feature_history.fill(0.0)
        self.rvc_feature_buffer = None
        self._pending_resample_ms = 0.0
        self.stitcher.reset()

    def _feature_buffer_for_window(self) -> np.ndarray | None:
        """Build w-okada's pre-retrieval feature buffer for this hop.

        RVCr2 stores the previous post-interpolation feature tensor, appends
        one zero row for each new 10 ms input frame, then keeps the final
        conversion-window rows.  The retrieval path uses every other row of
        this buffer to reconstruct the front context.  Returning ``None``
        keeps the offline/default helper behavior unchanged when there is no
        live history yet.
        """

        if self.generator is None or self.generator.rvc_version != "v2":
            return None
        frame_count = max(1, self.analysis_frames * FEATURE_SAMPLE_RATE // LIVE_INPUT_SAMPLE_RATE // 160)
        new_frame_count = max(
            1,
            self.chunk_frames * FEATURE_SAMPLE_RATE // LIVE_INPUT_SAMPLE_RATE // 160,
        )
        channels = int(self.generator.feature_channels)
        if self.rvc_feature_buffer is None:
            return np.zeros((frame_count, channels), dtype=np.float32)
        previous = np.asarray(self.rvc_feature_buffer, dtype=np.float32).reshape(
            -1, channels
        )
        combined = np.concatenate(
            (previous, np.zeros((new_frame_count, channels), dtype=np.float32)),
            axis=0,
        )
        if combined.shape[0] < frame_count:
            combined = np.concatenate(
                (
                    np.zeros((frame_count - combined.shape[0], channels), dtype=np.float32),
                    combined,
                ),
                axis=0,
            )
        return np.ascontiguousarray(combined[-frame_count:], dtype=np.float32)

    def _uses_v2_feature_history(self) -> bool:
        return (
            self.feature_history is not None
            and str(getattr(self.generator, "rvc_version", "")).casefold() == "v2"
        )

    def _append_v2_feature_history(self, samples_48k: np.ndarray) -> float:
        """Match w-okada's per-hop 48 kHz → 16 kHz history boundary.

        RVCr2 receives a live block, resamples that block with resampy's
        ``kaiser_fast`` filter, and only then appends it to the retained
        feature-rate audio buffer. Resampling the whole retained window would
        change the filter edge conditions on every hop and can make the same
        model sound different from w-okada at chunk boundaries.
        """

        if self.feature_history is None:
            return 0.0
        started = perf_counter()
        converted = resample_kaiser_fast(
            samples_48k,
            LIVE_INPUT_SAMPLE_RATE,
            FEATURE_SAMPLE_RATE,
        )
        expected = int(
            round(self.chunk_frames * FEATURE_SAMPLE_RATE / LIVE_INPUT_SAMPLE_RATE)
        )
        if converted.shape[0] > expected:
            converted = converted[:expected]
        elif converted.shape[0] < expected:
            converted = np.pad(converted, (expected - converted.shape[0], 0))
        if expected >= self.feature_history.shape[0]:
            self.feature_history[:] = converted[-self.feature_history.shape[0] :]
        else:
            self.feature_history[:-expected] = self.feature_history[expected:]
            self.feature_history[-expected:] = converted
        return (perf_counter() - started) * 1_000.0

    def _silence_front_audio_frames(self) -> int:
        """Return the rounded retained front context in live 48 kHz frames.

        RVCr2 rounds ``convertSize`` to a complete 160-sample feature hop.
        The retained live window is the equivalent 48 kHz duration, so this
        value can differ from the user's Extra value by at most one rounded
        feature step.  The unrounded value used by w-okada's front trim is
        exposed by :meth:`_processing_extra_frames` below.
        """

        return max(0, self.analysis_frames - self.stitcher.candidate_frames)

    def _processing_extra_frames(self) -> int:
        """Return w-okada's Extra value after the 48 kHz → 16 kHz conversion."""

        return max(
            0,
            (self.extra_frames * FEATURE_SAMPLE_RATE) // LIVE_INPUT_SAMPLE_RATE,
        )

    def _silence_front_feature_frames(self) -> int:
        # Pipeline.py uses ``floor(silence_front * 16000) // 360``.  Keeping
        # the user's Extra value here (rather than the rounded retained window)
        # preserves the exact retrieval-front boundary.
        return self._processing_extra_frames() // 360

    def _generator_convert_length(self) -> int:
        """Return the output length requested from the RVC generator.

        w-okada's RVCr2 path requests the full target-rate output for its
        rounded 16 kHz convert window, then the host keeps the final SOLA
        candidate. Asking the generator for only that candidate changes the
        decoder's tail alignment and is one source of model-to-host quality
        drift.
        """

        candidate = self.stitcher.candidate_frames
        if self.generator is None or self.generator.rvc_version != "v2":
            target_rate = self.generator.target_sample_rate if self.generator else LIVE_INPUT_SAMPLE_RATE
            return int(round(candidate * target_rate / LIVE_INPUT_SAMPLE_RATE))

        processing_chunk = (self.chunk_frames * FEATURE_SAMPLE_RATE) // LIVE_INPUT_SAMPLE_RATE
        processing_crossfade = (self.crossfade_frames * FEATURE_SAMPLE_RATE) // LIVE_INPUT_SAMPLE_RATE
        processing_search = (self.sola_search_frames * FEATURE_SAMPLE_RATE) // LIVE_INPUT_SAMPLE_RATE
        processing_extra = self._processing_extra_frames()
        convert_size = int(
            np.ceil(
                (processing_chunk + processing_crossfade + processing_search + processing_extra)
                / 160.0
            )
            * 160
        )
        return int(
            ((convert_size - processing_extra) / FEATURE_SAMPLE_RATE)
            * self.generator.target_sample_rate
        )

    def _convert_analysis_window(
        self, samples_48k: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self.generator is None or self.features is None:
            raise ValueError("No live RVC model is loaded.")
        samples = np.asarray(samples_48k, dtype=np.float32).reshape(-1)
        if samples.shape[0] != self.analysis_frames:
            raise ValueError(
                f"The RVC analysis window requires exactly {self.analysis_frames} samples."
            )
        # Match RVC.generate_input's crop: the current input hop plus the
        # overlap, excluding the overlap itself.  This is the volume used by
        # w-okada to keep the generated output tied to the source loudness.
        use_v2_history = self._uses_v2_feature_history()
        if use_v2_history:
            waveform = self.feature_history
            assert waveform is not None
            feature_chunk = int(
                round(self.chunk_frames * FEATURE_SAMPLE_RATE / LIVE_INPUT_SAMPLE_RATE)
            )
            feature_crossfade = int(
                round(
                    self.crossfade_frames
                    * FEATURE_SAMPLE_RATE
                    / LIVE_INPUT_SAMPLE_RATE
                )
            )
            crop_start = max(0, waveform.shape[0] - (feature_chunk + feature_crossfade))
            crop_end = max(crop_start, waveform.shape[0] - feature_crossfade)
            input_volume, output_gain = rvc_volume_gain(waveform[crop_start:crop_end])
        else:
            crop_start = max(
                0,
                self.analysis_frames - (self.chunk_frames + self.crossfade_frames),
            )
            crop_end = max(crop_start, self.analysis_frames - self.crossfade_frames)
            input_volume, output_gain = rvc_volume_gain(samples[crop_start:crop_end])
        self.last_input_volume = input_volume
        self.last_output_gain = output_gain
        if use_v2_history:
            # The feature history was resampled at the live-hop boundary in
            # ``process``. Consume that timing here so status remains per-hop.
            waveform = waveform.astype(np.float32, copy=False)
            resample_ms = self._pending_resample_ms
            self._pending_resample_ms = 0.0
        else:
            stage_started = perf_counter()
            waveform = resample_kaiser_fast(
                samples, LIVE_INPUT_SAMPLE_RATE, FEATURE_SAMPLE_RATE
            )
            # w-okada's default RVCQuality=0 path does not add a reflected
            # feature pad; its zero front context is supplied by the live
            # conversion window itself. Keep the feature waveform at the
            # resampler's exact length so ContentVec/RMVPE see the same edge
            # geometry.
            waveform = waveform.astype(np.float32, copy=False)
            resample_ms = (perf_counter() - stage_started) * 1_000.0

        stage_started = perf_counter()
        content = self.features.extract_content(waveform)
        content_ms = (perf_counter() - stage_started) * 1_000.0

        stage_started = perf_counter()
        # w-okada leaves the zero-padded front conversion context in the
        # ContentVec window, but asks RMVPE to analyze only the real tail and
        # restores zero F0 frames at the front of the pitch buffer.  Keep that
        # boundary explicit so pitch decisions do not depend on synthetic
        # padding.
        pitch_front_16k = self._processing_extra_frames()
        pitchf = self.features.extract_pitch(
            waveform,
            self.f0_threshold,
            silence_front_samples=pitch_front_16k,
            output_frames=max(1, waveform.shape[0] // 160),
        )
        pitch_ms = (perf_counter() - stage_started) * 1_000.0

        silence_front_frames = self._silence_front_feature_frames()
        target_convert_length = self._generator_convert_length()
        front_features = self._feature_buffer_for_window()
        feature_buffer_out: list[np.ndarray] = []

        output, _, _, generator_ms, retrieval_ms = _run_generator(
            self.generator,
            content,
            pitchf,
            speaker_id=self.speaker_id,
            pitch_shift=self.pitch_shift,
            retrieval_index=self.retrieval_index,
            index_ratio=self.index_ratio,
            protect_ratio=self.protect_ratio,
            silence_front_frames=silence_front_frames,
            convert_length=target_convert_length,
            front_features=front_features,
            feature_buffer_out=feature_buffer_out,
        )
        if feature_buffer_out:
            self.rvc_feature_buffer = feature_buffer_out[0]
        # RVC checkpoints may target 32, 40, or 48 kHz. The native live path
        # is intentionally fixed at 48 kHz, so convert the generator output
        # back to the device rate before SOLA stitching and frame alignment.
        if self.generator.target_sample_rate != LIVE_INPUT_SAMPLE_RATE:
            stage_started = perf_counter()
            output = resample_kaiser_fast(
                np.asarray(output, dtype=np.float32),
                self.generator.target_sample_rate,
                LIVE_INPUT_SAMPLE_RATE,
            )
            resample_ms += (perf_counter() - stage_started) * 1_000.0
        if output.shape[0] > self.analysis_frames:
            output = output[-self.analysis_frames :]
        elif output.shape[0] < self.analysis_frames:
            # PyTorch realtime inference returns the generated tail when a
            # convert length is supplied. Keep that tail at the end of the
            # retained window so the SOLA candidate remains aligned with the
            # current hop; padding on the right would shift it into silence.
            output = np.pad(output, (self.analysis_frames - output.shape[0], 0))
        # w-okada applies sqrt(vol) after target-rate generation and before
        # overlap stitching.  Apply the same normalization after any target
        # sample-rate conversion so the live route has matching loudness.
        output = np.ascontiguousarray(output * np.float32(output_gain), dtype=np.float32)
        return output, {
            "resample": resample_ms,
            "content": content_ms,
            "pitch": pitch_ms,
            "retrieval": retrieval_ms,
            "generator": generator_ms,
        }

    def process(
        self,
        samples_48k: np.ndarray,
        *,
        record: bool = True,
        suppress_silence: bool = True,
    ) -> np.ndarray:
        if self.generator is None or self.features is None:
            raise ValueError("No live RVC model is loaded.")
        samples = np.asarray(samples_48k, dtype=np.float32).reshape(-1)
        if samples.shape[0] != self.chunk_frames:
            raise ValueError(
                f"Live audio frames must contain exactly {self.chunk_frames} float32 samples."
            )
        started = perf_counter()
        self.last_input_rms, self.last_input_peak = input_signal_levels(samples)
        self.max_input_rms = max(self.max_input_rms, self.last_input_rms)
        self.max_input_peak = max(self.max_input_peak, self.last_input_peak)
        if suppress_silence and is_silent_input(samples):
            # Do not let stale SOLA tails or a model's decoder bias leak into an
            # idle output. Resetting only the stream buffers keeps telemetry and
            # the worker session intact while making the next voiced frame start
            # from a clean analysis window.
            if not self._silence_active:
                self._reset_stream_buffers()
            self._silence_active = True
            output = np.zeros(self.chunk_frames, dtype=np.float32)
            if record:
                self.process_calls += 1
                self.silence_suppressed_calls += 1
                self.last_process_ms = (perf_counter() - started) * 1_000.0
                self.last_resample_ms = 0.0
                self.last_content_ms = 0.0
                self.last_pitch_ms = 0.0
                self.last_retrieval_ms = 0.0
                self.last_generator_ms = 0.0
                self.last_stitch_ms = 0.0
                self.last_sola_offset_frames = 0
            return output
        if self._silence_active:
            self._reset_stream_buffers()
            self._silence_active = False
        if self._uses_v2_feature_history():
            self._pending_resample_ms = self._append_v2_feature_history(samples)
        else:
            self.input_history[:-self.chunk_frames] = self.input_history[self.chunk_frames:]
            self.input_history[-self.chunk_frames:] = samples
        converted, timings = self._convert_analysis_window(self.input_history)

        stitch_started = perf_counter()
        candidate = converted[-self.stitcher.candidate_frames :]
        stitched = self.stitcher.process(candidate)
        stitch_ms = (perf_counter() - stitch_started) * 1_000.0
        output = np.ascontiguousarray(stitched.audio, dtype=np.float32)
        if record:
            self.process_calls += 1
            self.last_process_ms = (perf_counter() - started) * 1_000.0
            self.last_resample_ms = timings["resample"]
            self.last_content_ms = timings["content"]
            self.last_pitch_ms = timings["pitch"]
            self.last_retrieval_ms = timings["retrieval"]
            self.last_generator_ms = timings["generator"]
            self.last_stitch_ms = stitch_ms
            self.last_sola_offset_frames = stitched.offset_frames
        return output


def _json_payload(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _control(processor: LiveRvcProcessor, payload: bytes) -> tuple[dict[str, Any], bool]:
    request = json.loads(payload.decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("A worker control request must be a JSON object.")
    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("Worker control params must be a JSON object.")
    if method == "handshake":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "transport": "framed-stdio",
            "audioEncoding": "float32-le-mono",
            "sampleRate": LIVE_INPUT_SAMPLE_RATE,
            "chunkFrames": LIVE_CHUNK_FRAMES,
            "analysisFrames": LIVE_ANALYSIS_FRAMES,
            "crossfadeFrames": LIVE_CROSSFADE_FRAMES,
            "solaSearchFrames": LIVE_SOLA_SEARCH_FRAMES,
        }, False
    if method == "load_model":
        return processor.load(params), False
    if method == "status":
        return processor.status(), False
    if method == "set_settings":
        return processor.set_settings(params), False
    if method == "calibrate":
        return processor.calibrate(), False
    if method == "unload":
        return processor.unload(), False
    if method == "shutdown":
        return processor.status(), True
    raise ValueError(f"Unsupported live worker method: {method!r}.")


def serve_worker(input_stream: BinaryIO, output_stream: BinaryIO) -> int:
    processor = LiveRvcProcessor()
    while True:
        try:
            frame = read_frame(input_stream)
        except EOFError:
            return 0
        try:
            should_stop = False
            if frame.kind == JSON_REQUEST:
                result, should_stop = _control(processor, frame.payload)
                response = Frame(JSON_RESPONSE, frame.request_id, _json_payload(result))
            elif frame.kind == AUDIO_REQUEST:
                samples = np.frombuffer(frame.payload, dtype="<f4").copy()
                converted = processor.process(samples)
                response = Frame(AUDIO_RESPONSE, frame.request_id, converted.astype("<f4").tobytes())
            elif frame.kind == SHUTDOWN:
                response = Frame(JSON_RESPONSE, frame.request_id, _json_payload(processor.status()))
                should_stop = True
            else:
                raise ValueError(f"Unsupported live worker frame kind: {frame.kind}.")
        except Exception as error:
            response = Frame(
                ERROR_RESPONSE,
                frame.request_id,
                _json_payload({"error": str(error), "errorType": type(error).__name__}),
            )
            should_stop = False
        write_frame(output_stream, response)
        if should_stop:
            return 0


def run_worker() -> int:
    seed_value = os.environ.get("VC_NEXT_TORCH_SEED", "").strip()
    if seed_value:
        try:
            seed = int(seed_value)
        except ValueError as error:
            raise ValueError("VC_NEXT_TORCH_SEED must be an integer.") from error
        import torch

        torch.manual_seed(seed)
    return serve_worker(sys.stdin.buffer, sys.stdout.buffer)

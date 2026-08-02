from __future__ import annotations

import json
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
from .rvc_compat.loader import LoadedGenerator, load_generator
from .rvc_compat.offline import OnnxFeaturePipeline, _run_generator
from .rvc_compat.retrieval import (
    FaissFeatureIndex,
    validate_index_ratio,
    validate_protect_ratio,
)
from .streaming import SolaStitcher
from .stream_config import get_stream_profile, validate_f0_threshold, validate_pitch_shift


LIVE_INPUT_SAMPLE_RATE = 48_000
LIVE_CHUNK_FRAMES = 9_600
LIVE_ANALYSIS_FRAMES = 24_000
LIVE_CROSSFADE_FRAMES = 1_920
LIVE_SOLA_SEARCH_FRAMES = 576
FEATURE_SAMPLE_RATE = 16_000
FEATURE_CONTEXT_FRAMES = 160


def discover_feature_models(model_path: str) -> tuple[str, str]:
    model = Path(model_path).expanduser().resolve()
    for root in (model.parent, *model.parents):
        for modules in (root / "modules", root / "main" / "modules"):
            contentvec = modules / "contentvec" / "contentvec-f.onnx"
            rmvpe = modules / "rmvpe" / "rmvpe_20231006.onnx"
            if contentvec.is_file() and rmvpe.is_file():
                return str(contentvec), str(rmvpe)
    raise ValueError(
        "ContentVec and RMVPE assets were not found above the selected model. "
        "Import from a w-okada model_dir or configure the engine assets explicitly."
    )


class LiveRvcProcessor:
    def __init__(self) -> None:
        self.generator: LoadedGenerator | None = None
        self.features: OnnxFeaturePipeline | None = None
        self.retrieval_index: FaissFeatureIndex | None = None
        self.model_path: str | None = None
        self.contentvec_path: str | None = None
        self.rmvpe_path: str | None = None
        self.pitch_shift = 0.0
        self.speaker_id = 0
        self.index_ratio = 0.0
        self.protect_ratio = 0.5
        self.f0_threshold = 0.03
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
        self._configure_stream("balanced")

    def _configure_stream(
        self,
        preset: object,
        *,
        chunk_frames: object | None = None,
        extra_frames: object | None = None,
    ) -> None:
        profile = get_stream_profile(
            preset,
            chunk_frames=chunk_frames,
            extra_frames=extra_frames,
        )
        self.streaming_preset = profile.name
        self.chunk_frames = profile.chunk_frames
        self.analysis_frames = profile.analysis_frames
        self.crossfade_frames = profile.crossfade_frames
        self.sola_search_frames = profile.sola_search_frames
        self.input_history = np.zeros(self.analysis_frames, dtype=np.float32)
        self.stitcher = SolaStitcher(
            self.chunk_frames,
            self.crossfade_frames,
            self.sola_search_frames,
        )

    def status(self) -> dict[str, Any]:
        loaded = self.generator
        return {
            "state": "ready" if loaded is not None else "empty",
            "protocolVersion": PROTOCOL_VERSION,
            "modelPath": self.model_path,
            "contentvecPath": self.contentvec_path,
            "rmvpePath": self.rmvpe_path,
            "indexPath": self.retrieval_index.path if self.retrieval_index else None,
            "indexLoaded": self.retrieval_index is not None,
            "indexDimension": self.retrieval_index.dimension if self.retrieval_index else None,
            "indexVectorCount": self.retrieval_index.vector_count if self.retrieval_index else 0,
            "indexType": self.retrieval_index.index_type if self.retrieval_index else None,
            "sampleRate": LIVE_INPUT_SAMPLE_RATE,
            "chunkFrames": self.chunk_frames,
            "chunkMilliseconds": self.chunk_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "analysisFrames": self.analysis_frames,
            "extraFrames": self.analysis_frames,
            "analysisMilliseconds": self.analysis_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "crossfadeFrames": self.crossfade_frames,
            "crossfadeMilliseconds": self.crossfade_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "solaSearchFrames": self.sola_search_frames,
            "solaSearchMilliseconds": self.sola_search_frames * 1_000 / LIVE_INPUT_SAMPLE_RATE,
            "streamPrimed": self.stitcher.primed,
            "rvcVersion": loaded.rvc_version if loaded else None,
            "targetSampleRate": loaded.target_sample_rate if loaded else None,
            "speakerCount": loaded.speaker_count if loaded else None,
            "precision": loaded.precision if loaded else None,
            "device": loaded.device if loaded else None,
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
            "providers": self.features.providers if self.features else [],
        }

    def load(self, params: dict[str, Any]) -> dict[str, Any]:
        model_path = str(params.get("modelPath", "")).strip()
        if not model_path:
            raise ValueError("A modelPath is required.")
        contentvec_path = params.get("contentvecPath")
        rmvpe_path = params.get("rmvpePath")
        if not contentvec_path or not rmvpe_path:
            discovered_contentvec, discovered_rmvpe = discover_feature_models(model_path)
            contentvec_path = contentvec_path or discovered_contentvec
            rmvpe_path = rmvpe_path or discovered_rmvpe

        loaded = load_generator(model_path)
        features = OnnxFeaturePipeline(str(contentvec_path), str(rmvpe_path))
        index_path = str(params.get("indexPath") or "").strip()
        retrieval_index = (
            FaissFeatureIndex.load(index_path, loaded.feature_channels)
            if index_path
            else None
        )
        index_ratio = validate_index_ratio(params.get("indexRatio", 0.0))
        protect_ratio = validate_protect_ratio(params.get("protectRatio", 0.5))
        f0_threshold = validate_f0_threshold(params.get("f0Threshold", 0.03))
        stream_profile = get_stream_profile(
            params.get("streamingPreset", "balanced"),
            chunk_frames=params.get("chunkFrames"),
            extra_frames=params.get("extraFrames"),
        )
        speaker_id = int(params.get("speakerId", 0))
        if index_ratio > 0.0 and retrieval_index is None:
            raise ValueError("An index ratio above zero requires a selected .index file.")
        if not 0 <= speaker_id < loaded.speaker_count:
            raise ValueError("The requested speaker ID is outside the checkpoint range.")

        self.generator = loaded
        self.features = features
        self.retrieval_index = retrieval_index
        self.model_path = loaded.model_path
        self.contentvec_path = str(Path(contentvec_path).resolve())
        self.rmvpe_path = str(Path(rmvpe_path).resolve())
        self.pitch_shift = validate_pitch_shift(params.get("pitchShift", 0.0))
        self.speaker_id = speaker_id
        self.index_ratio = index_ratio
        self.protect_ratio = protect_ratio
        self.f0_threshold = f0_threshold
        self._configure_stream(
            stream_profile.name,
            chunk_frames=stream_profile.chunk_frames,
            extra_frames=stream_profile.analysis_frames,
        )
        self.reset_stream()

        started = perf_counter()
        self._convert_analysis_window(
            np.zeros(self.analysis_frames, dtype=np.float32)
        )
        self.warmup_ms = (perf_counter() - started) * 1_000.0
        self.reset_stream()
        return self.status()

    def unload(self) -> dict[str, Any]:
        self.generator = None
        self.features = None
        self.retrieval_index = None
        self.model_path = None
        self.contentvec_path = None
        self.rmvpe_path = None
        self.pitch_shift = 0.0
        self.speaker_id = 0
        self.index_ratio = 0.0
        self.protect_ratio = 0.5
        self.f0_threshold = 0.03
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
            or stream_profile.analysis_frames != self.analysis_frames
        )
        profile_changed = (
            stream_profile.name != self.streaming_preset
            or stream_profile.chunk_frames != self.chunk_frames
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
                extra_frames=stream_profile.analysis_frames,
            )
        if changed:
            self.reset_stream()
        return self.status()

    def reset_stream(self) -> None:
        self.input_history.fill(0.0)
        self.stitcher.reset()
        self.process_calls = 0
        self.last_process_ms = 0.0
        self.last_resample_ms = 0.0
        self.last_content_ms = 0.0
        self.last_pitch_ms = 0.0
        self.last_retrieval_ms = 0.0
        self.last_generator_ms = 0.0
        self.last_stitch_ms = 0.0
        self.last_sola_offset_frames = 0

    def _convert_analysis_window(
        self, samples_48k: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        import torch
        import torchaudio.functional as audio_functional

        if self.generator is None or self.features is None:
            raise ValueError("No live RVC model is loaded.")
        samples = np.asarray(samples_48k, dtype=np.float32).reshape(-1)
        if samples.shape[0] != self.analysis_frames:
            raise ValueError(
                f"The RVC analysis window requires exactly {self.analysis_frames} samples."
            )
        stage_started = perf_counter()
        waveform = audio_functional.resample(
            torch.from_numpy(samples), LIVE_INPUT_SAMPLE_RATE, FEATURE_SAMPLE_RATE
        ).numpy()
        waveform = np.pad(
            waveform,
            (FEATURE_CONTEXT_FRAMES, FEATURE_CONTEXT_FRAMES),
            mode="reflect",
        ).astype(np.float32, copy=False)
        resample_ms = (perf_counter() - stage_started) * 1_000.0

        stage_started = perf_counter()
        content = self.features.extract_content(waveform)
        content_ms = (perf_counter() - stage_started) * 1_000.0

        stage_started = perf_counter()
        pitchf = self.features.extract_pitch(waveform, self.f0_threshold)
        pitch_ms = (perf_counter() - stage_started) * 1_000.0

        output, _, _, generator_ms, retrieval_ms = _run_generator(
            self.generator,
            content,
            pitchf,
            speaker_id=self.speaker_id,
            pitch_shift=self.pitch_shift,
            retrieval_index=self.retrieval_index,
            index_ratio=self.index_ratio,
            protect_ratio=self.protect_ratio,
        )
        # RVC checkpoints may target 32, 40, or 48 kHz. The native live path
        # is intentionally fixed at 48 kHz, so convert the generator output
        # back to the device rate before SOLA stitching and frame alignment.
        if self.generator.target_sample_rate != LIVE_INPUT_SAMPLE_RATE:
            stage_started = perf_counter()
            output = audio_functional.resample(
                torch.from_numpy(np.asarray(output, dtype=np.float32)),
                self.generator.target_sample_rate,
                LIVE_INPUT_SAMPLE_RATE,
            ).numpy()
            resample_ms += (perf_counter() - stage_started) * 1_000.0
        if output.shape[0] > self.analysis_frames:
            difference = output.shape[0] - self.analysis_frames
            left = difference // 2
            output = output[left : left + self.analysis_frames]
        elif output.shape[0] < self.analysis_frames:
            output = np.pad(output, (0, self.analysis_frames - output.shape[0]))
        output = np.ascontiguousarray(output, dtype=np.float32)
        return output, {
            "resample": resample_ms,
            "content": content_ms,
            "pitch": pitch_ms,
            "retrieval": retrieval_ms,
            "generator": generator_ms,
        }

    def process(self, samples_48k: np.ndarray, *, record: bool = True) -> np.ndarray:
        if self.generator is None or self.features is None:
            raise ValueError("No live RVC model is loaded.")
        samples = np.asarray(samples_48k, dtype=np.float32).reshape(-1)
        if samples.shape[0] != self.chunk_frames:
            raise ValueError(
                f"Live audio frames must contain exactly {self.chunk_frames} float32 samples."
            )
        started = perf_counter()
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
    return serve_worker(sys.stdin.buffer, sys.stdout.buffer)

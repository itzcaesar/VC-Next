from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class StreamProfile:
    name: str
    chunk_frames: int
    analysis_frames: int
    crossfade_frames: int
    sola_search_frames: int


STREAM_PROFILES = {
    # w-okada's VoiceChangerV2 defaults to a 4096-sample SOLA overlap. Keep
    # that overlap for normal RVC compatibility profiles; custom very small
    # hops are reduced below so the overlap never consumes the whole hop.
    "quality": StreamProfile("quality", 12_000, 28_800, 4_096, 720),
    "balanced": StreamProfile("balanced", 9_600, 24_000, 4_096, 576),
    "latency": StreamProfile("latency", 7_680, 19_200, 4_096, 480),
}


MIN_STREAM_FRAMES = 480
MAX_STREAM_FRAMES = 480_000
# w-okada's RVCSettings uses 4,096 samples of extra conversion context.  Keep
# this as the minimum front context even when a caller only specifies Chunk;
# the public Extra/context value can request a larger retained window.
DEFAULT_EXTRA_CONTEXT_FRAMES = 4_096


def _validate_stream_frames(value: object, label: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a whole number of samples.") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be a whole number of samples.")
    frames = int(numeric)
    if not MIN_STREAM_FRAMES <= frames <= MAX_STREAM_FRAMES:
        raise ValueError(
            f"{label} must be between {MIN_STREAM_FRAMES} and {MAX_STREAM_FRAMES} samples."
        )
    return frames


def get_stream_profile(
    value: object,
    *,
    chunk_frames: object | None = None,
    extra_frames: object | None = None,
    rvc_version: str | None = None,
) -> StreamProfile:
    name = str(value or "balanced").strip().lower()
    try:
        profile = STREAM_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported streaming preset {name!r}; expected quality, balanced, or latency."
        ) from error

    if chunk_frames is None and extra_frames is None:
        return profile

    chunk = _validate_stream_frames(
        profile.chunk_frames if chunk_frames is None else chunk_frames,
        "Chunk",
    )
    requested_extra = _validate_stream_frames(
        profile.analysis_frames if extra_frames is None else extra_frames,
        "Extra/context",
    )
    # Keep overlap and SOLA search proportional to the selected hop. The
    # analysis window is never allowed to be shorter than the complete stitch
    # candidate, which prevents invalid boundaries and click-producing output.
    crossfade = min(profile.crossfade_frames, max(2, chunk // 2), chunk - 1)
    search = min(profile.sola_search_frames, max(0, chunk // 8))
    # RVC v1 works in the live 48 kHz domain and rounds its input window to a
    # 128-sample generator boundary. RVC v2 (RVCr2 in w-okada) first resamples
    # the live hop to 16 kHz, rounds ``convertSize`` to a 160-sample HuBERT
    # hop, and converts that window back to the device rate. Keeping the two
    # geometries explicit avoids a small but audible feature-window shift on
    # v2 voices while preserving the historical shape for unknown/unloaded
    # models.
    if str(rvc_version or "").casefold() == "v2":
        processing_chunk = (chunk * 16_000) // 48_000
        processing_crossfade = (crossfade * 16_000) // 48_000
        processing_search = (search * 16_000) // 48_000
        processing_extra = (DEFAULT_EXTRA_CONTEXT_FRAMES * 16_000) // 48_000
        convert_size = (
            processing_chunk
            + processing_crossfade
            + processing_search
            + processing_extra
        )
        convert_size = int(math.ceil(convert_size / 160.0) * 160)
        v2_analysis = int(math.ceil(convert_size * 48_000 / 16_000))
        analysis = max(requested_extra, v2_analysis)
        analysis = int(math.ceil(analysis / 480.0) * 480)
    else:
        # RVC's live generator truncates feature/output hops on 128-sample
        # boundaries. Round the retained window up before it reaches the
        # model so custom Chunk/Extra values follow the same geometry as
        # w-okada's v1-compatible path.
        analysis = max(
            requested_extra,
            chunk + crossfade + search + DEFAULT_EXTRA_CONTEXT_FRAMES,
        )
        analysis = int(math.ceil(analysis / 128.0) * 128)
    return replace(
        profile,
        chunk_frames=chunk,
        analysis_frames=analysis,
        crossfade_frames=crossfade,
        sola_search_frames=search,
    )


def validate_pitch_shift(value: object) -> float:
    pitch = float(value)
    if not math.isfinite(pitch) or not -50.0 <= pitch <= 50.0:
        raise ValueError("Pitch shift must be between -50 and +50 semitones.")
    return pitch


def validate_f0_threshold(value: object) -> float:
    threshold = float(value)
    # RMVPE's ONNX input is a periodicity threshold.  w-okada's
    # RMVPEOnnxPitchExtractor uses 0.30 by default; keep that value available
    # instead of constraining the compatibility path to the older 0.01–0.20
    # UI range.
    if not math.isfinite(threshold) or not 0.01 <= threshold <= 0.99:
        raise ValueError("The RMVPE threshold must be between 0.01 and 0.99.")
    return threshold

from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class StreamProfile:
    name: str
    chunk_frames: int
    extra_frames: int
    analysis_frames: int
    crossfade_frames: int
    sola_search_frames: int


STREAM_PROFILES = {
    # The public Extra value is w-okada's ``extraConvertSize`` at the 48 kHz
    # device boundary.  ``analysis_frames`` is derived below from Chunk,
    # Extra, overlap and SOLA search; it is not a second user setting.
    "quality": StreamProfile("quality", 12_000, 28_800, 45_696, 4_096, 720),
    "balanced": StreamProfile("balanced", 9_600, 24_000, 38_272, 4_096, 576),
    "latency": StreamProfile("latency", 7_680, 19_200, 31_488, 4_096, 480),
}


MIN_STREAM_FRAMES = 480
MAX_STREAM_FRAMES = 480_000
# w-okada's RVCSettings uses 4,096 samples of extra conversion context. Keep
# it available as the documented compatibility default and as a useful option
# for callers that want the smallest front context.
DEFAULT_EXTRA_CONTEXT_FRAMES = 4_096


def _analysis_frames(
    *,
    chunk: int,
    extra: int,
    crossfade: int,
    search: int,
    rvc_version: str | None,
) -> int:
    """Derive the retained live window from w-okada's convertSize formula."""

    version = str(rvc_version or "").casefold()
    if version == "v2":
        processing_chunk = (chunk * 16_000) // 48_000
        processing_crossfade = (crossfade * 16_000) // 48_000
        processing_search = (search * 16_000) // 48_000
        processing_extra = (extra * 16_000) // 48_000
        convert_size = int(
            math.ceil(
                (processing_chunk + processing_crossfade + processing_search + processing_extra)
                / 160.0
            )
            * 160
        )
        # The native live history is maintained at 48 kHz.  RVCr2's 160-sample
        # feature hop therefore maps back to an exact 480-device-frame step.
        return max(chunk + crossfade + search, convert_size * 48_000 // 16_000)

    # RVC v1 runs in the device/model sample-rate domain and uses the model's
    # 128-sample generator boundary for the retained input window.
    return max(
        chunk + crossfade + search,
        int(
            math.ceil(
                (chunk + crossfade + search + extra) / 128.0
            )
            * 128
        ),
    )


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

    chunk = _validate_stream_frames(
        profile.chunk_frames if chunk_frames is None else chunk_frames,
        "Chunk",
    )
    requested_extra = _validate_stream_frames(
        profile.extra_frames if extra_frames is None else extra_frames,
        "Extra/context",
    )
    # Keep overlap and SOLA search proportional to the selected hop. The
    # The analysis window is never allowed to be shorter than the complete
    # stitch candidate, which prevents invalid boundaries and click-producing
    # output.
    crossfade = min(profile.crossfade_frames, max(2, chunk // 2), chunk - 1)
    search = min(profile.sola_search_frames, max(0, chunk // 8))
    analysis = _analysis_frames(
        chunk=chunk,
        extra=requested_extra,
        crossfade=crossfade,
        search=search,
        rvc_version=rvc_version,
    )
    return replace(
        profile,
        chunk_frames=chunk,
        extra_frames=requested_extra,
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

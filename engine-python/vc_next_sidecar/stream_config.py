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
    "quality": StreamProfile("quality", 12_000, 28_800, 2_400, 720),
    "balanced": StreamProfile("balanced", 9_600, 24_000, 1_920, 576),
    "latency": StreamProfile("latency", 7_680, 19_200, 1_440, 480),
}


MIN_STREAM_FRAMES = 480
MAX_STREAM_FRAMES = 480_000


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
    crossfade = min(profile.crossfade_frames, max(2, chunk // 4))
    search = min(profile.sola_search_frames, max(0, chunk // 8))
    analysis = max(requested_extra, chunk + crossfade + search)
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
    if not math.isfinite(threshold) or not 0.01 <= threshold <= 0.20:
        raise ValueError("The RMVPE threshold must be between 0.01 and 0.20.")
    return threshold

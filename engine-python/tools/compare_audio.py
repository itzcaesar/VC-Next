"""Align and compare two mono voice-conversion recordings.

The tool is deliberately independent from the live engine. It is useful when
the same checkpoint is recorded once through w-okada and once through VC Next:
it reports the best sample alignment, level difference, waveform error, and
silence behavior without pretending that an unaligned recording is a model
quality comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from vc_next_sidecar.rvc_compat.resampling import resample_kaiser_fast


def load_mono(path: str | Path, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    source_rate = int(sample_rate)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if target_rate is not None and sample_rate != target_rate:
        mono = resample_kaiser_fast(mono, sample_rate, target_rate)
        sample_rate = target_rate
    return np.ascontiguousarray(mono, dtype=np.float32), source_rate


def trim_silence(samples: np.ndarray, threshold: float) -> tuple[np.ndarray, int, int]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    active = np.flatnonzero(np.abs(values) >= threshold)
    if active.size == 0:
        return np.zeros(0, dtype=np.float32), 0, int(values.size)
    start = int(active[0])
    end = int(active[-1]) + 1
    return np.ascontiguousarray(values[start:end]), start, end


def _fft_cross_correlation(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    size = reference.size + candidate.size - 1
    fft_size = 1 << max(1, (size - 1).bit_length())
    reference_fft = np.fft.rfft(reference, fft_size)
    candidate_fft = np.fft.rfft(candidate, fft_size)
    return np.fft.irfft(reference_fft * np.conj(candidate_fft), fft_size)


def best_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_lag_frames: int,
) -> tuple[int, float]:
    """Return candidate-leading lag and normalized correlation.

    A positive lag means the candidate starts later than the reference and
    should be compared against ``reference[lag:]``. The FFT keeps the search
    bounded for multi-minute recordings.
    """

    if reference.size == 0 or candidate.size == 0:
        return 0, 0.0
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    reference -= reference.mean()
    candidate -= candidate.mean()
    correlation = _fft_cross_correlation(reference, candidate)
    limit = min(max_lag_frames, max(reference.size, candidate.size) - 1)
    lags = np.arange(-limit, limit + 1, dtype=np.int64)
    positive = lags >= 0
    raw_scores = np.empty(lags.shape, dtype=np.float64)
    raw_scores[positive] = correlation[lags[positive]]
    raw_scores[~positive] = correlation[correlation.size + lags[~positive]]
    # The final aligned metric uses exact per-window normalization. For the
    # lag search, a global energy denominator is both stable and vectorized;
    # excluding very short overlaps prevents an edge from winning solely due
    # to the smaller comparison window.
    overlap = np.where(
        positive,
        np.minimum(reference.size - lags, candidate.size),
        np.minimum(reference.size, candidate.size + lags),
    )
    minimum_overlap = max(16, min(reference.size, candidate.size) // 2)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    scores = np.where(
        overlap >= minimum_overlap,
        raw_scores / denominator if denominator else -1.0,
        -1.0,
    )
    winner = int(np.argmax(scores))
    return int(lags[winner]), float(scores[winner])


def align(reference: np.ndarray, candidate: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        length = min(reference.size - lag, candidate.size)
        return reference[lag : lag + length], candidate[:length]
    length = min(reference.size, candidate.size + lag)
    return reference[:length], candidate[-lag : -lag + length]


def _rms(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def compare_recordings(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_rate: int,
    max_lag_ms: float = 1_000.0,
    silence_threshold: float = 0.0005,
) -> dict[str, Any]:
    reference = np.nan_to_num(np.asarray(reference, dtype=np.float32).reshape(-1))
    candidate = np.nan_to_num(np.asarray(candidate, dtype=np.float32).reshape(-1))
    lag, correlation = best_lag(
        reference,
        candidate,
        max_lag_frames=max(1, round(sample_rate * max_lag_ms / 1_000.0)),
    )
    aligned_reference, aligned_candidate = align(reference, candidate, lag)
    if aligned_reference.size:
        reference_centered = aligned_reference.astype(np.float64) - float(np.mean(aligned_reference))
        candidate_centered = aligned_candidate.astype(np.float64) - float(np.mean(aligned_candidate))
        denominator = float(np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered))
        aligned_correlation = float(
            np.dot(reference_centered, candidate_centered) / denominator
        ) if denominator else 0.0
        error = aligned_candidate.astype(np.float64) - aligned_reference.astype(np.float64)
    else:
        aligned_correlation = 0.0
        error = np.zeros(0, dtype=np.float64)
    return {
        "sampleRate": sample_rate,
        "referenceFrames": int(reference.size),
        "candidateFrames": int(candidate.size),
        "alignedFrames": int(aligned_reference.size),
        "alignedSeconds": round(aligned_reference.size / sample_rate, 3),
        "bestLagFrames": lag,
        "bestLagMs": round(lag * 1_000.0 / sample_rate, 3),
        "correlation": round(aligned_correlation, 7),
        "searchCorrelation": round(correlation, 7),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0, 8),
        "mae": round(float(np.mean(np.abs(error))) if error.size else 0.0, 8),
        "referenceRms": round(_rms(reference), 8),
        "candidateRms": round(_rms(candidate), 8),
        "gainRatio": round(_rms(candidate) / _rms(reference), 7) if _rms(reference) else None,
        "referencePeak": round(float(np.max(np.abs(reference))) if reference.size else 0.0, 8),
        "candidatePeak": round(float(np.max(np.abs(candidate))) if candidate.size else 0.0, 8),
        "referenceSilenceRatio": round(float(np.mean(np.abs(reference) < silence_threshold)) if reference.size else 1.0, 7),
        "candidateSilenceRatio": round(float(np.mean(np.abs(candidate) < silence_threshold)) if candidate.size else 1.0, 7),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Align and compare two voice-conversion WAV recordings")
    parser.add_argument("--reference", required=True, help="Reference recording, for example w-okada output")
    parser.add_argument("--candidate", required=True, help="VC Next recording")
    parser.add_argument("--max-lag-ms", type=float, default=1_000.0)
    parser.add_argument("--silence-threshold", type=float, default=0.0005)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        reference, sample_rate = load_mono(args.reference)
        candidate, candidate_rate = load_mono(args.candidate)
        if candidate_rate != sample_rate:
            candidate, _ = load_mono(args.candidate, sample_rate)
        report = {
            "reference": Path(args.reference).name,
            "candidate": Path(args.candidate).name,
            "referenceSampleRate": sample_rate,
            "candidateResampledFrom": candidate_rate if candidate_rate != sample_rate else None,
            "comparison": compare_recordings(
                reference,
                candidate,
                sample_rate=sample_rate,
                max_lag_ms=args.max_lag_ms,
                silence_threshold=args.silence_threshold,
            ),
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

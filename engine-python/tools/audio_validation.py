"""Measure Windows audio loopback delay and long-session callback stability.

This tool intentionally stays outside the real-time VC Next engine. It is a QA
harness for a physical/virtual route: play a sparse impulse train, record the
return route, and report detected delays plus callback overflow flags. Install
the optional ``sounddevice`` package before using the hardware modes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImpulseMatch:
    expected_frame: int
    detected_frame: int | None
    delay_frames: int | None
    peak: float


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def match_impulses(
    playback: np.ndarray,
    recorded: np.ndarray,
    *,
    sample_rate: int,
    threshold: float = 0.15,
    search_before_ms: float = 20.0,
    search_after_ms: float = 500.0,
) -> list[ImpulseMatch]:
    """Find each played impulse in the nearby recorded window.

    The search is local around the known playback frame, so memory and runtime
    remain bounded even for multi-hour soak captures.
    """

    playback = np.asarray(playback, dtype=np.float32).reshape(-1)
    recorded = np.asarray(recorded, dtype=np.float32).reshape(-1)
    if playback.size == 0 or recorded.size == 0:
        return []
    impulse_mask = np.abs(playback) >= threshold
    starts = np.flatnonzero(impulse_mask & ~np.r_[False, impulse_mask[:-1]])
    before = round(sample_rate * search_before_ms / 1_000.0)
    after = round(sample_rate * search_after_ms / 1_000.0)
    matches: list[ImpulseMatch] = []
    for expected in starts.tolist():
        left = max(0, expected - before)
        right = min(recorded.size, expected + after + 1)
        if right <= left:
            matches.append(ImpulseMatch(expected, None, None, 0.0))
            continue
        window = np.abs(recorded[left:right])
        peak_index = int(np.argmax(window))
        peak = float(window[peak_index])
        detected = left + peak_index if peak >= threshold else None
        matches.append(
            ImpulseMatch(
                expected,
                detected,
                detected - expected if detected is not None else None,
                peak,
            )
        )
    return matches


def summarize_matches(matches: list[ImpulseMatch], sample_rate: int) -> dict[str, Any]:
    delays_ms = [match.delay_frames * 1_000.0 / sample_rate for match in matches if match.delay_frames is not None]
    return {
        "impulses": len(matches),
        "detected": len(delays_ms),
        "rejected": len(matches) - len(delays_ms),
        "delayMs": {
            "p50": percentile(delays_ms, 0.50),
            "p95": percentile(delays_ms, 0.95),
            "min": min(delays_ms) if delays_ms else None,
            "max": max(delays_ms) if delays_ms else None,
        },
        "matches": [
            {
                "expectedFrame": match.expected_frame,
                "detectedFrame": match.detected_frame,
                "delayFrames": match.delay_frames,
                "peak": round(match.peak, 6),
            }
            for match in matches
        ],
    }


def _load_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Hardware validation requires the optional 'sounddevice' package. "
            "Install engine-python/requirements-audio-validation.txt first."
        ) from error
    return sd


def _device_label(sd: Any, device: int | str | None) -> str:
    if device is None:
        return "default"
    if isinstance(device, int):
        return str(sd.query_devices(device)["name"])
    return str(device)


def build_impulse_schedule(
    *,
    total_frames: int,
    block_size: int,
    interval_frames: int,
    impulse_count: int | None,
) -> tuple[int, list[int]]:
    """Return the capture length and exact impulse positions for loopback QA."""

    if interval_frames <= 0 or block_size <= 0:
        raise ValueError("block_size and interval_frames must be positive")
    if impulse_count is not None:
        if impulse_count <= 0:
            raise ValueError("impulse_count must be positive")
        extended_frames = max(
            total_frames,
            block_size + impulse_count * interval_frames + block_size,
        )
        return extended_frames, [
            block_size + index * interval_frames for index in range(impulse_count)
        ]
    return total_frames, list(range(block_size, total_frames, interval_frames))


def run_capture(
    *,
    mode: str,
    sample_rate: int,
    seconds: float,
    block_size: int,
    input_device: int | str | None,
    output_device: int | str | None,
    impulse_interval: float,
    impulse_count: int | None,
    threshold: float,
) -> dict[str, Any]:
    sd = _load_sounddevice()
    interval_frames = max(block_size, round(sample_rate * impulse_interval))
    total_frames = max(1, round(sample_rate * seconds))
    if mode == "loopback":
        # Keep a complete interval after the final impulse so the return peak
        # can be detected even when the route is slower than expected.
        total_frames, impulse_frames = build_impulse_schedule(
            total_frames=total_frames,
            block_size=block_size,
            interval_frames=interval_frames,
            impulse_count=impulse_count,
        )
    else:
        impulse_frames = []
    playback = np.zeros(total_frames, dtype=np.float32)
    for frame in impulse_frames:
        playback[frame : min(total_frames, frame + max(1, sample_rate // 1_000))] = 0.8

    recorded_chunks: list[np.ndarray] = []
    soak_frames = 0
    soak_finite = True
    soak_peak = 0.0
    soak_sum_abs = 0.0
    status_messages: list[str] = []
    cursor = 0

    def callback(indata, outdata, frames, _time, status):
        nonlocal cursor, soak_frames, soak_finite, soak_peak, soak_sum_abs
        if status:
            if len(status_messages) < 100:
                status_messages.append(str(status))
        end = min(total_frames, cursor + frames)
        outdata.fill(0.0)
        if mode == "loopback" and end > cursor:
            outdata[: end - cursor, 0] = playback[cursor:end]
        samples = np.asarray(indata[:, 0], dtype=np.float32)
        if mode == "loopback":
            recorded_chunks.append(samples.copy())
        else:
            soak_frames += int(samples.size)
            soak_finite = soak_finite and bool(np.isfinite(samples).all())
            soak_peak = max(soak_peak, float(np.max(np.abs(samples))) if samples.size else 0.0)
            soak_sum_abs += float(np.sum(np.abs(samples), dtype=np.float64))
        cursor = end
        if cursor >= total_frames:
            raise sd.CallbackStop()

    started = time.perf_counter()
    with sd.Stream(
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        channels=1,
        device=(input_device, output_device),
        callback=callback,
    ):
        while cursor < total_frames:
            time.sleep(min(0.25, max(0.01, (total_frames - cursor) / sample_rate)))
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    recorded = np.concatenate(recorded_chunks)[:total_frames] if recorded_chunks else np.zeros(0, dtype=np.float32)
    report: dict[str, Any] = {
        "mode": mode,
        "sampleRate": sample_rate,
        "blockSize": block_size,
        "requestedSeconds": seconds,
        "captureSeconds": round(total_frames / sample_rate, 3),
        "requestedImpulseCount": impulse_count if mode == "loopback" else None,
        "capturedFrames": int(recorded.size if mode == "loopback" else soak_frames),
        "elapsedMs": round(elapsed_ms, 2),
        "inputDevice": _device_label(sd, input_device),
        "outputDevice": _device_label(sd, output_device),
        "callbackStatus": status_messages,
        "callbackWarnings": len(status_messages),
    }
    if mode == "loopback":
        report["loopback"] = summarize_matches(
            match_impulses(playback, recorded, sample_rate=sample_rate, threshold=threshold),
            sample_rate,
        )
    else:
        report["soak"] = {
            "finite": soak_finite,
            "peak": round(soak_peak, 6),
            "meanAbs": round(soak_sum_abs / soak_frames, 6) if soak_frames else 0.0,
            "frames": soak_frames,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure endpoint loopback delay or callback stability")
    parser.add_argument("--mode", choices=("loopback", "soak"), default="loopback")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--block-size", type=int, default=480)
    parser.add_argument("--input-device", type=str)
    parser.add_argument("--output-device", type=str)
    parser.add_argument("--impulse-interval", type=float, default=1.0)
    parser.add_argument(
        "--impulse-count",
        type=int,
        help="For loopback mode, emit exactly this many impulses (the capture is extended if needed).",
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.block_size <= 0:
        parser.error("--seconds and --block-size must be positive")
    if args.impulse_count is not None and args.impulse_count <= 0:
        parser.error("--impulse-count must be positive")
    if args.impulse_count is not None and args.mode != "loopback":
        parser.error("--impulse-count is only valid with --mode loopback")

    input_device: int | str | None = int(args.input_device) if args.input_device and args.input_device.isdigit() else args.input_device
    output_device: int | str | None = int(args.output_device) if args.output_device and args.output_device.isdigit() else args.output_device
    try:
        report = run_capture(
            mode=args.mode,
            sample_rate=args.sample_rate,
            seconds=args.seconds,
            block_size=args.block_size,
            input_device=input_device,
            output_device=output_device,
            impulse_interval=args.impulse_interval,
            impulse_count=args.impulse_count,
            threshold=args.threshold,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "hint": "Check that both endpoints support the selected sample rate and are not exclusively owned by another application.",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

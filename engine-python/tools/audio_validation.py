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
import threading
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


def resolve_stream_device(sd: Any, device: int | str | None, *, input_direction: bool) -> int | str | None:
    """Resolve a duplicate Windows endpoint name to its WASAPI instance."""

    if not isinstance(device, str) or not device.strip():
        return device
    matches: list[tuple[int, str]] = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("name") != device:
            continue
        channels_key = "max_input_channels" if input_direction else "max_output_channels"
        if int(info.get(channels_key, 0)) < 1:
            continue
        hostapi = sd.query_hostapis(int(info["hostapi"]))["name"]
        matches.append((index, str(hostapi)))
    if not matches:
        return device
    wasapi = [index for index, hostapi in matches if hostapi == "Windows WASAPI"]
    if wasapi:
        return wasapi[0]
    if len(matches) == 1:
        return matches[0][0]
    raise ValueError(
        f"Multiple {'input' if input_direction else 'output'} devices found for {device!r}: "
        + ", ".join(f"[{index}] {hostapi}" for index, hostapi in matches)
    )


def validate_capture_devices(sd: Any, input_device: int | str | None, output_device: int | str | None) -> None:
    """Reject host topologies that cannot safely open this full-duplex probe.

    PortAudio's WDM-KS implementation can block while opening a simultaneous
    input/output stream on some Windows drivers.  The production Rust route
    uses CPAL/WASAPI, so silently attempting that topology in this QA helper
    only creates a misleading frozen process.  Fail before opening the stream
    and tell the caller which host to select instead.
    """

    if not isinstance(input_device, int) or not isinstance(output_device, int):
        return
    try:
        input_info = sd.query_devices(input_device)
        output_info = sd.query_devices(output_device)
        input_host = sd.query_hostapis(int(input_info["hostapi"]))["name"]
        output_host = sd.query_hostapis(int(output_info["hostapi"]))["name"]
    except Exception:
        return
    if input_host == "Windows WDM-KS" or output_host == "Windows WDM-KS":
        raise RuntimeError(
            "The full-duplex validation probe does not open Windows WDM-KS "
            "endpoints safely. Select matching Windows WASAPI endpoints "
            f"instead (input host: {input_host}; output host: {output_host})."
        )


def wasapi_extra_settings(sd: Any, input_device: int | str | None, output_device: int | str | None) -> Any:
    """Allow shared WASAPI to convert mismatched endpoint sample rates."""

    if not isinstance(input_device, int) or not isinstance(output_device, int):
        return None
    try:
        input_info = sd.query_devices(input_device)
        output_info = sd.query_devices(output_device)
        input_host = sd.query_hostapis(int(input_info["hostapi"]))["name"]
        output_host = sd.query_hostapis(int(output_info["hostapi"]))["name"]
        if input_host != "Windows WASAPI" or output_host != "Windows WASAPI":
            return None
        settings = sd.WasapiSettings(exclusive=False, auto_convert=True)
        return (settings, settings)
    except Exception:
        return None


def endpoint_default_sample_rate(sd: Any, device: int | str | None, fallback: int) -> int:
    """Return an endpoint's advertised default rate without opening a stream."""

    if fallback <= 0:
        raise ValueError("fallback sample rate must be positive")
    try:
        info = sd.query_devices(device)
        value = float(info.get("default_samplerate", fallback))
        if np.isfinite(value) and value >= 8_000:
            return max(1, round(value))
    except Exception:
        pass
    return fallback


def map_impulse_frames(
    impulse_frames: list[int],
    *,
    source_rate: int,
    target_rate: int,
) -> list[int]:
    """Map scheduled output frames into the independent capture clock."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate and target_rate must be positive")
    return [round(frame * target_rate / source_rate) for frame in impulse_frames]


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


def capture_timeout_seconds(total_frames: int, sample_rate: int) -> float:
    """Return a bounded wall-clock budget for a callback capture.

    PortAudio normally advances the callback cursor in real time.  A device
    that opens but never delivers callbacks used to leave this QA harness
    sleeping forever, which made a bad Windows route look like a frozen app.
    Leave enough slack for shared-mode scheduling while keeping endpoint
    failures actionable.
    """

    if total_frames <= 0 or sample_rate <= 0:
        raise ValueError("total_frames and sample_rate must be positive")
    capture_seconds = total_frames / sample_rate
    return max(5.0, capture_seconds * 1.5 + 2.0)


def run_split_loopback(
    *,
    input_rate: int,
    output_rate: int,
    seconds: float,
    block_size: int,
    input_device: int | str | None,
    output_device: int | str | None,
    impulse_interval: float,
    impulse_count: int | None,
    threshold: float,
) -> dict[str, Any]:
    """Measure a route with separate clocks for input and output endpoints.

    VB-CABLE and some WASAPI endpoints advertise different default rates (for
    example 44.1 kHz capture and 48 kHz playback).  PortAudio's ``Stream``
    cannot represent that topology as one full-duplex stream.  Opening the
    directions independently mirrors the native CPAL engine and prevents the
    QA helper from silently testing a rate-conversion assumption instead of the
    selected Windows route.
    """

    sd = _load_sounddevice()
    if input_rate <= 0 or output_rate <= 0:
        raise ValueError("input_rate and output_rate must be positive")
    input_device = resolve_stream_device(sd, input_device, input_direction=True)
    output_device = resolve_stream_device(sd, output_device, input_direction=False)
    validate_capture_devices(sd, input_device, output_device)

    interval_frames = max(block_size, round(output_rate * impulse_interval))
    total_output_frames = max(1, round(output_rate * seconds))
    total_output_frames, impulse_frames = build_impulse_schedule(
        total_frames=total_output_frames,
        block_size=block_size,
        interval_frames=interval_frames,
        impulse_count=impulse_count,
    )
    output_playback = np.zeros(total_output_frames, dtype=np.float32)
    pulse_width = max(1, output_rate // 1_000)
    for frame in impulse_frames:
        output_playback[frame : min(total_output_frames, frame + pulse_width)] = 0.8

    # Leave enough capture tail for the normal search window and for a shared
    # Windows mixer to drain its period after the final output write.
    tail_seconds = max(0.5, 500.0 / 1_000.0)
    total_input_frames = max(
        1,
        round((total_output_frames / output_rate + tail_seconds) * input_rate),
    )
    input_block_size = max(1, round(block_size * input_rate / output_rate))
    mapped_impulses = map_impulse_frames(
        impulse_frames,
        source_rate=output_rate,
        target_rate=input_rate,
    )
    input_playback = np.zeros(total_input_frames, dtype=np.float32)
    input_pulse_width = max(1, input_rate // 1_000)
    for frame in mapped_impulses:
        input_playback[frame : min(total_input_frames, frame + input_pulse_width)] = 0.8

    recorded_chunks: list[np.ndarray] = []
    status_messages: list[str] = []
    captured_frames = 0
    capture_done = threading.Event()

    def input_callback(indata, frames, _time, status):
        nonlocal captured_frames
        if status and len(status_messages) < 100:
            status_messages.append(str(status))
        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
        recorded_chunks.append(samples)
        captured_frames += int(samples.size)
        if captured_frames >= total_input_frames:
            capture_done.set()
            raise sd.CallbackStop()

    started = time.perf_counter()
    timeout_seconds = capture_timeout_seconds(total_input_frames, input_rate)
    output_cursor = 0
    with sd.InputStream(
        samplerate=input_rate,
        blocksize=input_block_size,
        dtype="float32",
        channels=1,
        device=input_device,
        callback=input_callback,
    ):
        with sd.OutputStream(
            samplerate=output_rate,
            blocksize=block_size,
            dtype="float32",
            channels=1,
            device=output_device,
        ) as output_stream:
            while output_cursor < total_output_frames:
                end = min(total_output_frames, output_cursor + block_size)
                block = output_playback[output_cursor:end]
                if block.size < block_size:
                    block = np.pad(block, (0, block_size - block.size))
                output_stream.write(np.ascontiguousarray(block.reshape(-1, 1)))
                output_cursor = end

            while not capture_done.is_set():
                elapsed = time.perf_counter() - started
                if elapsed >= timeout_seconds:
                    raise RuntimeError(
                        "The split input callback did not deliver the requested capture "
                        f"within {timeout_seconds:.1f}s ({captured_frames}/{total_input_frames} frames). "
                        "The selected endpoint may be silent, busy, or incompatible "
                        "with its advertised sample rate."
                    )
                time.sleep(0.02)

    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    recorded = (
        np.concatenate(recorded_chunks)[:total_input_frames]
        if recorded_chunks
        else np.zeros(0, dtype=np.float32)
    )
    matches = match_impulses(
        input_playback,
        recorded,
        sample_rate=input_rate,
        threshold=threshold,
    )
    return {
        "mode": "loopback",
        "topology": "split-stream",
        "sampleRate": output_rate,
        "inputSampleRate": input_rate,
        "outputSampleRate": output_rate,
        "blockSize": block_size,
        "inputBlockSize": input_block_size,
        "requestedSeconds": seconds,
        "captureSeconds": round(total_input_frames / input_rate, 3),
        "requestedImpulseCount": impulse_count,
        "capturedFrames": int(recorded.size),
        "elapsedMs": round(elapsed_ms, 2),
        "inputDevice": _device_label(sd, input_device),
        "outputDevice": _device_label(sd, output_device),
        "callbackStatus": status_messages,
        "callbackWarnings": len(status_messages),
        "loopback": summarize_matches(matches, input_rate),
    }


def run_capture(
    *,
    mode: str,
    sample_rate: int,
    input_sample_rate: int | None = None,
    output_sample_rate: int | None = None,
    seconds: float,
    block_size: int,
    input_device: int | str | None,
    output_device: int | str | None,
    impulse_interval: float,
    impulse_count: int | None,
    threshold: float,
) -> dict[str, Any]:
    sd = _load_sounddevice()
    input_device = resolve_stream_device(sd, input_device, input_direction=True)
    output_device = resolve_stream_device(sd, output_device, input_direction=False)
    validate_capture_devices(sd, input_device, output_device)
    input_rate = input_sample_rate or sample_rate
    output_rate = output_sample_rate or sample_rate
    if input_rate <= 0 or output_rate <= 0:
        raise ValueError("sample rates must be positive")
    if mode == "loopback" and input_rate != output_rate:
        return run_split_loopback(
            input_rate=input_rate,
            output_rate=output_rate,
            seconds=seconds,
            block_size=block_size,
            input_device=input_device,
            output_device=output_device,
            impulse_interval=impulse_interval,
            impulse_count=impulse_count,
            threshold=threshold,
        )
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
    timeout_seconds = capture_timeout_seconds(total_frames, sample_rate)
    stream_kwargs = dict(
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        channels=1,
        device=(input_device, output_device),
        callback=callback,
    )
    extra_settings = wasapi_extra_settings(sd, input_device, output_device)
    if extra_settings is not None:
        stream_kwargs["extra_settings"] = extra_settings
    with sd.Stream(**stream_kwargs):
        while cursor < total_frames:
            elapsed = time.perf_counter() - started
            if elapsed >= timeout_seconds:
                raise RuntimeError(
                    "The audio callback did not deliver the requested capture "
                    f"within {timeout_seconds:.1f}s ({cursor}/{total_frames} frames). "
                    "The selected endpoint may be silent, busy, or incompatible "
                    "with the requested sample rate."
                )
            remaining = max(0.01, (total_frames - cursor) / sample_rate)
            time.sleep(min(0.25, remaining, timeout_seconds - elapsed))
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    recorded = np.concatenate(recorded_chunks)[:total_frames] if recorded_chunks else np.zeros(0, dtype=np.float32)
    report: dict[str, Any] = {
        "mode": mode,
        "topology": "full-duplex",
        "sampleRate": sample_rate,
        "inputSampleRate": sample_rate,
        "outputSampleRate": sample_rate,
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
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        help="Override the capture endpoint rate. If it differs from --output-sample-rate, use split streams.",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        help="Override the playback endpoint rate. If it differs from --input-sample-rate, use split streams.",
    )
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
            input_sample_rate=args.input_sample_rate,
            output_sample_rate=args.output_sample_rate,
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

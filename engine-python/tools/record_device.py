"""Record a named Windows input endpoint to a WAV file for route QA.

This is intentionally a small diagnostic helper rather than part of the
runtime engine.  It lets the native speech-loopback harness capture the
converted side of a virtual cable so we can inspect the actual end-to-end
signal, not just callback telemetry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import wave

import numpy as np
import sounddevice as sd


def resolve_input_device(device: str | int) -> int | str:
    """Prefer the Windows WASAPI instance when an endpoint name is duplicated."""
    if isinstance(device, int):
        return device
    matches: list[tuple[int, str]] = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("name") != device or int(info.get("max_input_channels", 0)) < 1:
            continue
        hostapi = sd.query_hostapis(int(info["hostapi"]))["name"]
        matches.append((index, hostapi))
    if not matches:
        return device
    for index, hostapi in matches:
        if hostapi == "Windows WASAPI":
            return index
    if len(matches) == 1:
        return matches[0][0]
    raise ValueError(
        f"Multiple input devices found for {device!r}: "
        + ", ".join(f"[{index}] {hostapi}" for index, hostapi in matches)
    )


def record_device(*, device: str, output: Path, seconds: float, sample_rate: int, block_size: int) -> dict[str, object]:
    if seconds <= 0 or sample_rate <= 0 or block_size <= 0:
        raise ValueError("seconds, sample-rate, and block-size must be positive")
    selected = resolve_input_device(device)
    total_frames = max(1, round(seconds * sample_rate))
    chunks: list[np.ndarray] = []
    frames = 0
    statuses: list[str] = []

    def callback(indata, count, _time, status):
        nonlocal frames
        if status and len(statuses) < 100:
            statuses.append(str(status))
        chunks.append(np.asarray(indata[:, 0], dtype=np.float32).copy())
        frames += int(count)
        if frames >= total_frames:
            raise sd.CallbackStop()

    started = time.perf_counter()
    with sd.InputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        channels=1,
        device=selected,
        callback=callback,
    ):
        while frames < total_frames:
            time.sleep(min(0.25, max(0.01, (total_frames - frames) / sample_rate)))

    samples = np.concatenate(chunks)[:total_frames] if chunks else np.zeros(0, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2", copy=False)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return {
        "ok": True,
        "device": device,
        "selectedDevice": str(selected),
        "sampleRate": sample_rate,
        "requestedSeconds": seconds,
        "capturedFrames": int(samples.size),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "peak": round(float(np.max(np.abs(samples))) if samples.size else 0.0, 6),
        "meanAbs": round(float(np.mean(np.abs(samples))) if samples.size else 0.0, 6),
        "callbackStatus": statuses,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a named input endpoint to a mono WAV")
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--block-size", type=int, default=480)
    args = parser.parse_args()
    try:
        print(json.dumps(record_device(device=args.device, output=args.output, seconds=args.seconds, sample_rate=args.sample_rate, block_size=args.block_size), indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

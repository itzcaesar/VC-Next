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


def wasapi_shared_settings(device: int | str) -> object | None:
    """Match the native route's shared WASAPI/rate-conversion policy."""
    if not isinstance(device, int):
        return None
    try:
        info = sd.query_devices(device)
        hostapi = sd.query_hostapis(int(info["hostapi"]))["name"]
        if hostapi != "Windows WASAPI":
            return None
        return sd.WasapiSettings(exclusive=False, auto_convert=True)
    except Exception:
        return None


def record_device(
    *,
    device: str,
    output: Path,
    seconds: float,
    sample_rate: int,
    block_size: int,
    ready_file: str | Path | None = None,
    stop_file: str | Path | None = None,
) -> dict[str, object]:
    if seconds <= 0 or sample_rate <= 0 or block_size <= 0:
        raise ValueError("seconds, sample-rate, and block-size must be positive")
    selected = resolve_input_device(device)
    total_frames = max(1, round(seconds * sample_rate))
    chunks: list[np.ndarray] = []
    frames = 0
    stopped = False
    statuses: list[str] = []
    ready_path = Path(ready_file) if ready_file is not None else None
    stop_path = Path(stop_file) if stop_file is not None else None

    def callback(indata, count, _time, status):
        nonlocal frames, stopped
        if status and len(statuses) < 100:
            statuses.append(str(status))
        chunks.append(np.asarray(indata[:, 0], dtype=np.float32).copy())
        frames += int(count)
        if frames >= total_frames or (stop_path is not None and stop_path.is_file()):
            stopped = True
            raise sd.CallbackStop()

    started = time.perf_counter()
    stream_kwargs = dict(
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="float32",
        channels=1,
        device=selected,
        callback=callback,
    )
    extra_settings = wasapi_shared_settings(selected)
    if extra_settings is not None:
        stream_kwargs["extra_settings"] = extra_settings
    with sd.InputStream(**stream_kwargs):
        if ready_path is not None:
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(
                json.dumps(
                    {
                        "device": device,
                        "selectedDevice": str(selected),
                        "sampleRate": sample_rate,
                        "openedAt": time.time(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        while frames < total_frames and not stopped:
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
    parser.add_argument(
        "--ready-file",
        help="Write a JSON marker after the input stream opens.",
    )
    parser.add_argument(
        "--stop-file",
        help="Stop after the marker file appears and write the captured WAV.",
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                record_device(
                    device=args.device,
                    output=args.output,
                    seconds=args.seconds,
                    sample_rate=args.sample_rate,
                    block_size=args.block_size,
                    ready_file=args.ready_file,
                    stop_file=args.stop_file,
                ),
                indent=2,
            )
        )
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

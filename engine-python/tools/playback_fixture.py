"""Loop a known speech fixture into a named Windows output endpoint.

This is an optional QA helper for the native-route validation harness. It is
deliberately separate from the production engine: it feeds a virtual cable so
the capture side can be exercised with repeatable speech without touching a
physical microphone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from vc_next_sidecar.rvc_compat.resampling import resample_kaiser_fast


def resolve_output_device(device: str | int) -> int | str:
    """Resolve duplicate Windows endpoint names to the WASAPI instance.

    PortAudio exposes the same endpoint through MME, DirectSound, and WASAPI.
    sounddevice rejects an ambiguous name, while the native validator uses the
    Windows audio host.  Prefer the matching WASAPI endpoint so the fixture and
    capture route share the same device family.
    """
    if isinstance(device, int):
        return device
    matches: list[tuple[int, dict, str]] = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("name") != device or int(info.get("max_output_channels", 0)) < 1:
            continue
        hostapi = sd.query_hostapis(int(info["hostapi"]))["name"]
        matches.append((index, info, hostapi))
    if not matches:
        # Let PortAudio produce its normal descriptive error for non-Windows or
        # partial device names that sounddevice can resolve itself.
        return device
    for index, _info, hostapi in matches:
        if hostapi == "Windows WASAPI":
            return index
    if len(matches) == 1:
        return matches[0][0]
    raise ValueError(
        f"Multiple output devices found for {device!r}: "
        + ", ".join(f"[{index}] {hostapi}" for index, _info, hostapi in matches)
    )


def play_fixture(
    *,
    input_path: str,
    device: str,
    seconds: float,
    sample_rate: int = 48_000,
) -> dict[str, object]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    waveform, source_rate = sf.read(input_path, dtype="float32", always_2d=False)
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if source_rate != sample_rate:
        samples = resample_kaiser_fast(samples, source_rate, sample_rate)
    if samples.size == 0:
        raise ValueError("The fixture contains no audio samples.")

    block_frames = 480
    cursor = 0
    total_frames = max(1, round(seconds * sample_rate))
    written = 0
    started = time.perf_counter()
    output_device = resolve_output_device(device)
    with sd.OutputStream(
        samplerate=sample_rate,
        blocksize=block_frames,
        dtype="float32",
        channels=1,
        device=output_device,
    ) as stream:
        while written < total_frames:
            count = min(block_frames, total_frames - written)
            end = cursor + count
            if end <= samples.size:
                block = samples[cursor:end]
            else:
                first = samples[cursor:]
                remaining = count - first.size
                block = np.concatenate((first, samples[:remaining]))
            stream.write(np.ascontiguousarray(block.reshape(-1, 1)))
            cursor = (cursor + count) % samples.size
            written += count

    return {
        "input": Path(input_path).name,
        "device": device,
        "sourceSampleRate": source_rate,
        "outputSampleRate": sample_rate,
        "writtenFrames": written,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "peak": round(float(np.max(np.abs(samples))), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a speech fixture into a Windows output endpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    try:
        result = play_fixture(
            input_path=args.input,
            device=args.device,
            seconds=args.seconds,
        )
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

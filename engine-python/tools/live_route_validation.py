"""Validate a real-time converted microphone-to-output route.

This is an acceptance harness rather than the production Rust audio engine. It
uses the same persistent framed RVC worker as VC Next, but drives it from a
full-duplex ``sounddevice`` callback so a physical or virtual Windows route can
be exercised without the desktop UI. The callback never waits for inference;
bounded queues make drops and underruns visible in the report instead.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

from live_worker_smoke import _request
from vc_next_sidecar.framed_protocol import (
    AUDIO_REQUEST,
    AUDIO_RESPONSE,
    JSON_REQUEST,
    JSON_RESPONSE,
    SHUTDOWN,
    Frame,
)
from vc_next_sidecar.live_worker import LIVE_INPUT_SAMPLE_RATE


QUEUE_BLOCKS = 64
MAX_STATUS_MESSAGES = 100
PRIME_CHUNKS = 2


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class ChunkAccumulator:
    """Collect arbitrary callback blocks into exact worker-sized chunks."""

    def __init__(self, chunk_frames: int) -> None:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        self.chunk_frames = chunk_frames
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size == 0:
            return []
        combined = np.concatenate((self._pending, incoming))
        complete = (combined.size // self.chunk_frames) * self.chunk_frames
        chunks = [
            np.ascontiguousarray(combined[offset : offset + self.chunk_frames])
            for offset in range(0, complete, self.chunk_frames)
        ]
        self._pending = np.ascontiguousarray(combined[complete:])
        return chunks

    def finish(self) -> list[np.ndarray]:
        if self._pending.size == 0:
            return []
        padded = np.pad(
            self._pending,
            (0, self.chunk_frames - self._pending.size),
        ).astype(np.float32, copy=False)
        self._pending = np.zeros(0, dtype=np.float32)
        return [np.ascontiguousarray(padded)]


@dataclass
class RouteCounters:
    callback_frames: int = 0
    input_drops: int = 0
    output_drops: int = 0
    output_underruns: int = 0
    output_frames: int = 0
    output_peak: float = 0.0
    callback_status: list[str] | None = None

    def __post_init__(self) -> None:
        if self.callback_status is None:
            self.callback_status = []


def _portable_status(status: dict[str, Any]) -> dict[str, Any]:
    portable = dict(status)
    for key in ("modelPath", "contentvecPath", "rmvpePath", "indexPath"):
        value = portable.get(key)
        if value:
            portable[key] = Path(str(value)).name
    return portable


def _device_label(sd: Any, device: int | str | None) -> str:
    if device is None:
        return "default"
    if isinstance(device, int):
        return str(sd.query_devices(device)["name"])
    return str(device)


def _load_worker(
    process: subprocess.Popen[bytes],
    *,
    model: str,
    index: str | None,
    preset: str,
    chunk_frames: int | None,
    extra_frames: int | None,
    index_ratio: float,
    protect_ratio: float,
    pitch_shift: float,
    speaker_id: int,
    f0_threshold: float,
    use_package_defaults: bool,
) -> tuple[dict[str, Any], float]:
    handshake = _request(
        process,
        Frame(JSON_REQUEST, 1, b'{"method":"handshake","params":{}}'),
    )
    if handshake.kind != JSON_RESPONSE:
        raise RuntimeError("The live worker handshake returned the wrong frame kind.")
    params: dict[str, Any] = {
        "modelPath": model,
        "streamingPreset": preset,
        "speakerId": speaker_id,
        "f0Threshold": f0_threshold,
    }
    if use_package_defaults:
        if index:
            params["indexPath"] = index
    else:
        params.update(
            {
                "indexPath": index,
                "pitchShift": pitch_shift,
                "indexRatio": index_ratio,
                "protectRatio": protect_ratio,
            }
        )
    if chunk_frames is not None:
        params["chunkFrames"] = chunk_frames
    if extra_frames is not None:
        params["extraFrames"] = extra_frames
    payload = json.dumps(
        {"method": "load_model", "params": params},
        separators=(",", ":"),
    ).encode("utf-8")
    started = time.perf_counter()
    loaded = _request(process, Frame(JSON_REQUEST, 2, payload))
    load_ms = (time.perf_counter() - started) * 1_000.0
    if loaded.kind != JSON_RESPONSE:
        raise RuntimeError("The live worker model load returned the wrong frame kind.")
    return json.loads(loaded.payload), load_ms


def run_route(
    *,
    model: str,
    index: str | None,
    input_device: int | str | None,
    output_device: int | str | None,
    seconds: float,
    block_size: int,
    preset: str,
    chunk_frames: int | None,
    extra_frames: int | None,
    index_ratio: float,
    protect_ratio: float,
    pitch_shift: float,
    speaker_id: int,
    f0_threshold: float,
    strict: bool,
    use_package_defaults: bool,
) -> dict[str, Any]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not use_package_defaults and index_ratio > 0.0 and not index:
        raise ValueError("index is required when index_ratio is above zero")

    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Converted-route validation requires the optional 'sounddevice' package. "
            "Install engine-python/requirements-audio-validation.txt first."
        ) from error

    engine_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "vc_next_sidecar", "--worker"],
        cwd=engine_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_BLOCKS)
    output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_BLOCKS)
    stop_event = threading.Event()
    worker_error: list[str] = []
    process_times: list[float] = []
    deadline_misses = 0
    first_produced_at: float | None = None
    first_played_at: float | None = None
    counters = RouteCounters()
    worker_status: dict[str, Any] = {}
    worker_thread: threading.Thread | None = None

    try:
        worker_status, load_ms = _load_worker(
            process,
            model=model,
            index=index,
            preset=preset,
            chunk_frames=chunk_frames,
            extra_frames=extra_frames,
            index_ratio=index_ratio,
            protect_ratio=protect_ratio,
            pitch_shift=pitch_shift,
            speaker_id=speaker_id,
            f0_threshold=f0_threshold,
            use_package_defaults=use_package_defaults,
        )
        worker_chunk_frames = int(worker_status["chunkFrames"])
        worker_deadline_ms = float(worker_status["chunkMilliseconds"])
        accumulator = ChunkAccumulator(worker_chunk_frames)

        def process_chunk(chunk: np.ndarray) -> None:
            nonlocal deadline_misses, first_produced_at
            request_started = time.perf_counter()
            response = _request(
                process,
                Frame(
                    AUDIO_REQUEST,
                    3 + len(process_times),
                    np.asarray(chunk, dtype="<f4").tobytes(),
                ),
            )
            elapsed_ms = (time.perf_counter() - request_started) * 1_000.0
            process_times.append(elapsed_ms)
            if elapsed_ms > worker_deadline_ms:
                deadline_misses += 1
            if response.kind != AUDIO_RESPONSE:
                raise RuntimeError("The live worker returned the wrong audio frame kind.")
            converted = np.frombuffer(response.payload, dtype="<f4").copy()
            if converted.shape[0] != worker_chunk_frames or not np.isfinite(converted).all():
                raise RuntimeError("The live worker returned invalid converted audio.")
            if first_produced_at is None:
                first_produced_at = time.perf_counter()
            try:
                output_queue.put_nowait(converted)
            except queue.Full:
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    pass
                counters.output_drops += 1
                output_queue.put_nowait(converted)

        def worker_loop() -> None:
            try:
                while not stop_event.is_set() or not input_queue.empty():
                    try:
                        block = input_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    for chunk in accumulator.push(block):
                        process_chunk(chunk)
                for chunk in accumulator.finish():
                    process_chunk(chunk)
            except Exception as error:  # pragma: no cover - exercised by hardware
                worker_error.append(str(error))
                stop_event.set()

        worker_thread = threading.Thread(target=worker_loop, name="vc-next-route-worker")
        worker_thread.start()

        stream_started_at = time.perf_counter()
        cursor = 0
        total_frames = max(1, round(seconds * LIVE_INPUT_SAMPLE_RATE))
        pending_output: deque[np.ndarray] = deque()
        pending_offset = 0
        primed = False

        def callback(indata: Any, outdata: Any, frames: int, _time_info: Any, status: Any) -> None:
            nonlocal cursor, pending_offset, primed, first_played_at
            if status and len(counters.callback_status or []) < MAX_STATUS_MESSAGES:
                counters.callback_status.append(str(status))
            samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
            try:
                input_queue.put_nowait(samples)
            except queue.Full:
                counters.input_drops += 1
            outdata.fill(0.0)
            if not primed and output_queue.qsize() < PRIME_CHUNKS:
                counters.callback_frames += frames
                cursor += frames
                if cursor >= total_frames:
                    raise sd.CallbackStop()
                return
            if not primed:
                primed = True
            written = 0
            while written < frames:
                if not pending_output:
                    try:
                        pending_output.append(output_queue.get_nowait())
                        pending_offset = 0
                    except queue.Empty:
                        counters.output_underruns += 1
                        break
                current = pending_output[0]
                available = current.shape[0] - pending_offset
                count = min(frames - written, available)
                outdata[written : written + count, 0] = current[pending_offset : pending_offset + count]
                written += count
                pending_offset += count
                if pending_offset >= current.shape[0]:
                    pending_output.popleft()
                    pending_offset = 0
            counters.callback_frames += frames
            counters.output_frames += written
            if written:
                if first_played_at is None:
                    first_played_at = time.perf_counter()
                counters.output_peak = max(
                    counters.output_peak,
                    float(np.max(np.abs(outdata[:written, 0]))),
                )
            cursor += frames
            if cursor >= total_frames:
                raise sd.CallbackStop()

        with sd.Stream(
            samplerate=LIVE_INPUT_SAMPLE_RATE,
            blocksize=block_size,
            dtype="float32",
            channels=1,
            device=(input_device, output_device),
            callback=callback,
        ):
            while cursor < total_frames and not worker_error:
                time.sleep(0.05)

        stop_event.set()
        drain_deadline = time.perf_counter() + max(5.0, worker_deadline_ms / 1_000.0 * 4.0)
        while worker_thread.is_alive() and time.perf_counter() < drain_deadline:
            time.sleep(0.02)
        if worker_thread.is_alive():
            worker_error.append("The converted route worker did not drain before the timeout.")
        worker_thread.join(timeout=0.2)

        final_status = _request(
            process,
            Frame(JSON_REQUEST, 10_000_000, b'{"method":"status","params":{}}'),
        )
        if final_status.kind == JSON_RESPONSE:
            worker_status = json.loads(final_status.payload)
        elapsed_seconds = time.perf_counter() - stream_started_at
        report: dict[str, Any] = {
            "mode": "live-route",
            "requestedSeconds": seconds,
            "elapsedSeconds": round(elapsed_seconds, 3),
            "sampleRate": LIVE_INPUT_SAMPLE_RATE,
            "blockSize": block_size,
            "inputDevice": _device_label(sd, input_device),
            "outputDevice": _device_label(sd, output_device),
            "capturedFrames": counters.callback_frames,
            "outputFrames": counters.output_frames,
            "finite": not bool(worker_error),
            "outputPeak": round(counters.output_peak, 6),
            "loadRoundTripMs": round(load_ms, 1),
            "firstProducedLatencyMs": round(
                (first_produced_at - stream_started_at) * 1_000.0, 1
            )
            if first_produced_at is not None
            else None,
            "firstPlayedLatencyMs": round(
                (first_played_at - stream_started_at) * 1_000.0, 1
            )
            if first_played_at is not None
            else None,
            "primed": primed,
            "primeChunks": PRIME_CHUNKS,
            "processMs": {
                "p50": percentile(process_times, 0.50),
                "p95": percentile(process_times, 0.95),
                "max": max(process_times) if process_times else None,
            },
            "workerCalls": len(process_times),
            "deadlineMs": worker_deadline_ms,
            "deadlineMisses": deadline_misses,
            "callbackWarnings": len(counters.callback_status or []),
            "callbackStatus": counters.callback_status,
            "inputDrops": counters.input_drops,
            "outputDrops": counters.output_drops,
            "outputUnderruns": counters.output_underruns,
            "workerErrors": worker_error,
            "status": _portable_status(worker_status),
        }
        failed = bool(worker_error)
        if strict:
            failed = failed or any(
                (
                    counters.input_drops,
                    counters.output_drops,
                    counters.output_underruns,
                    deadline_misses,
                    len(counters.callback_status or []),
                )
            )
        report["acceptancePassed"] = not failed
        return report
    finally:
        stop_event.set()
        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=2.0)
        if process.poll() is None and (worker_thread is None or not worker_thread.is_alive()):
            try:
                _request(process, Frame(SHUTDOWN, 20_000_000, b""))
            except Exception:
                process.kill()
        elif process.poll() is None:
            process.kill()
        _, stderr = process.communicate(timeout=10)
        if process.returncode not in (0, None) and stderr:
            print(stderr.decode("utf-8", errors="replace"), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a realtime converted microphone route")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index")
    parser.add_argument("--input-device", required=True)
    parser.add_argument("--output-device", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--block-size", type=int, default=480)
    parser.add_argument(
        "--streaming-preset", choices=("quality", "balanced", "latency"), default="quality"
    )
    parser.add_argument("--chunk-frames", type=int)
    parser.add_argument("--extra-frames", type=int)
    parser.add_argument("--index-ratio", type=float, default=0.3)
    parser.add_argument("--protect-ratio", type=float, default=0.33)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--f0-threshold", type=float, default=0.30)
    parser.add_argument(
        "--use-package-defaults",
        action="store_true",
        help="Import pitch/index/Protect/Chunk/embedder defaults from params.json.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.block_size <= 0:
        parser.error("--seconds and --block-size must be positive")
    if args.index_ratio < 0.0 or args.index_ratio > 1.0:
        parser.error("--index-ratio must be between 0 and 1")
    input_device: int | str = int(args.input_device) if args.input_device.isdigit() else args.input_device
    output_device: int | str = int(args.output_device) if args.output_device.isdigit() else args.output_device
    try:
        report = run_route(
            model=args.model,
            index=args.index,
            input_device=input_device,
            output_device=output_device,
            seconds=args.seconds,
            block_size=args.block_size,
            preset=args.streaming_preset,
            chunk_frames=args.chunk_frames,
            extra_frames=args.extra_frames,
            index_ratio=args.index_ratio,
            protect_ratio=args.protect_ratio,
            pitch_shift=args.pitch_shift,
            speaker_id=args.speaker_id,
            f0_threshold=args.f0_threshold,
            strict=args.strict,
            use_package_defaults=args.use_package_defaults,
        )
    except Exception as error:
        report = {
            "mode": "live-route",
            "acceptancePassed": False,
            "error": str(error),
        }
        print(json.dumps(report, indent=2), file=sys.stderr)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 2
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report.get("acceptancePassed") else 2


if __name__ == "__main__":
    raise SystemExit(main())

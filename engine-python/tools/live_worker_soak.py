"""Run a bounded, report-producing converted-audio soak through the live worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

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


def _load_source(path: str | None, frames: int) -> np.ndarray:
    if path is None:
        return np.zeros(frames, dtype=np.float32)
    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.mean(audio, axis=1, dtype=np.float32))
    if sample_rate != LIVE_INPUT_SAMPLE_RATE:
        waveform = audio_functional.resample(
            waveform, sample_rate, LIVE_INPUT_SAMPLE_RATE
        )
    source = waveform.contiguous().numpy()
    if source.size < frames:
        source = np.pad(source, (0, frames - source.size))
    return np.ascontiguousarray(source, dtype=np.float32)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _portable_status(status: dict[str, object]) -> dict[str, object]:
    """Keep copied soak reports from exposing a user's local directory layout."""

    portable = dict(status)
    for key in ("modelPath", "contentvecPath", "rmvpePath", "indexPath"):
        value = portable.get(key)
        if value:
            portable[key] = Path(str(value)).name
    return portable


def main() -> int:
    parser = argparse.ArgumentParser(description="Soak the persistent live RVC worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index")
    parser.add_argument("--input", help="Optional speech WAV; it loops for the requested duration")
    parser.add_argument("--seconds", type=float, default=7_200.0)
    parser.add_argument("--index-ratio", type=float, default=0.3)
    parser.add_argument("--protect-ratio", type=float, default=0.33)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--f0-threshold", type=float, default=0.30)
    parser.add_argument(
        "--streaming-preset", choices=("quality", "balanced", "latency"), default="balanced"
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        help="Optional custom streaming hop used to evaluate a tuned safety margin.",
    )
    parser.add_argument(
        "--extra-frames",
        type=int,
        help="Optional custom analysis/context window for --chunk-frames.",
    )
    parser.add_argument("--status-interval", type=float, default=30.0)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace requests to the model's 48 kHz audio timeline instead of running as fast as possible.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.status_interval <= 0:
        parser.error("--seconds and --status-interval must be positive")
    if args.index_ratio > 0.0 and not args.index:
        parser.error("--index is required when --index-ratio is above zero")

    process = subprocess.Popen(
        [sys.executable, "-m", "vc_next_sidecar", "--worker"],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    report: dict[str, object] = {}
    try:
        handshake = _request(process, Frame(JSON_REQUEST, 1, b'{"method":"handshake","params":{}}'))
        if handshake.kind != JSON_RESPONSE:
            raise RuntimeError("The live worker handshake returned the wrong frame kind.")
        load_params = {
            "modelPath": args.model,
            "indexPath": args.index,
            "pitchShift": args.pitch_shift,
            "indexRatio": args.index_ratio,
            "protectRatio": args.protect_ratio,
            "speakerId": args.speaker_id,
            "f0Threshold": args.f0_threshold,
            "streamingPreset": args.streaming_preset,
        }
        if args.chunk_frames is not None:
            load_params["chunkFrames"] = args.chunk_frames
        if args.extra_frames is not None:
            load_params["extraFrames"] = args.extra_frames
        load_request = json.dumps(
            {
                "method": "load_model",
                "params": load_params,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        load_started = time.perf_counter()
        loaded_frame = _request(process, Frame(JSON_REQUEST, 2, load_request))
        load_ms = (time.perf_counter() - load_started) * 1_000.0
        loaded = json.loads(loaded_frame.payload)
        status = loaded
        chunk_frames = int(status["chunkFrames"])
        chunk_ms = float(status["chunkMilliseconds"])
        source = _load_source(args.input, max(chunk_frames, LIVE_INPUT_SAMPLE_RATE))
        total_calls = max(1, int(np.ceil(args.seconds * LIVE_INPUT_SAMPLE_RATE / chunk_frames)))
        process_times: list[float] = []
        deadline_misses = 0
        worst_call_index = 0
        worst_process_ms = 0.0
        finite = True
        output_peak = 0.0
        output_frames = 0
        started = time.perf_counter()
        next_status_at = started + args.status_interval
        for call_index in range(total_calls):
            if args.realtime:
                target_time = started + call_index * chunk_frames / LIVE_INPUT_SAMPLE_RATE
                wait_seconds = target_time - time.perf_counter()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            offset = (call_index * chunk_frames) % source.shape[0]
            if offset + chunk_frames <= source.shape[0]:
                chunk = source[offset : offset + chunk_frames]
            else:
                tail = source[offset:]
                chunk = np.concatenate((tail, source[: chunk_frames - tail.shape[0]]))
            request_started = time.perf_counter()
            response = _request(
                process,
                Frame(AUDIO_REQUEST, 3 + call_index, chunk.astype("<f4", copy=False).tobytes()),
            )
            elapsed_ms = (time.perf_counter() - request_started) * 1_000.0
            if response.kind != AUDIO_RESPONSE:
                raise RuntimeError("The live worker returned the wrong audio frame kind.")
            converted = np.frombuffer(response.payload, dtype="<f4")
            if converted.shape[0] != chunk_frames:
                raise RuntimeError("The live worker returned an invalid frame count.")
            finite = finite and bool(np.isfinite(converted).all())
            output_peak = max(output_peak, float(np.max(np.abs(converted))))
            output_frames += int(converted.shape[0])
            process_times.append(elapsed_ms)
            if elapsed_ms > worst_process_ms:
                worst_process_ms = elapsed_ms
                worst_call_index = call_index + 1
            if elapsed_ms > chunk_ms:
                deadline_misses += 1
            now = time.perf_counter()
            if now >= next_status_at and call_index + 1 < total_calls:
                status_frame = _request(
                    process,
                    Frame(JSON_REQUEST, 3 + total_calls + call_index, b'{"method":"status","params":{}}'),
                )
                status = json.loads(status_frame.payload)
                next_status_at = now + args.status_interval

        final_status = _request(
            process,
            Frame(JSON_REQUEST, 4 + total_calls, b'{"method":"status","params":{}}'),
        )
        status = json.loads(final_status.payload)
        elapsed_seconds = time.perf_counter() - started
        report = {
            "mode": "live-worker-soak",
            "requestedSeconds": args.seconds,
            "simulatedAudioSeconds": round(
                total_calls * chunk_frames / LIVE_INPUT_SAMPLE_RATE, 3
            ),
            "realtime": args.realtime,
            "elapsedSeconds": round(elapsed_seconds, 3),
            "calls": total_calls,
            "outputFrames": output_frames,
            "finite": finite,
            "outputPeak": round(output_peak, 6),
            "loadRoundTripMs": round(load_ms, 1),
            "processMs": {
                "p50": _percentile(process_times, 0.50),
                "p95": _percentile(process_times, 0.95),
                "max": max(process_times) if process_times else None,
            },
            "deadlineMs": chunk_ms,
            "deadlineMisses": deadline_misses,
            "deadlineMissRatio": deadline_misses / total_calls if total_calls else 0.0,
            "worstCall": worst_call_index,
            "worstOverrunMs": max(0.0, worst_process_ms - chunk_ms),
            "status": _portable_status(status),
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        _request(process, Frame(SHUTDOWN, 5 + total_calls, b""))
        return 0 if finite and deadline_misses == 0 else 2
    except Exception as error:
        report = {"mode": "live-worker-soak", "error": str(error)}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2
    finally:
        if process.poll() is None:
            process.kill()
        _, stderr = process.communicate(timeout=10)
        if process.returncode not in (0, None) and stderr:
            print(stderr.decode("utf-8", errors="replace"), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

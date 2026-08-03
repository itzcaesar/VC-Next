from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from vc_next_sidecar.framed_protocol import (
    AUDIO_REQUEST,
    AUDIO_RESPONSE,
    ERROR_RESPONSE,
    JSON_REQUEST,
    JSON_RESPONSE,
    SHUTDOWN,
    Frame,
    read_frame,
    write_frame,
)
from vc_next_sidecar.live_worker import LIVE_INPUT_SAMPLE_RATE


def _read_input(path: str | None, frames: int) -> np.ndarray:
    if path is None:
        return np.zeros(frames, dtype=np.float32)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.mean(audio, axis=1, dtype=np.float32))
    if sample_rate != LIVE_INPUT_SAMPLE_RATE:
        waveform = audio_functional.resample(
            waveform, sample_rate, LIVE_INPUT_SAMPLE_RATE
        )
    if waveform.numel() < frames:
        waveform = torch.nn.functional.pad(
            waveform, (0, frames - waveform.numel())
        )
    return waveform[:frames].contiguous().numpy()


def _request(process: subprocess.Popen[bytes], frame: Frame) -> Frame:
    assert process.stdin is not None
    assert process.stdout is not None
    write_frame(process.stdin, frame)
    response = read_frame(process.stdout)
    if response.request_id != frame.request_id:
        raise RuntimeError("The live worker returned a mismatched request ID.")
    if response.kind == ERROR_RESPONSE:
        error = json.loads(response.payload)
        raise RuntimeError(error.get("error", "The live worker rejected the request."))
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the persistent live RVC worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--index")
    parser.add_argument(
        "--contentvec",
        help="Optional explicit ContentVec .onnx or Fairseq HuBERT .pt/.pth feature embedder; otherwise discover ContentVec from the package.",
    )
    parser.add_argument(
        "--rmvpe",
        help="Optional explicit RMVPE ONNX asset; otherwise discover it from the package.",
    )
    parser.add_argument("--index-ratio", type=float, default=0.0)
    parser.add_argument("--protect-ratio", type=float, default=0.33)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--f0-threshold", type=float, default=0.30)
    parser.add_argument(
        "--use-package-defaults",
        action="store_true",
        help="Let the worker import pitch/index/protection/chunk/embedder defaults from params.json.",
    )
    parser.add_argument(
        "--streaming-preset",
        choices=("quality", "balanced", "latency"),
        default="balanced",
    )
    parser.add_argument("--chunks", type=int, default=4)
    args = parser.parse_args()
    if args.chunks < 2:
        parser.error("--chunks must be at least 2 so the SOLA stream can prime.")

    process = subprocess.Popen(
        [sys.executable, "-m", "vc_next_sidecar", "--worker"],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        handshake = _request(
            process,
            Frame(JSON_REQUEST, 1, b'{"method":"handshake","params":{}}'),
        )
        if handshake.kind != JSON_RESPONSE:
            raise RuntimeError("The live worker handshake returned the wrong frame kind.")

        load_params: dict[str, object] = {
            "modelPath": args.model,
            "streamingPreset": args.streaming_preset,
        }
        if args.use_package_defaults:
            # An explicit index can still be supplied to override package discovery,
            # while all model-tuning fields remain metadata-driven.
            if args.index:
                load_params["indexPath"] = args.index
        else:
            load_params.update(
                {
                    "indexPath": args.index,
                    "pitchShift": args.pitch_shift,
                    "indexRatio": args.index_ratio,
                    "protectRatio": args.protect_ratio,
                    "speakerId": args.speaker_id,
                    "f0Threshold": args.f0_threshold,
                }
            )
        if args.contentvec:
            load_params["contentvecPath"] = args.contentvec
        if args.rmvpe:
            load_params["rmvpePath"] = args.rmvpe
        load_payload = json.dumps(
            {"method": "load_model", "params": load_params},
            separators=(",", ":"),
        ).encode("utf-8")
        load_started = perf_counter()
        loaded = _request(process, Frame(JSON_REQUEST, 2, load_payload))
        load_ms = (perf_counter() - load_started) * 1_000.0
        status = json.loads(loaded.payload)
        chunk_frames = int(status["chunkFrames"])

        source = _read_input(args.input, chunk_frames * args.chunks)
        converted_chunks: list[np.ndarray] = []
        process_times: list[float] = []
        for chunk_index in range(args.chunks):
            chunk = source[
                chunk_index * chunk_frames : (chunk_index + 1) * chunk_frames
            ]
            process_started = perf_counter()
            converted_frame = _request(
                process,
                Frame(
                    AUDIO_REQUEST,
                    3 + chunk_index,
                    chunk.astype("<f4").tobytes(),
                ),
            )
            process_times.append((perf_counter() - process_started) * 1_000.0)
            if converted_frame.kind != AUDIO_RESPONSE:
                raise RuntimeError("The live worker returned the wrong audio frame kind.")
            converted_chunk = np.frombuffer(
                converted_frame.payload, dtype="<f4"
            ).copy()
            if (
                converted_chunk.shape[0] != chunk_frames
                or not np.isfinite(converted_chunk).all()
            ):
                raise RuntimeError("The live worker returned invalid audio.")
            converted_chunks.append(converted_chunk)
        converted = np.concatenate(converted_chunks)
        final_status = _request(
            process,
            Frame(
                JSON_REQUEST,
                3 + args.chunks,
                b'{"method":"status","params":{}}',
            ),
        )
        status = json.loads(final_status.payload)
        if args.output:
            destination = Path(args.output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, converted, LIVE_INPUT_SAMPLE_RATE, subtype="PCM_16")

        print(
            json.dumps(
                {
                    "handshake": json.loads(handshake.payload),
                    "status": status,
                    "loadRoundTripMs": round(load_ms, 1),
                    "audioRoundTripMs": [round(value, 1) for value in process_times],
                    "averageAudioRoundTripMs": round(
                        sum(process_times) / len(process_times), 1
                    ),
                    "deadlineMet": max(process_times) < status["chunkMilliseconds"],
                    "chunks": args.chunks,
                    "frames": int(converted.shape[0]),
                    "finite": True,
                    "peak": round(float(np.max(np.abs(converted))), 6),
                    "primingPeak": round(
                        float(np.max(np.abs(converted_chunks[0]))), 6
                    ),
                    "streamingPeak": round(
                        float(np.max(np.abs(converted[chunk_frames:]))), 6
                    ),
                },
                indent=2,
            )
        )
        _request(process, Frame(SHUTDOWN, 4 + args.chunks, b""))
        return 0
    finally:
        if process.poll() is None:
            process.kill()
        _, stderr = process.communicate(timeout=10)
        if process.returncode not in (0, None) and stderr:
            print(stderr.decode("utf-8", errors="replace"), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

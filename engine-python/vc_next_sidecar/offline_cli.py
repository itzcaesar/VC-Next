from __future__ import annotations

import argparse
import json

from .rvc_compat.offline import convert_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one trusted offline RVC conversion")
    parser.add_argument("--input", required=True, help="source audio file")
    parser.add_argument("--output", required=True, help="destination WAV file")
    parser.add_argument("--model", required=True, help="trusted RVC .pth checkpoint")
    parser.add_argument(
        "--contentvec",
        required=True,
        help="ContentVec .onnx or Fairseq HuBERT .pt/.pth feature embedder",
    )
    parser.add_argument("--rmvpe", required=True, help="RMVPE ONNX model")
    parser.add_argument("--index", help="optional sibling FAISS .index file")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--index-ratio", type=float, default=0.0)
    parser.add_argument("--protect-ratio", type=float, default=0.5)
    parser.add_argument("--f0-threshold", type=float, default=0.30)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()

    result = convert_file(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        contentvec_path=args.contentvec,
        rmvpe_path=args.rmvpe,
        index_path=args.index,
        speaker_id=args.speaker_id,
        pitch_shift=args.pitch_shift,
        index_ratio=args.index_ratio,
        protect_ratio=args.protect_ratio,
        f0_threshold=args.f0_threshold,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

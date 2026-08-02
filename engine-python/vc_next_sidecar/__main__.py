from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from .protocol import ProtocolError, error_response, handle_request


def _serve(input_stream: TextIO, output_stream: TextIO, once: bool) -> int:
    for line in input_stream:
        if not line.strip():
            continue
        request_id = "unknown"
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ProtocolError("invalid_request", "A request must be a JSON object.")
            request_id = str(request.get("requestId", "unknown"))
            response = handle_request(request)
        except json.JSONDecodeError as error:
            response = error_response(
                request_id,
                ProtocolError("invalid_json", f"Invalid JSON request: {error.msg}."),
            )
        except ProtocolError as error:
            response = error_response(request_id, error)
        except Exception as error:  # keep the protocol boundary stable on unexpected failures
            response = error_response(
                request_id,
                ProtocolError("internal_error", f"Sidecar request failed: {error}"),
            )

        output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
        output_stream.flush()
        if once:
            return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VC Next local inference sidecar")
    parser.add_argument("--once", action="store_true", help="process one request and exit")
    parser.add_argument("--worker", action="store_true", help="run the persistent framed live worker")
    args = parser.parse_args()
    if args.worker:
        from .live_worker import run_worker

        return run_worker()
    return _serve(sys.stdin, sys.stdout, args.once)


if __name__ == "__main__":
    raise SystemExit(main())

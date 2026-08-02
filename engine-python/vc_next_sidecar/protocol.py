from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import ENGINE_VERSION, PROTOCOL_VERSION
from .checkpoint_probe import inspect_trusted_checkpoint
from .model_probe import inspect_model
from .runtime import probe_runtime


@dataclass(frozen=True)
class ProtocolError(Exception):
    code: str
    message: str


def _handshake() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "engineVersion": ENGINE_VERSION,
        "methods": [
            "handshake",
            "probe_runtime",
            "inspect_model",
            "inspect_trusted_checkpoint",
        ],
        "transport": "stdio-json-lines",
        "audioTransport": "not-connected",
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    request_id = str(request.get("requestId", "unknown"))
    version = request.get("protocolVersion")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            "protocol_version_mismatch",
            f"Expected protocol version {PROTOCOL_VERSION}, received {version!r}.",
        )

    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        raise ProtocolError("invalid_params", "Request params must be an object.")

    if method == "handshake":
        result = _handshake()
    elif method == "probe_runtime":
        result = probe_runtime()
    elif method == "inspect_model":
        model_path = params.get("path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ProtocolError("invalid_model_path", "A non-empty model path is required.")
        try:
            result = inspect_model(model_path)
        except ValueError as error:
            raise ProtocolError("model_inspection_failed", str(error)) from error
    elif method == "inspect_trusted_checkpoint":
        model_path = params.get("path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ProtocolError("invalid_model_path", "A non-empty model path is required.")
        try:
            result = inspect_trusted_checkpoint(model_path)
        except ValueError as error:
            raise ProtocolError("checkpoint_inspection_failed", str(error)) from error
    else:
        raise ProtocolError("unknown_method", f"Unsupported sidecar method: {method!r}.")

    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "ok": True,
        "result": result,
        "error": None,
    }


def error_response(request_id: str, error: ProtocolError) -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "ok": False,
        "result": None,
        "error": {"code": error.code, "message": error.message},
    }

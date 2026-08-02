from __future__ import annotations

from dataclasses import dataclass
import io
import struct
from typing import BinaryIO


MAGIC = b"VCN1"
PROTOCOL_VERSION = 1
HEADER = struct.Struct("<4sB3xII")
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024

JSON_REQUEST = 1
JSON_RESPONSE = 2
AUDIO_REQUEST = 3
AUDIO_RESPONSE = 4
ERROR_RESPONSE = 5
SHUTDOWN = 6


@dataclass(frozen=True)
class Frame:
    kind: int
    request_id: int
    payload: bytes


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError("The framed sidecar stream ended unexpectedly.")
        data.extend(chunk)
    return bytes(data)


def read_frame(stream: BinaryIO) -> Frame:
    header = _read_exact(stream, HEADER.size)
    magic, kind, request_id, payload_size = HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError("The live sidecar frame magic is invalid.")
    if payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError("The live sidecar frame exceeds the payload limit.")
    return Frame(kind, request_id, _read_exact(stream, payload_size))


def write_frame(stream: BinaryIO, frame: Frame) -> None:
    if len(frame.payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("The live sidecar frame exceeds the payload limit.")
    stream.write(HEADER.pack(MAGIC, frame.kind, frame.request_id, len(frame.payload)))
    stream.write(frame.payload)
    stream.flush()


def encode_frame(frame: Frame) -> bytes:
    stream = io.BytesIO()
    write_frame(stream, frame)
    return stream.getvalue()


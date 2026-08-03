from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT / "tools"))

from record_device import record_device  # noqa: E402


class _FakeInputStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]

    def __enter__(self):
        try:
            self.callback(np.ones((4, 1), dtype=np.float32), 4, None, None)
        except _FakeInputStream.callback_stop:
            pass
        return self

    def __exit__(self, *_args):
        return False


class RecordDeviceTests(unittest.TestCase):
    def test_stop_marker_flushes_short_capture_and_writes_ready_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capture.wav"
            ready = root / "capture.ready.json"
            stop = root / "capture.stop"
            stop.touch()

            class CallbackStop(Exception):
                pass

            _FakeInputStream.callback_stop = CallbackStop
            with patch("record_device.sd.InputStream", _FakeInputStream):
                with patch("record_device.sd.CallbackStop", CallbackStop):
                    with patch("record_device.resolve_input_device", return_value=3):
                        with patch("record_device.wasapi_shared_settings", return_value=None):
                            summary = record_device(
                                device="Cable output",
                                output=output,
                                seconds=30,
                                sample_rate=48_000,
                                block_size=480,
                                ready_file=ready,
                                stop_file=stop,
                            )

            self.assertTrue(summary["ok"])
            self.assertEqual(summary["capturedFrames"], 4)
            self.assertEqual(summary["peak"], 1.0)
            self.assertTrue(ready.is_file())
            with wave.open(str(output), "rb") as wav:
                self.assertEqual(wav.getframerate(), 48_000)
                self.assertEqual(wav.getnframes(), 4)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT / "tools"))

from playback_fixture import resolve_output_device  # noqa: E402


class PlaybackFixtureTests(unittest.TestCase):
    def test_duplicate_windows_endpoint_names_prefer_wasapi(self) -> None:
        devices = [
            {"name": "Cable Input", "max_output_channels": 2, "hostapi": 0},
            {"name": "Cable Input", "max_output_channels": 2, "hostapi": 1},
        ]
        hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
        with patch("playback_fixture.sd.query_devices", return_value=devices):
            with patch("playback_fixture.sd.query_hostapis", side_effect=hostapis.__getitem__):
                self.assertEqual(resolve_output_device("Cable Input"), 1)


if __name__ == "__main__":
    unittest.main()

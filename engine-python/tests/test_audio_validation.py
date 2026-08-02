from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT / "tools"))

from audio_validation import (  # noqa: E402
    build_impulse_schedule,
    capture_timeout_seconds,
    match_impulses,
    percentile,
    resolve_stream_device,
    summarize_matches,
    validate_capture_devices,
    wasapi_extra_settings,
)


class AudioValidationTests(unittest.TestCase):
    def test_percentile_interpolates_and_handles_empty(self) -> None:
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([4.0], 0.95), 4.0)
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)

    def test_impulse_matching_reports_delay_and_rejection(self) -> None:
        playback = np.zeros(2_000, dtype=np.float32)
        playback[400] = 0.8
        playback[1_400] = 0.8
        recorded = np.zeros_like(playback)
        recorded[460] = 0.7
        matches = match_impulses(
            playback,
            recorded,
            sample_rate=1_000,
            search_after_ms=200,
        )
        self.assertEqual([match.delay_frames for match in matches], [60, None])
        summary = summarize_matches(matches, 1_000)
        self.assertEqual(summary["detected"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["delayMs"]["p50"], 60.0)

    def test_exact_impulse_count_extends_capture_window(self) -> None:
        total_frames, positions = build_impulse_schedule(
            total_frames=48_000,
            block_size=480,
            interval_frames=48_000,
            impulse_count=100,
        )
        self.assertEqual(len(positions), 100)
        self.assertEqual(positions[0], 480)
        self.assertEqual(positions[-1], 4_752_480)
        self.assertGreaterEqual(total_frames, positions[-1] + 480)

    def test_default_impulse_schedule_preserves_requested_window(self) -> None:
        total_frames, positions = build_impulse_schedule(
            total_frames=100_000,
            block_size=480,
            interval_frames=48_000,
            impulse_count=None,
        )
        self.assertEqual(total_frames, 100_000)
        self.assertEqual(positions, [480, 48_480, 96_480])

    def test_capture_timeout_has_runtime_slack_and_rejects_invalid_geometry(self) -> None:
        self.assertAlmostEqual(capture_timeout_seconds(48_000, 48_000), 5.0)
        with self.assertRaises(ValueError):
            capture_timeout_seconds(0, 48_000)

    def test_wdmks_full_duplex_probe_fails_before_open(self) -> None:
        class FakeSoundDevice:
            @staticmethod
            def query_devices(index):
                return {"hostapi": 3}

            @staticmethod
            def query_hostapis(_index):
                return {"name": "Windows WDM-KS"}

        with self.assertRaisesRegex(RuntimeError, "WDM-KS"):
            validate_capture_devices(FakeSoundDevice(), 88, 85)

    def test_named_endpoint_prefers_wasapi_instance(self) -> None:
        class FakeSoundDevice:
            @staticmethod
            def query_devices():
                return [
                    {"name": "Cable", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
                    {"name": "Cable", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0},
                ]

            @staticmethod
            def query_hostapis(index):
                return {"name": ["MME", "Windows WASAPI"][index]}

        self.assertEqual(resolve_stream_device(FakeSoundDevice(), "Cable", input_direction=True), 1)

    def test_wasapi_extra_settings_enable_shared_rate_conversion(self) -> None:
        class FakeSoundDevice:
            class WasapiSettings:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            @staticmethod
            def query_devices(index):
                return {"hostapi": index}

            @staticmethod
            def query_hostapis(index):
                return {"name": "Windows WASAPI" if index in (1, 2) else "MME"}

        settings = wasapi_extra_settings(FakeSoundDevice(), 1, 2)
        self.assertEqual(settings[0].kwargs, {"exclusive": False, "auto_convert": True})


if __name__ == "__main__":
    unittest.main()

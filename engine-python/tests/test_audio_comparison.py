import importlib.util
from pathlib import Path
import unittest

import numpy as np

_tool_path = Path(__file__).resolve().parents[1] / "tools" / "compare_audio.py"
_spec = importlib.util.spec_from_file_location("compare_audio", _tool_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load comparison tool: {_tool_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
best_lag = _module.best_lag
compare_recordings = _module.compare_recordings


class AudioComparisonTests(unittest.TestCase):
    def test_best_lag_finds_shifted_recording(self):
        time = np.arange(24_000, dtype=np.float32) / 24_000.0
        reference = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        candidate = np.concatenate((np.zeros(1_152, dtype=np.float32), reference))
        lag, score = best_lag(reference, candidate, max_lag_frames=2_000)
        self.assertEqual(lag, -1_152)
        self.assertGreater(score, 0.99)

    def test_compare_reports_gain_and_finite_output(self):
        reference = np.sin(np.linspace(0.0, 20.0, 4_000, dtype=np.float32)) * 0.2
        candidate = reference * 0.5
        report = compare_recordings(reference, candidate, sample_rate=48_000)
        self.assertTrue(report["finite"])
        self.assertGreater(report["correlation"], 0.99)
        self.assertAlmostEqual(report["gainRatio"], 0.5, places=2)
        self.assertLess(report["rmse"], 0.08)


if __name__ == "__main__":
    unittest.main()

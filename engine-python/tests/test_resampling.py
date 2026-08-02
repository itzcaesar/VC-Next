from __future__ import annotations

import unittest

import numpy as np

from vc_next_sidecar.rvc_compat.resampling import resample_kaiser_fast


class ResamplingTests(unittest.TestCase):
    def test_same_rate_returns_contiguous_copy(self) -> None:
        source = np.linspace(-1.0, 1.0, 31, dtype=np.float32)
        converted = resample_kaiser_fast(source, 48_000, 48_000)
        self.assertEqual(converted.dtype, np.float32)
        self.assertTrue(converted.flags.c_contiguous)
        self.assertTrue(np.array_equal(converted, source))
        self.assertIsNot(converted, source)

    def test_kaiser_fast_resample_has_expected_rvcv2_shape(self) -> None:
        time = np.arange(33_120, dtype=np.float32) / 48_000.0
        source = (0.2 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
        converted = resample_kaiser_fast(source, 48_000, 16_000)
        self.assertEqual(converted.shape, (11_040,))
        self.assertEqual(converted.dtype, np.float32)
        self.assertTrue(converted.flags.c_contiguous)
        self.assertTrue(np.isfinite(converted).all())
        self.assertGreater(float(np.max(np.abs(converted))), 0.05)


if __name__ == "__main__":
    unittest.main()

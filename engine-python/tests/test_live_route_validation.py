from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT / "tools"))

from live_route_validation import ChunkAccumulator, percentile  # noqa: E402


class LiveRouteValidationTests(unittest.TestCase):
    def test_chunk_accumulator_emits_exact_chunks_and_retains_tail(self) -> None:
        accumulator = ChunkAccumulator(4)
        self.assertEqual(accumulator.push(np.array([1, 2], dtype=np.float32)), [])
        chunks = accumulator.push(np.array([3, 4, 5, 6, 7], dtype=np.float32))
        self.assertEqual([chunk.tolist() for chunk in chunks], [[1, 2, 3, 4]])
        padded = accumulator.finish()
        self.assertEqual([chunk.tolist() for chunk in padded], [[5, 6, 7, 0]])

    def test_chunk_accumulator_finish_pads_only_the_tail(self) -> None:
        accumulator = ChunkAccumulator(4)
        accumulator.push(np.array([1, 2, 3], dtype=np.float32))
        padded = accumulator.finish()
        self.assertEqual([chunk.tolist() for chunk in padded], [[1, 2, 3, 0]])
        self.assertEqual(accumulator.finish(), [])

    def test_percentile_returns_interpolated_values(self) -> None:
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([2.0, 4.0], 0.5), 3.0)


if __name__ == "__main__":
    unittest.main()

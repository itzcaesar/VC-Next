from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest


ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from vc_next_sidecar.framed_protocol import (  # noqa: E402
    AUDIO_REQUEST,
    HEADER,
    JSON_REQUEST,
    MAGIC,
    Frame,
    encode_frame,
    read_frame,
)


class FramedProtocolTests(unittest.TestCase):
    def test_frame_round_trip_preserves_binary_payload(self) -> None:
        expected = Frame(AUDIO_REQUEST, 42, b"\x00\x01\x02\xff")
        actual = read_frame(io.BytesIO(encode_frame(expected)))
        self.assertEqual(actual, expected)

    def test_json_and_audio_frames_have_distinct_kinds(self) -> None:
        self.assertNotEqual(JSON_REQUEST, AUDIO_REQUEST)

    def test_invalid_magic_is_rejected(self) -> None:
        encoded = bytearray(encode_frame(Frame(JSON_REQUEST, 1, b"{}")))
        encoded[:4] = b"BAD!"
        with self.assertRaises(ValueError):
            read_frame(io.BytesIO(encoded))

    def test_truncated_payload_is_rejected(self) -> None:
        encoded = encode_frame(Frame(JSON_REQUEST, 1, b"{}"))[:-1]
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(encoded))

    def test_header_is_fixed_width(self) -> None:
        self.assertEqual(HEADER.size, 16)
        self.assertEqual(MAGIC, b"VCN1")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class AssetDiscoveryTests(unittest.TestCase):
    def test_w_okada_layout_is_discovered_above_model_dir(self) -> None:
        from vc_next_sidecar.live_worker import discover_feature_models

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model_dir" / "8" / "voice.pth"
            contentvec = root / "modules" / "contentvec" / "contentvec-f.onnx"
            rmvpe = root / "modules" / "rmvpe" / "rmvpe_20231006.onnx"
            model.parent.mkdir(parents=True)
            contentvec.parent.mkdir(parents=True)
            rmvpe.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            contentvec.write_bytes(b"content")
            rmvpe.write_bytes(b"pitch")

            actual_contentvec, actual_rmvpe = discover_feature_models(str(model))

        self.assertEqual(actual_contentvec, str(contentvec.resolve()))
        self.assertEqual(actual_rmvpe, str(rmvpe.resolve()))

    def test_handshake_reports_the_streaming_shape(self) -> None:
        import json

        from vc_next_sidecar.live_worker import LiveRvcProcessor, _control

        result, should_stop = _control(
            LiveRvcProcessor(),
            json.dumps({"method": "handshake", "params": {}}).encode("utf-8"),
        )

        self.assertFalse(should_stop)
        self.assertEqual(result["chunkFrames"], 9_600)
        self.assertEqual(result["analysisFrames"], 24_000)
        self.assertEqual(result["crossfadeFrames"], 1_920)
        self.assertEqual(result["solaSearchFrames"], 576)

    def test_streaming_presets_have_valid_distinct_shapes(self) -> None:
        from vc_next_sidecar.stream_config import STREAM_PROFILES, get_stream_profile

        self.assertLess(
            STREAM_PROFILES["latency"].chunk_frames,
            STREAM_PROFILES["balanced"].chunk_frames,
        )
        self.assertGreater(
            STREAM_PROFILES["quality"].analysis_frames,
            STREAM_PROFILES["balanced"].analysis_frames,
        )
        for profile in STREAM_PROFILES.values():
            self.assertEqual(profile.chunk_frames % 480, 0)
            self.assertLess(
                profile.chunk_frames
                + profile.crossfade_frames
                + profile.sola_search_frames,
                profile.analysis_frames,
            )
        with self.assertRaises(ValueError):
            get_stream_profile("turbo")

    def test_f0_threshold_range_is_validated(self) -> None:
        from vc_next_sidecar.stream_config import validate_f0_threshold

        self.assertEqual(validate_f0_threshold(0.03), 0.03)
        for value in (0.0, 0.21, float("nan")):
            with self.assertRaises(ValueError):
                validate_f0_threshold(value)

    def test_custom_stream_shape_is_validated_and_stitch_safe(self) -> None:
        from vc_next_sidecar.stream_config import get_stream_profile

        profile = get_stream_profile(
            "balanced", chunk_frames=49_152, extra_frames=3_840
        )
        self.assertEqual(profile.chunk_frames, 49_152)
        self.assertGreaterEqual(
            profile.analysis_frames,
            profile.chunk_frames
            + profile.crossfade_frames
            + profile.sola_search_frames,
        )
        with self.assertRaises(ValueError):
            get_stream_profile("balanced", chunk_frames=479)

    def test_pitch_shift_accepts_wide_range(self) -> None:
        from vc_next_sidecar.stream_config import validate_pitch_shift

        self.assertEqual(validate_pitch_shift(-50), -50.0)
        self.assertEqual(validate_pitch_shift(50), 50.0)
        for value in (-50.1, 50.1, float("nan")):
            with self.assertRaises(ValueError):
                validate_pitch_shift(value)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("faiss"),
    "NumPy and FAISS are not installed",
)
class RetrievalIndexTests(unittest.TestCase):
    def test_inverse_distance_weights_are_normalized(self) -> None:
        import numpy as np

        from vc_next_sidecar.rvc_compat.retrieval import inverse_distance_weights

        weights = inverse_distance_weights(
            np.asarray(
                [[0.0, 1.0, np.inf], [2.0, 2.0, 2.0]], dtype=np.float32
            )
        )

        np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-6)
        self.assertGreater(float(weights[0, 0]), float(weights[0, 1]))
        self.assertEqual(float(weights[0, 2]), 0.0)

    def test_flat_index_loads_and_blends_features(self) -> None:
        import faiss
        import numpy as np

        from vc_next_sidecar.rvc_compat.retrieval import FaissFeatureIndex

        vectors = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.index"
            index = faiss.IndexFlatL2(3)
            index.add(vectors)
            faiss.write_index(index, str(path))
            loaded = FaissFeatureIndex.load(str(path), expected_dimension=3)
            source = np.asarray([[[0.95, 0.05, 0.0]]], dtype=np.float32)
            blended = loaded.blend(source, 1.0)

        self.assertEqual(loaded.vector_count, 3)
        self.assertEqual(loaded.dimension, 3)
        self.assertEqual(blended.shape, source.shape)
        self.assertGreater(float(blended[0, 0, 0]), float(blended[0, 0, 1]))

    def test_index_dimension_mismatch_is_rejected(self) -> None:
        import faiss
        import numpy as np

        from vc_next_sidecar.rvc_compat.retrieval import FaissFeatureIndex

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.index"
            index = faiss.IndexFlatL2(3)
            index.add(np.ones((1, 3), dtype=np.float32))
            faiss.write_index(index, str(path))
            with self.assertRaisesRegex(ValueError, "does not match"):
                FaissFeatureIndex.load(str(path), expected_dimension=768)

    def test_invalid_retrieval_settings_are_rejected(self) -> None:
        from vc_next_sidecar.rvc_compat.retrieval import (
            validate_index_ratio,
            validate_protect_ratio,
        )

        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                validate_index_ratio(value)
        for value in (-0.1, 0.6, float("inf")):
            with self.assertRaises(ValueError):
                validate_protect_ratio(value)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class SolaStitcherTests(unittest.TestCase):
    def test_equal_power_strengths_are_complementary(self) -> None:
        import numpy as np

        from vc_next_sidecar.streaming import equal_power_strengths

        previous, current = equal_power_strengths(1_920)

        np.testing.assert_allclose(previous + current, 1.0, atol=1e-6)
        self.assertEqual(previous.dtype, np.float32)
        self.assertEqual(current.dtype, np.float32)

    def test_first_candidate_primes_with_one_silent_hop(self) -> None:
        import numpy as np

        from vc_next_sidecar.streaming import SolaStitcher

        stitcher = SolaStitcher(hop_frames=16, overlap_frames=4, search_frames=3)
        result = stitcher.process(np.ones(stitcher.candidate_frames, dtype=np.float32))

        self.assertFalse(result.primed)
        self.assertEqual(result.audio.shape, (16,))
        self.assertEqual(float(np.max(np.abs(result.audio))), 0.0)

    def test_alignment_finds_the_repeated_previous_tail(self) -> None:
        import numpy as np

        from vc_next_sidecar.streaming import SolaStitcher

        stitcher = SolaStitcher(hop_frames=16, overlap_frames=4, search_frames=3)
        first = np.linspace(-0.8, 0.9, stitcher.candidate_frames, dtype=np.float32)
        stitcher.process(first)
        repeated_tail = (
            first[-stitcher.overlap_frames :] * stitcher.previous_strength
        )
        second = np.concatenate(
            (
                np.asarray([0.2, -0.2], dtype=np.float32),
                repeated_tail,
                np.linspace(-0.4, 0.7, 17, dtype=np.float32),
            )
        )

        result = stitcher.process(second)

        self.assertTrue(result.primed)
        self.assertEqual(result.offset_frames, 2)
        self.assertEqual(result.audio.shape, (16,))
        self.assertTrue(np.isfinite(result.audio).all())

    def test_reset_requires_priming_again(self) -> None:
        import numpy as np

        from vc_next_sidecar.streaming import SolaStitcher

        stitcher = SolaStitcher(hop_frames=16, overlap_frames=4, search_frames=3)
        candidate = np.ones(stitcher.candidate_frames, dtype=np.float32)
        stitcher.process(candidate)
        stitcher.reset()

        self.assertFalse(stitcher.process(candidate).primed)


if __name__ == "__main__":
    unittest.main()

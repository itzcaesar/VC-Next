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

    def test_common_alternate_rmvpe_name_is_discovered(self) -> None:
        from vc_next_sidecar.live_worker import discover_feature_models

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "Voice Changer" / "model" / "voice.pth"
            contentvec = root / "Voice Changer" / "main" / "modules" / "contentvec" / "contentvec.onnx"
            rmvpe = root / "Voice Changer" / "main" / "modules" / "rmvpe" / "rmvpe_onnx.onnx"
            model.parent.mkdir(parents=True)
            contentvec.parent.mkdir(parents=True)
            rmvpe.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            contentvec.write_bytes(b"content")
            rmvpe.write_bytes(b"pitch")

            actual_contentvec, actual_rmvpe = discover_feature_models(str(model))

        self.assertEqual(actual_contentvec, str(contentvec.resolve()))
        self.assertEqual(actual_rmvpe, str(rmvpe.resolve()))

    def test_rinna_hubert_embedder_folder_is_discovered(self) -> None:
        from vc_next_sidecar.live_worker import discover_feature_models

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "Voice Changer" / "model" / "voice.pth"
            contentvec = root / "Voice Changer" / "main" / "modules" / "rinna_hubert" / "rinna_hubert_base-f.onnx"
            rmvpe = root / "Voice Changer" / "main" / "modules" / "rmvpe" / "rmvpe.onnx"
            model.parent.mkdir(parents=True)
            contentvec.parent.mkdir(parents=True)
            rmvpe.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            contentvec.write_bytes(b"content")
            rmvpe.write_bytes(b"pitch")

            actual_contentvec, actual_rmvpe = discover_feature_models(str(model))

        self.assertEqual(actual_contentvec, str(contentvec.resolve()))
        self.assertEqual(actual_rmvpe, str(rmvpe.resolve()))

    def test_hubert_l12_package_hint_matches_w_okada_contentvec(self) -> None:
        import json

        from vc_next_sidecar.live_worker import discover_feature_models

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "Voice Changer" / "model_dir" / "5" / "voice.pth"
            contentvec = root / "Voice Changer" / "main" / "modules" / "contentvec" / "contentvec-f.onnx"
            rinna = root / "Voice Changer" / "main" / "modules" / "rinna_hubert" / "rinna_hubert_base-f.onnx"
            rmvpe = root / "Voice Changer" / "main" / "modules" / "rmvpe" / "rmvpe_onnx.onnx"
            model.parent.mkdir(parents=True)
            contentvec.parent.mkdir(parents=True)
            rinna.parent.mkdir(parents=True)
            rmvpe.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            contentvec.write_bytes(b"contentvec")
            rinna.write_bytes(b"rinna")
            rmvpe.write_bytes(b"pitch")
            (model.parent / "params.json").write_text(
                json.dumps({"embedder": "hubert_base_l12"}), encoding="utf-8"
            )

            actual_contentvec, actual_rmvpe = discover_feature_models(
                str(model), "hubert_base_l12"
            )

        self.assertEqual(actual_contentvec, str(contentvec.resolve()))
        self.assertEqual(actual_rmvpe, str(rmvpe.resolve()))

    def test_explicit_rinna_hint_prefers_rinna_when_both_assets_exist(self) -> None:
        from vc_next_sidecar.live_worker import discover_feature_models

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "Voice Changer" / "model_dir" / "5" / "voice.pth"
            contentvec = root / "Voice Changer" / "main" / "modules" / "contentvec" / "contentvec-f.onnx"
            rinna = root / "Voice Changer" / "main" / "modules" / "rinna_hubert" / "rinna_hubert_base-f.onnx"
            rmvpe = root / "Voice Changer" / "main" / "modules" / "rmvpe" / "rmvpe_onnx.onnx"
            model.parent.mkdir(parents=True)
            contentvec.parent.mkdir(parents=True)
            rinna.parent.mkdir(parents=True)
            rmvpe.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            contentvec.write_bytes(b"contentvec")
            rinna.write_bytes(b"rinna")
            rmvpe.write_bytes(b"pitch")

            actual_contentvec, actual_rmvpe = discover_feature_models(
                str(model), "rinna_hubert"
            )

        self.assertEqual(actual_contentvec, str(rinna.resolve()))
        self.assertEqual(actual_rmvpe, str(rmvpe.resolve()))

    def test_missing_feature_assets_report_search_locations(self) -> None:
        from vc_next_sidecar.live_worker import discover_feature_models

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "Voice Changer" / "model" / "voice.pth"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")

            with self.assertRaisesRegex(ValueError, "Missing: ContentVec \(.onnx\), RMVPE \(.onnx\)") as context:
                discover_feature_models(str(model))

        self.assertIn("searched:", str(context.exception))
        self.assertIn("main\\modules", str(context.exception))

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
        self.assertEqual(result["crossfadeFrames"], 4_096)
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

    def test_calibration_discards_profile_switch_warmup_and_restores_shape(self) -> None:
        from vc_next_sidecar.live_worker import LiveRvcProcessor

        processor = LiveRvcProcessor()
        processor.generator = object()
        processor.features = object()
        calls: list[int] = []

        def fake_convert(samples):
            calls.append(int(samples.shape[0]))
            return samples, {}

        processor._convert_analysis_window = fake_convert
        result = processor.calibrate()

        self.assertEqual(len(calls), 12)
        self.assertEqual(result["restoredPreset"], "balanced")
        self.assertEqual(processor.streaming_preset, "balanced")
        self.assertEqual(processor.chunk_frames, 9_600)
        self.assertTrue(all(profile["stable"] for profile in result["profiles"]))
        self.assertTrue(all(profile["sampleCount"] == 3 for profile in result["profiles"]))

    def test_f0_threshold_range_is_validated(self) -> None:
        from vc_next_sidecar.stream_config import validate_f0_threshold

        self.assertEqual(validate_f0_threshold(0.30), 0.30)
        for value in (0.0, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                validate_f0_threshold(value)

    def test_custom_stream_shape_is_validated_and_stitch_safe(self) -> None:
        from vc_next_sidecar.stream_config import get_stream_profile

        w_okada_shape = get_stream_profile(
            "balanced", chunk_frames=24_000, extra_frames=24_000
        )
        self.assertEqual(w_okada_shape.crossfade_frames, 4_096)
        self.assertEqual(w_okada_shape.sola_search_frames, 576)
        self.assertEqual(w_okada_shape.analysis_frames, 32_768)

        v2_shape = get_stream_profile(
            "balanced", chunk_frames=24_000, extra_frames=24_000, rvc_version="v2"
        )
        # RVCr2 converts this 48 kHz window at 16 kHz, rounds convertSize to a
        # 160-sample feature hop, then returns to the device rate: 33,120 frames.
        self.assertEqual(v2_shape.analysis_frames, 33_120)
        self.assertEqual(v2_shape.analysis_frames % 480, 0)

        profile = get_stream_profile(
            "balanced", chunk_frames=49_152, extra_frames=3_840
        )
        self.assertEqual(profile.chunk_frames, 49_152)
        self.assertEqual(profile.analysis_frames % 128, 0)
        self.assertGreaterEqual(
            profile.analysis_frames,
            profile.chunk_frames
            + profile.crossfade_frames
            + profile.sola_search_frames,
        )
        tiny = get_stream_profile("balanced", chunk_frames=3_072, extra_frames=3_840)
        self.assertLess(tiny.crossfade_frames, 4_096)
        self.assertGreaterEqual(
            tiny.analysis_frames,
            tiny.chunk_frames + tiny.crossfade_frames + tiny.sola_search_frames,
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

    def test_load_rejects_invalid_settings_before_touching_model_files(self) -> None:
        from vc_next_sidecar.live_worker import LiveRvcProcessor

        processor = LiveRvcProcessor()
        with self.assertRaisesRegex(ValueError, "Pitch shift"):
            processor.load({"modelPath": "C:/missing/voice.pth", "pitchShift": 50.1})
        self.assertEqual(processor.status()["state"], "empty")

    def test_load_reports_missing_feature_assets_clearly(self) -> None:
        from vc_next_sidecar.live_worker import LiveRvcProcessor

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.pth"
            model.write_bytes(b"checkpoint")
            processor = LiveRvcProcessor()
            with self.assertRaisesRegex(ValueError, "ContentVec and RMVPE assets"):
                processor.load({"modelPath": str(model)})
            self.assertEqual(processor.status()["state"], "empty")

    def test_silence_gate_rejects_empty_noise_floor_and_isolated_peaks(self) -> None:
        import numpy as np

        from vc_next_sidecar.live_worker import (
            is_silent_input,
            input_signal_levels,
        )

        empty = np.full(9_600, 0.00018, dtype=np.float32)
        virtual_bus_floor = np.full(9_600, 0.0016, dtype=np.float32)
        isolated_peak = np.zeros(9_600, dtype=np.float32)
        isolated_peak[4_800] = 0.01
        speech = np.zeros(9_600, dtype=np.float32)
        speech[4_000:5_000] = 0.01
        quiet_speech = np.zeros(9_600, dtype=np.float32)
        quiet_speech[4_000:5_000] = 0.005

        empty_rms, empty_peak = input_signal_levels(empty)
        self.assertAlmostEqual(empty_rms, 0.00018, places=6)
        self.assertAlmostEqual(empty_peak, 0.00018, places=6)
        self.assertTrue(is_silent_input(empty))
        self.assertTrue(is_silent_input(virtual_bus_floor))
        self.assertTrue(is_silent_input(isolated_peak))
        self.assertFalse(is_silent_input(speech))
        self.assertFalse(is_silent_input(quiet_speech))

    def test_rvc_volume_gain_matches_w_okada_formula(self) -> None:
        import numpy as np

        from vc_next_sidecar.live_worker import rvc_volume_gain

        source = np.asarray([-0.25, 0.25, 0.5, -0.5], dtype=np.float32)
        rms, gain = rvc_volume_gain(source)
        self.assertAlmostEqual(rms, float(np.sqrt(np.mean(np.square(source)))), places=6)
        self.assertAlmostEqual(gain, float(np.sqrt(rms)), places=6)

    def test_silent_live_frame_bypasses_inference_and_resets_stitch_state(self) -> None:
        import numpy as np

        from vc_next_sidecar.live_worker import LiveRvcProcessor

        processor = LiveRvcProcessor()
        processor.generator = object()
        processor.features = object()
        converted_calls: list[int] = []

        def fake_convert(samples):
            converted_calls.append(int(samples.shape[0]))
            return np.zeros_like(samples), {
                "resample": 0.0,
                "content": 0.0,
                "pitch": 0.0,
                "retrieval": 0.0,
                "generator": 0.0,
            }

        processor._convert_analysis_window = fake_convert
        silent = processor.process(np.zeros(processor.chunk_frames, dtype=np.float32))
        self.assertEqual(float(np.max(np.abs(silent))), 0.0)
        self.assertEqual(converted_calls, [])
        self.assertEqual(processor.silence_suppressed_calls, 1)
        self.assertFalse(processor.stitcher.primed)

        voiced = np.zeros(processor.chunk_frames, dtype=np.float32)
        voiced[processor.chunk_frames // 3 : processor.chunk_frames // 2] = 0.01
        processor.process(voiced)
        self.assertEqual(converted_calls, [processor.analysis_frames])

    def test_v2_history_resamples_each_live_hop_before_appending(self) -> None:
        import numpy as np
        from types import SimpleNamespace

        from vc_next_sidecar.live_worker import LiveRvcProcessor
        from vc_next_sidecar.rvc_compat.resampling import resample_kaiser_fast

        processor = LiveRvcProcessor()
        processor.generator = SimpleNamespace(rvc_version="v2")
        processor.features = object()
        processor._configure_stream(
            "balanced",
            chunk_frames=9_600,
            extra_frames=24_000,
            rvc_version="v2",
        )

        def fake_convert(samples):
            return np.zeros_like(samples), {
                "resample": processor._pending_resample_ms,
                "content": 0.0,
                "pitch": 0.0,
                "retrieval": 0.0,
                "generator": 0.0,
            }

        processor._convert_analysis_window = fake_convert
        source = np.linspace(-0.1, 0.1, processor.chunk_frames, dtype=np.float32)
        processor.process(source)

        expected = resample_kaiser_fast(source, 48_000, 16_000)
        self.assertIsNotNone(processor.feature_history)
        np.testing.assert_allclose(
            processor.feature_history[-expected.shape[0] :],
            expected,
            rtol=1e-5,
            atol=1e-6,
        )


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

    def test_flat_index_defaults_to_w_okada_nearest_neighbor_and_can_weight_neighbors(self) -> None:
        import faiss
        import numpy as np

        from vc_next_sidecar.rvc_compat.retrieval import FaissFeatureIndex

        vectors = np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.index"
            index = faiss.IndexFlatL2(2)
            index.add(vectors)
            faiss.write_index(index, str(path))
            loaded = FaissFeatureIndex.load(str(path), expected_dimension=2)
            source = np.asarray([[[0.8, 0.2]]], dtype=np.float32)
            nearest = loaded.blend(source, 1.0)
            weighted = loaded.blend(source, 1.0, neighbor_count=2)

        np.testing.assert_allclose(nearest, np.asarray([[[1.0, 0.0]]], dtype=np.float32))
        self.assertNotEqual(float(weighted[0, 0, 1]), 0.0)

    def test_retrieval_front_context_preserves_feature_length(self) -> None:
        import faiss
        import numpy as np

        from vc_next_sidecar.rvc_compat.offline import _blend_retrieval_with_silence_front
        from vc_next_sidecar.rvc_compat.retrieval import FaissFeatureIndex

        vectors = np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.index"
            index = faiss.IndexFlatL2(2)
            index.add(vectors)
            faiss.write_index(index, str(path))
            loaded = FaissFeatureIndex.load(str(path), expected_dimension=2)
            source = np.asarray(
                [[[0.0, 0.0], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]],
                dtype=np.float32,
            )
            blended = _blend_retrieval_with_silence_front(source, loaded, 1.0, 1)

        self.assertEqual(blended.shape, source.shape)
        np.testing.assert_allclose(blended[0, 0], source[0, 0])

    def test_retrieval_front_context_can_use_rolling_live_features(self) -> None:
        import faiss
        import numpy as np

        from vc_next_sidecar.rvc_compat.offline import _blend_retrieval_with_silence_front
        from vc_next_sidecar.rvc_compat.retrieval import FaissFeatureIndex

        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.index"
            index = faiss.IndexFlatL2(2)
            index.add(vectors)
            faiss.write_index(index, str(path))
            loaded = FaissFeatureIndex.load(str(path), expected_dimension=2)
            source = np.asarray(
                [[[0.0, 0.0], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]],
                dtype=np.float32,
            )
            rolling = np.asarray(
                [[9.0, 8.0], [7.0, 6.0], [5.0, 4.0], [3.0, 2.0]],
                dtype=np.float32,
            )
            blended = _blend_retrieval_with_silence_front(
                source,
                loaded,
                1.0,
                1,
                rolling,
            )

        self.assertEqual(blended.shape, source.shape)
        # The rolling prefix occupies the same slot that w-okada's
        # ``feature_buffer[:npyOffset:2]`` occupies after tail cropping.
        np.testing.assert_allclose(blended[0, 0], np.asarray([9.0, 8.0]))

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

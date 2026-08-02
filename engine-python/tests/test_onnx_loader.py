from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

# Keep Torch resident while the fake ONNX Runtime module is patched below.
# ``patch.dict(sys.modules, ...)`` restores newly imported modules when its
# scope exits; importing it once here avoids reinitializing Torch's native
# extension in the same test process.
import torch  # noqa: E402,F401

from vc_next_sidecar.rvc_compat.loader import _onnx_sample_rate, load_onnx_generator  # noqa: E402
from vc_next_sidecar.rvc_compat.offline import OnnxFeaturePipeline, _select_content_output  # noqa: E402


class _FakeInput:
    def __init__(self, name: str, shape: list[object]) -> None:
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, *, missing_feats: bool = False) -> None:
        names = ["p_len", "pitch", "pitchf", "sid"]
        if not missing_feats:
            names.insert(0, "feats")
        self._inputs = [
            _FakeInput(name, [1, "T", 768] if name == "feats" else [1])
            for name in names
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [types.SimpleNamespace(name="audio")]

    def get_providers(self):
        # Simulate a machine where the CUDA provider package is installed but
        # its native DLLs are unavailable: the loader must expose CPU clearly.
        return ["CPUExecutionProvider"]


class _FakePitchSession:
    def __init__(self) -> None:
        self.waveform = None
        self.threshold = None

    def run(self, _outputs, inputs):
        self.waveform = inputs["waveform"]
        self.threshold = inputs["threshold"]
        return [[120.0, 121.0, 122.0]]


class OnnxLoaderTests(unittest.TestCase):
    def test_sample_rate_uses_metadata_then_filename_then_baseline(self) -> None:
        self.assertEqual(_onnx_sample_rate(Path("voice.onnx"), {"sample_rate": 48_000}), 48_000)
        self.assertEqual(_onnx_sample_rate(Path("voice_v2_32k_float.onnx"), {}), 32_000)
        self.assertEqual(_onnx_sample_rate(Path("voice_export.onnx"), {}), 40_000)

    def test_contentvec_head_selection_matches_rvc_version(self) -> None:
        outputs = [
            types.SimpleNamespace(name="unit12", shape=[1, "T", 768]),
            types.SimpleNamespace(name="units9", shape=[1, "T", 256]),
        ]
        self.assertEqual(_select_content_output(outputs, 256), "units9")
        self.assertEqual(_select_content_output(outputs, 768), "unit12")

    def test_contentvec_head_selection_rejects_missing_width(self) -> None:
        outputs = [types.SimpleNamespace(name="unit12", shape=[1, "T", 768])]
        with self.assertRaisesRegex(ValueError, "256-channel"):
            _select_content_output(outputs, 256)

    def test_rmvpe_front_context_is_trimmed_and_zero_frames_are_restored(self) -> None:
        pipeline = object.__new__(OnnxFeaturePipeline)
        session = _FakePitchSession()
        pipeline.pitch_session = session
        waveform = np.ones(640, dtype=np.float32)

        result = pipeline.extract_pitch(
            waveform,
            0.30,
            silence_front_samples=320,
            output_frames=4,
        )

        self.assertEqual(session.waveform.shape, (1, 320))
        self.assertAlmostEqual(float(session.threshold[0]), 0.30, places=6)
        self.assertEqual(result.tolist(), [0.0, 120.0, 121.0, 122.0])

    def _fake_runtime(self, *, missing_feats: bool = False):
        class SessionOptions:
            log_severity_level = 0

        def inference_session(*_args, **_kwargs):
            return _FakeSession(missing_feats=missing_feats)

        return types.SimpleNamespace(
            SessionOptions=SessionOptions,
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=inference_session,
        )

    def test_loads_five_input_export_and_reports_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "voice.onnx"
            model.write_bytes(b"onnx-placeholder")
            (root / "params.json").write_text(
                json.dumps({"sample_rate": 40_000, "is_f0": True, "speakers": ["A", "B"]}),
                encoding="utf-8",
            )
            with patch.dict(sys.modules, {"onnxruntime": self._fake_runtime()}):
                loaded = load_onnx_generator(str(model))

        self.assertEqual(loaded.backend, "onnx")
        self.assertEqual(loaded.device, "cpu")
        self.assertEqual(loaded.target_sample_rate, 40_000)
        self.assertEqual(loaded.speaker_count, 2)
        self.assertEqual(loaded.feature_channels, 768)

    def test_rejects_export_without_rvc_feature_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.onnx"
            model.write_bytes(b"onnx-placeholder")
            with patch.dict(sys.modules, {"onnxruntime": self._fake_runtime(missing_feats=True)}):
                with self.assertRaisesRegex(ValueError, "feats"):
                    load_onnx_generator(str(model))


if __name__ == "__main__":
    unittest.main()

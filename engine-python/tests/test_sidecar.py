from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_ROOT))

from vc_next_sidecar.model_probe import inspect_model, model_package_defaults  # noqa: E402
from vc_next_sidecar.checkpoint_probe import inspect_trusted_checkpoint, summarize_checkpoint  # noqa: E402
from vc_next_sidecar.protocol import ProtocolError, handle_request  # noqa: E402
from vc_next_sidecar.runtime import probe_runtime  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_handshake_is_versioned(self) -> None:
        response = handle_request(
            {
                "protocolVersion": 1,
                "requestId": "test",
                "method": "handshake",
                "params": {},
            }
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["protocolVersion"], 1)
        self.assertEqual(response["result"]["transport"], "stdio-json-lines")

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            handle_request(
                {
                    "protocolVersion": 1,
                    "requestId": "test",
                    "method": "unknown",
                    "params": {},
                }
            )
        self.assertEqual(context.exception.code, "unknown_method")

    def test_runtime_probe_reports_onnx_provider_contract(self) -> None:
        result = probe_runtime()
        self.assertIn("onnxRuntime", result)
        self.assertIn("availableProviders", result["onnxRuntime"])
        self.assertIn("cudaProviderAvailable", result["onnxRuntime"])
        capability = "onnx-cuda-provider" if result["onnxRuntime"]["cudaProviderAvailable"] else "onnx-cpu-only"
        self.assertIn(capability, result["capabilities"])

    def test_once_process_returns_one_json_response(self) -> None:
        request = json.dumps(
            {
                "protocolVersion": 1,
                "requestId": "subprocess",
                "method": "probe_runtime",
                "params": {},
            }
        )
        completed = subprocess.run(
            [sys.executable, "-m", "vc_next_sidecar", "--once"],
            cwd=ENGINE_ROOT,
            input=request + "\n",
            text=True,
            capture_output=True,
            check=True,
            # Importing the CUDA/ONNX runtime can take several seconds on a cold
            # Windows process. Keep this strict enough to catch a hung sidecar,
            # without making normal first-start initialization flaky.
            timeout=30,
        )
        response = json.loads(completed.stdout)
        self.assertTrue(response["ok"])
        self.assertEqual(response["requestId"], "subprocess")
        self.assertIn("python", response["result"])


class ModelInspectionTests(unittest.TestCase):
    def test_checkpoint_is_inspected_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.pth"
            model.write_bytes(b"PK\x03\x04safe-test-placeholder")
            result = inspect_model(str(model))
        self.assertEqual(result["role"], "rvc-checkpoint")
        self.assertFalse(result["checkpointLoaded"])
        self.assertTrue(result["safeInspectionOnly"])

    def test_unsupported_extension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.exe"
            model.write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                inspect_model(str(model))

    def test_matching_sibling_index_is_recommended_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "mayaputri.pth"
            unrelated = root / "added_IVF100_Flat_other_voice.index"
            matching = root / "added_IVF892_Flat_nprobe_1_mayaputri_v2.index"
            model.write_bytes(b"checkpoint")
            unrelated.write_bytes(b"index")
            matching.write_bytes(b"index")

            result = inspect_model(str(model))

        self.assertEqual(result["recommendedIndex"], str(matching.resolve()))
        self.assertEqual(result["siblingIndexes"][0], str(matching.resolve()))
        self.assertTrue(result["packageComplete"])

    def test_nearby_w_okada_package_index_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model_dir"
            selected_slot = model_dir / "5"
            neighboring_slot = model_dir / "7"
            selected_slot.mkdir(parents=True)
            neighboring_slot.mkdir(parents=True)
            model = selected_slot / "voice.pth"
            nearby = neighboring_slot / "voice.index"
            unrelated = neighboring_slot / "other.index"
            model.write_bytes(b"checkpoint")
            nearby.write_bytes(b"index")
            unrelated.write_bytes(b"index")

            result = inspect_model(str(model))

        self.assertEqual(result["recommendedIndex"], str(nearby.resolve()))
        self.assertIn(str(nearby.resolve()), result["siblingIndexes"])
        self.assertNotIn(str(unrelated.resolve()), result["siblingIndexes"])
        self.assertIn("surrounding model package", result["pairingNote"])

    def test_index_folder_next_to_model_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_folder = root / "model"
            index_folder = model_folder / "indexes"
            index_folder.mkdir(parents=True)
            model = model_folder / "e-girl_e350_s42700.pth"
            matching = index_folder / "added_IVF1611_Flat_nprobe_1_e-girl_v2.index"
            model.write_bytes(b"checkpoint")
            matching.write_bytes(b"index")

            result = inspect_model(str(model))

        self.assertEqual(result["recommendedIndex"], str(matching.resolve()))
        self.assertIn("surrounding model package", result["pairingNote"])

    def test_checkpoint_without_index_remains_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.pth"
            model.write_bytes(b"checkpoint")
            result = inspect_model(str(model))

        self.assertIsNone(result["recommendedIndex"])
        self.assertFalse(result["packageComplete"])
        self.assertIn("can still run", result["pairingNote"])

    def test_w_okada_params_are_exposed_as_safe_model_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "voice.pth"
            model.write_bytes(b"checkpoint")
            (root / "params.json").write_text(
                json.dumps(
                    {
                        "pitch_shift": 14,
                        "index_ratio": 0.3,
                        "protect_ratio": 0.5,
                        "chunk_sec": 0.5,
                        "extraConvertSize": 65280,
                        "embedder": "hubert_base_l12",
                        "pitch_estimator": "rmvpe_onnx",
                    }
                ),
                encoding="utf-8",
            )
            result = inspect_model(str(model))
            defaults = model_package_defaults(model)

        self.assertEqual(result["modelDefaults"]["pitchShift"], 14.0)
        self.assertEqual(result["modelDefaults"]["indexRatio"], 0.3)
        self.assertEqual(result["modelDefaults"]["protectRatio"], 0.5)
        self.assertEqual(result["modelDefaults"]["chunkFrames"], 24_000)
        self.assertEqual(result["modelDefaults"]["extraFrames"], 65_280)
        self.assertEqual(defaults["embedder"], "hubert_base_l12")

    def test_invalid_or_oversized_params_do_not_block_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "voice.pth"
            model.write_bytes(b"checkpoint")
            (root / "params.json").write_text(
                json.dumps({"pitch_shift": 999, "index_ratio": "bad"}),
                encoding="utf-8",
            )
            result = inspect_model(str(model))

        self.assertEqual(result["modelDefaults"], {})

    def test_rvc_checkpoint_schema_is_summarized_without_model_code(self) -> None:
        class FakeEmbedding:
            shape = (3, 256)

        result = summarize_checkpoint(
            {
                "config": [1, 2, 3, 40_000],
                "weight": {"emb_g.weight": FakeEmbedding(), "decoder.weight": object()},
                "version": "v2",
                "f0": 1,
            }
        )
        self.assertEqual(result["rvcVersion"], "v2")
        self.assertEqual(result["targetSampleRate"], 40_000)
        self.assertEqual(result["speakerCount"], 3)
        self.assertEqual(result["weightKeyCount"], 2)

    def test_invalid_rvc_checkpoint_schema_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_checkpoint({"config": [], "weight": {}})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_restricted_loader_reads_a_tensor_only_checkpoint(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "voice.pth"
            torch.save(
                {
                    "config": [1, 2, 3, 48_000],
                    "weight": {"emb_g.weight": torch.zeros((2, 256))},
                    "version": "v2",
                    "f0": 1,
                },
                model,
            )
            result = inspect_trusted_checkpoint(str(model))

        self.assertTrue(result["checkpointLoaded"])
        self.assertFalse(result["safeInspectionOnly"])
        self.assertEqual(result["loadPolicy"], "torch-weights-only")
        self.assertEqual(result["targetSampleRate"], 48_000)
        self.assertEqual(result["speakerCount"], 2)


if __name__ == "__main__":
    unittest.main()

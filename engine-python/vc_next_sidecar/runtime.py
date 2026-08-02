from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any

from . import ENGINE_VERSION, PROTOCOL_VERSION


PACKAGE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "numpy": ("numpy",),
    "torch": ("torch",),
    "torchaudio": ("torchaudio",),
    "onnxruntime": ("onnxruntime-gpu", "onnxruntime"),
    "faiss": ("faiss-gpu", "faiss-cpu"),
    "soundfile": ("soundfile",),
    "resampy": ("resampy",),
}

# ContentVec and RMVPE are part of every live RVC session, so ONNX Runtime is
# required even when the selected generator is a PyTorch checkpoint. FAISS is
# intentionally optional because a voice can run without retrieval.
REQUIRED_PACKAGES = {"numpy", "torch", "torchaudio", "soundfile", "onnxruntime"}


def _package_version(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def probe_runtime() -> dict[str, Any]:
    packages = {
        name: _package_version(candidates)
        for name, candidates in PACKAGE_CANDIDATES.items()
    }
    python_for_rvc = sys.version_info[:2] == (3, 11)
    missing_required = sorted(
        name
        for name in REQUIRED_PACKAGES
        if packages.get(name) is None
    )
    optional_missing = sorted(
        name
        for name, version in packages.items()
        if name not in REQUIRED_PACKAGES and version is None
    )

    torch_runtime: dict[str, Any] = {
        "imported": False,
        "cudaAvailable": False,
        "cudaVersion": None,
        "deviceName": None,
        "deviceCapability": None,
        "error": None,
    }
    if packages["torch"] is not None:
        try:
            import torch

            torch_runtime["imported"] = True
            torch_runtime["cudaAvailable"] = torch.cuda.is_available()
            torch_runtime["cudaVersion"] = torch.version.cuda
            if torch.cuda.is_available():
                torch_runtime["deviceName"] = torch.cuda.get_device_name(0)
                torch_runtime["deviceCapability"] = list(torch.cuda.get_device_capability(0))
        except Exception as error:
            torch_runtime["error"] = str(error)

    onnx_runtime: dict[str, Any] = {
        "imported": False,
        "availableProviders": [],
        "cudaProviderAvailable": False,
        "error": None,
    }
    if packages["onnxruntime"] is not None:
        try:
            import onnxruntime as ort

            providers = list(ort.get_available_providers())
            onnx_runtime["imported"] = True
            onnx_runtime["availableProviders"] = providers
            onnx_runtime["cudaProviderAvailable"] = "CUDAExecutionProvider" in providers
        except Exception as error:
            onnx_runtime["error"] = str(error)

    blockers: list[str] = []
    if not python_for_rvc:
        blockers.append("Python 3.11 environment not selected")
    blockers.extend(f"{name} is not installed" for name in missing_required)
    if packages["torch"] is not None and not torch_runtime["cudaAvailable"]:
        blockers.append("CUDA is not available to PyTorch")
    if packages["onnxruntime"] is not None and not onnx_runtime["imported"]:
        blockers.append("ONNX Runtime could not be imported")
    elif packages["onnxruntime"] is not None and not onnx_runtime["cudaProviderAvailable"]:
        blockers.append("ONNX Runtime CUDA provider is unavailable")

    return {
        "source": "python-sidecar",
        "protocolVersion": PROTOCOL_VERSION,
        "engineVersion": ENGINE_VERSION,
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "sidecarCompatible": sys.version_info >= (3, 10),
            "rvcEnvironmentCompatible": python_for_rvc,
            "recommendedVersion": "3.11",
        },
        "packages": packages,
        "torchRuntime": torch_runtime,
        "onnxRuntime": onnx_runtime,
        "capabilities": [
            "runtime-probe",
            "safe-model-inspection",
            "trusted-checkpoint-inspection",
            "offline-rvc-conversion",
            "faiss-index-retrieval" if packages["faiss"] else "index-retrieval-unavailable",
            "onnx-cuda-provider" if onnx_runtime["cudaProviderAvailable"] else "onnx-cpu-only",
            "versioned-stdio-control",
        ],
        "readyForRvc": not blockers,
        "blockers": blockers,
        "optionalMissing": optional_missing,
    }

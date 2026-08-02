# Python compatibility sidecar

## Purpose

The sidecar isolates Python, PyTorch, legacy RVC compatibility code, and model loading from the Tauri host and native audio callbacks. It is not a web server and does not expose a listening port.

## Implemented checkpoint

- Versioned JSON-line control protocol over standard input/output.
- Request-ID and protocol-version validation in Rust.
- Project-local Python 3.11 virtual-environment discovery.
- Runtime and installed-package probing.
- PyTorch CUDA capability probing when Torch is installed.
- Metadata-only inspection for `.pth`, `.onnx`, and `.index` files.
- Restricted RVC checkpoint schema inspection using PyTorch's weights-only loader.
- Minimal MIT-licensed RVC v1/v2 generator compatibility loader with provenance.
- Offline CUDA ContentVec + RMVPE + RVC v2 conversion proof.
- Persistent framed binary worker protocol over standard input/output.
- Long-lived ContentVec, RMVPE, and FP16 generator sessions.
- Raw float32 little-endian mono requests and responses at 48 kHz.
- Desktop load, warm-up, status, settings, unload, and shutdown controls.
- Native Windows model picker for `.pth` and `.onnx` imports.
- No checkpoint deserialization during the initial import pass.

## Pinned development baseline

- Python 3.11.9
- NumPy 1.26.4
- SoundFile 0.12.1
- PyTorch 2.9.0 with CUDA 12.8
- Torchaudio 2.9.0 with CUDA 12.8
- ONNX Runtime GPU 1.26.0 as an optional backend

The reference machine has been verified with PyTorch 2.9.0+cu128, CUDA 12.8, and an actual synchronized tensor operation on the NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9).

FAISS retrieval is optional per voice and uses the pinned Windows `faiss-cpu` wheel. A selected index is validated against the model feature dimension, reconstructed once when the model loads, and queried with inverse-distance neighbor weighting during each conversion. Partial IVF result rows are handled explicitly instead of interpreting FAISS `-1` padding as a valid vector.

## Security boundary

Model import initially reads only path, extension, size, container hints, and sibling index paths. It does not call `torch.load`, execute pickle payloads, or instantiate model code. The separate trusted-checkpoint pass uses `torch.load(..., weights_only=True)` and then validates the RVC config, weight mapping, target sample rate, pitch flag, and speaker embedding shape. It still does not instantiate generator code.

## Live checkpoint

The live worker accepts 7,680-, 9,600-, or 12,000-frame hops (160, 200, or 250 ms at 48 kHz) with matched Low-latency, Balanced, and Quality analysis/SOLA geometries. On the RTX 4050 reference machine, the indexed 160 ms profile averaged 107 ms across five requests and every tested profile met its hop deadline. The Rust adapter learns the selected frame count from worker status and owns the subprocess on a bounded I/O thread, so native audio callbacks never read from or write to Python directly.

## Next steps

1. Reduce the 160 ms hop while measuring pitch stability, boundary artifacts, and deadline margin.
2. Add cancellation, crash recovery, and automatic worker restart.
3. Keep optional FAISS retrieval isolated from models that do not provide an index.
4. Measure physical loopback latency and run a multi-hour live soak test.

# VC Next Python sidecar

This process is the compatibility boundary for Python RVC implementations. It is deliberately separate from the native audio callbacks and communicates through a versioned local protocol over standard input and output.

The control protocol itself is dependency-light. The installed Windows/NVIDIA checkpoint supports:

- protocol handshake
- Python and package capability probing
- safe model-file inspection without deserializing checkpoints
- restricted, weights-only RVC checkpoint schema inspection
- verified PyTorch CUDA execution on the RTX 4050 reference machine

It constructs audited RVC v1/v2 generators and supports both offline conversion and a persistent framed live worker. The worker keeps ContentVec, RMVPE, and the selected generator resident, then accepts bounded preset-sized float32 PCM chunks without reopening the model. The RVC environment is isolated in a Python 3.11 virtual environment under `.venv`.

## Development environment

```powershell
py -3.11 -m venv engine-python/.venv
engine-python/.venv/Scripts/python.exe -m pip install -e engine-python
engine-python/.venv/Scripts/python.exe -m pip install -r engine-python/requirements-rvc-core.txt
engine-python/.venv/Scripts/python.exe -m pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

ONNX Runtime is an optional backend:

```powershell
engine-python/.venv/Scripts/python.exe -m pip install -r engine-python/requirements-rvc-optional.txt
```

Run one request:

```powershell
'{"protocolVersion":1,"requestId":"manual","method":"probe_runtime","params":{}}' | engine-python/.venv/Scripts/python.exe -m vc_next_sidecar --once
```

Run tests from the project root:

```powershell
engine-python/.venv/Scripts/python.exe -m unittest discover -s engine-python/tests -v
```

Run the persistent worker protocol directly:

```powershell
engine-python/.venv/Scripts/python.exe -m vc_next_sidecar --worker
```

The live worker accepts an optional sibling FAISS `.index` path, retrieval ratio, RVC protect ratio, pitch shift, RMVPE threshold, streaming preset, and speaker ID. Index vectors are loaded once with the resident model rather than reconstructed for every audio hop.

The worker speaks framed binary standard I/O and is normally owned by the Tauri host. Use `tools/live_worker_smoke.py` for a full handshake, model load, PCM exchange, and shutdown test.

The reproducible offline spike is available as:

```powershell
engine-python/.venv/Scripts/python.exe -m vc_next_sidecar.offline_cli `
  --input <speech.wav> `
  --output <converted.wav> `
  --model <voice.pth> `
  --contentvec <contentvec-f.onnx> `
  --rmvpe <rmvpe.onnx> `
  --max-seconds 3
```

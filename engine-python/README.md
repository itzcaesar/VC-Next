# VC Next Python engine

The Python package provides RVC v1/v2 model compatibility, ContentVec and RMVPE inference, optional FAISS retrieval, offline conversion, and the persistent live worker used by the Tauri desktop host.

> [!NOTE]
> This package is an internal engine component. Normal users start VC Next through `npm run tauri dev`; they do not run the worker manually.

## Responsibilities

- Probe Python, package, Torch, CUDA, and ONNX readiness
- Inspect model metadata without deserializing checkpoints
- Validate trusted `.pth` checkpoints with weights-only loading
- Construct audited RVC v1/v2 inference generators
- Keep ContentVec, RMVPE, FAISS, and generator state resident
- Convert framed 48 kHz mono PCM for the Rust host
- Resample 32/40 kHz generator output to the 48 kHz live contract
- Validate pitch, retrieval, protection, speaker, Chunk, and Extra settings
- Maintain analysis history and stateful SOLA overlap

For the full design and security boundary, read [Python compatibility sidecar](../docs/python-sidecar.md).

## Environment setup

Run from the repository root in PowerShell:

```powershell
py -3.11 -m venv engine-python/.venv
.\engine-python\.venv\Scripts\python.exe -m pip install --upgrade pip
.\engine-python\.venv\Scripts\python.exe -m pip install -e engine-python
.\engine-python\.venv\Scripts\python.exe -m pip install -r engine-python\requirements-rvc-core.txt
.\engine-python\.venv\Scripts\python.exe -m pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.\engine-python\.venv\Scripts\python.exe -m pip install -r engine-python\requirements-rvc-optional.txt
```

| Requirements file | Contents |
| --- | --- |
| `requirements-rvc-core.txt` | NumPy, SoundFile, and FAISS CPU |
| `requirements-rvc-optional.txt` | ONNX Runtime GPU |

PyTorch and Torchaudio are installed separately so the CUDA wheel source is explicit.

## Runtime probe

Send one JSON-line request:

```powershell
'{"protocolVersion":1,"requestId":"manual","method":"probe_runtime","params":{}}' |
  .\engine-python\.venv\Scripts\python.exe -m vc_next_sidecar --once
```

The response reports:

- Python version and executable;
- installed package versions;
- Torch import state;
- CUDA availability, version, device, and capability;
- synchronized CUDA execution errors;
- required blockers and optional missing components.

## Worker modes

### One-shot mode

```powershell
.\engine-python\.venv\Scripts\python.exe -m vc_next_sidecar --once
```

Used for runtime probing and model inspection. Each process handles one JSON request and exits.

### Persistent live mode

```powershell
.\engine-python\.venv\Scripts\python.exe -m vc_next_sidecar --worker
```

Used by Tauri. It speaks framed binary standard I/O and accepts:

- `handshake`
- `load_model`
- `status`
- `set_settings`
- binary audio frames
- `unload`
- `shutdown`

Do not print arbitrary text to standard output while running this mode; it would corrupt the wire protocol.

## Model and asset layout

The worker can auto-discover common w-okada-style assets:

```text
<root>/main/
├── model_dir/<slot>/voice.pth
└── modules/
    ├── contentvec/contentvec-f.onnx
    └── rmvpe/rmvpe_20231006.onnx
```

An optional matching `.index` can be selected explicitly or paired from the checkpoint directory. Model assets are external inputs and must not be committed.

## Offline conversion

```powershell
.\engine-python\.venv\Scripts\python.exe -m vc_next_sidecar.offline_cli `
  --input C:\path\speech.wav `
  --output C:\path\converted.wav `
  --model C:\path\voice.pth `
  --contentvec C:\path\contentvec-f.onnx `
  --rmvpe C:\path\rmvpe_20231006.onnx `
  --max-seconds 3
```

The offline tool is useful for isolating checkpoint and feature-pipeline problems from native device routing.

## Live-worker smoke test

```powershell
.\engine-python\.venv\Scripts\python.exe engine-python\tools\live_worker_smoke.py --help
```

The tool covers handshake, model load, PCM exchange, status, and shutdown without starting the Tauri interface.

## Tests

```powershell
.\engine-python\.venv\Scripts\python.exe -m unittest discover -s engine-python\tests -p "test_*.py" -v
```

The suite covers JSON and binary protocols, runtime/model probes, stream configuration, pitch bounds, SOLA, retrieval helpers, and compatibility error paths. Tests requiring installed numeric/ML dependencies skip cleanly outside the full `.venv`; the verified environment runs the complete suite.

## Security rules

- Initial import must remain metadata-only.
- Trusted checkpoint loading must use `weights_only=True`.
- State dictionaries must match the inference architecture exactly.
- Standard output belongs exclusively to the active protocol.
- Frame sizes and request identities must be validated.
- User models, indexes, recordings, and feature assets stay outside Git.
- Upstream-derived files require provenance and notices.

## Source provenance

The minimal RVC generator compatibility files are documented in:

- [PROVENANCE.md](vc_next_sidecar/rvc_compat/PROVENANCE.md)
- [UPSTREAM_LICENSE](vc_next_sidecar/rvc_compat/UPSTREAM_LICENSE)
- [Upstream assessment](../docs/upstream-assessment.md)

## Related documentation

- [Project README](../README.md)
- [Architecture](../docs/architecture.md)
- [Offline RVC proof](../docs/offline-rvc-spike.md)
- [Persistent live RVC](../docs/live-rvc-spike.md)

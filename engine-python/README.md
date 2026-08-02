# VC Next Python engine

The Python package provides RVC v1/v2 PyTorch and exported five-input ONNX model compatibility, ContentVec and RMVPE inference, optional FAISS retrieval, offline conversion, and the persistent live worker used by the Tauri desktop host. Its live compatibility path defaults to the current w-okada nearest-vector (`k=1`) retrieval rule, source-volume matching, the RVC-v2 16 kHz feature-window geometry, the `resampy` `kaiser_fast` boundary filter, the upstream RMVPE ONNX threshold (`0.30`), and an RMS-plus-activity idle gate so decoder bias cannot become static without cutting off a short quiet syllable.

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
| `requirements-rvc-core.txt` | NumPy, SoundFile, FAISS CPU, and the w-okada-compatible `resampy` filter |
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
- ONNX Runtime availability and execution providers;
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
- `calibrate`
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

### w-okada package defaults

When a checkpoint has a sibling `params.json`, `load_model` safely imports its
model-specific compatibility settings when the request does not provide an
explicit value:

| Metadata | Worker setting |
| --- | --- |
| `pitch_shift` | `pitchShift` (−50…+50 semitones) |
| `index_ratio` | `indexRatio` (0…1) |
| `protect_ratio` | `protectRatio` (0…0.5) |
| `chunk_sec` | `chunkFrames` at the 48 kHz live rate |
| `embedder` | feature-asset preference, including `hubert_base_l12` |
| matching sibling `.index` | `recommendedIndex` when retrieval is enabled |

The UI sends explicit controls after import, so user changes always take
precedence. The worker also honors these defaults for direct framed-protocol
clients. Invalid or oversized metadata is ignored rather than blocking model
inspection or load.

Before RVC inference, the worker applies an idle-input gate. A hop at or below
the `0.002` RMS floor returns an exact zero hop when fewer than 2% of samples
show concentrated activity; this keeps isolated virtual-device spikes from
waking the decoder while allowing a short quiet syllable through. The worker
resets stream history at the silence/speech boundary and increments
`silenceSuppressedCalls`. The native Rust route repeats the same decision at
the complete live-block boundary as a last-resort mute during device startup
or worker recovery.
Calibration and warm-up disable the gate so their timings include real model
work.

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

Use `--use-package-defaults` to exercise a w-okada package's `params.json`
instead of supplying manual pitch/index/Protect values. With an empty input,
the smoke report should show `peak: 0.0` and a positive
`silenceSuppressedCalls` count.

## Tests

```powershell
.\engine-python\.venv\Scripts\python.exe -m unittest discover -s engine-python\tests -p "test_*.py" -v
```

The suite covers JSON and binary protocols, runtime/model probes, w-okada
package metadata and embedder selection, idle-input suppression, stream
configuration, pitch bounds, SOLA, retrieval helpers, and compatibility error
paths. Tests requiring installed numeric/ML dependencies skip cleanly outside
the full `.venv`; the verified environment runs the complete suite.

## Hardware validation harness

The optional `tools/audio_validation.py` script measures a physical or virtual loopback route and can run a long callback soak. It uses `sounddevice`, which is deliberately separate from the normal engine requirements:

```powershell
.\engine-python\.venv\Scripts\python.exe -m pip install -r engine-python\requirements-audio-validation.txt
.\engine-python\.venv\Scripts\python.exe engine-python\tools\audio_validation.py --mode loopback --seconds 30 --impulse-count 100 --report outputs\loopback.json
```

The loopback mode can emit an exact impulse count and reports detected P50/P95/min/max delay. Soak mode records callback warnings, finite-sample status, and signal statistics without pretending to certify the VC Next conversion pipeline. For converted-worker timing, use `tools/live_worker_soak.py`; `--chunk-frames 10560 --extra-frames 25920` evaluates the measured intermediate 220/540 ms safety profile. Add `--realtime` when the worker should be paced to the 48 kHz audio timeline; reports then distinguish simulated audio duration from wall-clock duration.

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

# VC Next

VC Next is a Windows-first feasibility prototype for a modern, local, low-latency RVC voice changer. It is a new application architecture, not a permanent fork of w-okada/voice-changer.

## Current checkpoint: persistent live RVC prototype

- Reference platform: Windows 11
- Reference GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM
- UI: React 19, TypeScript, Vite
- Desktop host: Tauri 2
- Engine direction: native audio core with swappable RVC compatibility backends

The Tauri host exposes native Windows audio-device discovery and routes audio through a dedicated inference worker between capture and playback. Fixed-capacity lock-free buffers isolate both audio callbacks from worker scheduling. When no voice is loaded the worker uses a no-op passthrough backend; after an imported RVC v2 checkpoint is loaded, it switches to the persistent Python RVC backend.

The local Python compatibility sidecar is now connected through a versioned standard-I/O control protocol. Its isolated Python 3.11 environment uses PyTorch 2.9.0+cu128 and ONNX Runtime GPU 1.26.0 on the RTX 4050. It reports runtime readiness, safely inspects `.pth` and `.onnx` files without deserializing them during ordinary import, and offers a separate restricted weights-only checkpoint validator.

A representative local RVC v2 model now loads and warms from the desktop UI, remains resident on the GPU, and processes raw float32 PCM through a bounded binary protocol between Rust and Python. Quality, Balanced, and Low-latency modes now select real stream geometries rather than changing labels: 250/600 ms, 200/500 ms, and 160/400 ms hop/analysis windows. FAISS retrieval, index strength, RVC consonant protection, pitch, RMVPE threshold, and speaker ID are connected through the live settings contract. Native input/output/monitor gain and a smoothed noise gate run inside the Rust audio path. Playback priming adapts after underruns, and bounded sample-slip correction keeps independent Output and Monitor clocks from accumulating unbounded queue drift. With the real Mayaputri `IndexIVFFlat` (34,823 × 768), all three indexed CUDA profiles stayed inside their hop deadlines; the 160 ms profile averaged 107 ms across five passes. Continuous resampling, physical loopback measurements, and extended soak testing remain pending.

## Run the interface preview

```powershell
npm install
npm run dev
```

The browser build uses clearly labeled preview devices and does not capture audio.

## Run the desktop host

After installing the Windows Tauri prerequisites and Rust:

```powershell
npm run tauri dev
```

Import a trusted RVC v2 `.pth` checkpoint, select it, and choose **Load voice**. Loading discovers compatible ContentVec and RMVPE assets from a w-okada-style installation above the model directory and warms the model on CUDA. Then select input and output devices, use headphones to prevent feedback, and choose **Start audio**. Both devices must currently use the same Windows sample rate. Without a loaded voice, audio uses the explicit passthrough backend.

## Documents

- [Architecture](docs/architecture.md)
- [Upstream assessment](docs/upstream-assessment.md)
- [Prototype targets](docs/prototype-targets.md)
- [Native audio spike](docs/native-audio-spike.md)
- [Python compatibility sidecar](docs/python-sidecar.md)
- [Offline RVC compatibility spike](docs/offline-rvc-spike.md)
- [Persistent live RVC spike](docs/live-rvc-spike.md)

## Upstream policy

The w-okada repository is retained separately as a reference checkout. Any source adapted into this project must be recorded with its origin and license. Proprietary or separately licensed components are excluded unless explicitly reviewed.

# Prototype targets and roadmap

This document separates completed engineering milestones from performance targets that still require measurement.

[← Documentation index](README.md)

## Reference system

| Component | Baseline |
| --- | --- |
| Operating system | Windows 11 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Reported VRAM | 6,141 MiB |
| NVIDIA driver | 610.62 during the captured checkpoint |
| Python | 3.11.9 |
| PyTorch | 2.9.0 + CUDA 12.8 |
| ONNX Runtime GPU | 1.26.0 |
| Native live rate | 48 kHz mono inference path |

This machine is a development reference, not a minimum-system specification.

## Milestone status

| Milestone | Status | Evidence |
| --- | --- | --- |
| Desktop product shell | Complete | Tauri/React studio UI, native file dialogs, state and diagnostics |
| Persistent voice library | Complete for prototype | Local entries, selection, names, search, rename, and non-destructive removal |
| Native Windows audio | Complete for prototype | Input/output/monitor streams, bounded queues, gains, gate, telemetry |
| Python compatibility boundary | Complete | Versioned one-shot and persistent standard-I/O protocols |
| Trusted checkpoint loading | Complete | Weights-only validation and exact v1/v2 generator mapping |
| Offline CUDA conversion | Complete | Real RVC v2 fixture and validated output WAV |
| Persistent live RVC | Complete for prototype | Resident sessions, binary PCM, SOLA, FAISS, settings |
| 32/40/48 kHz checkpoint support | Complete for tested fixtures | Generator output normalized to 48 kHz live audio |
| Worker recovery | Complete for transport failures | Timeouts, restart, model/settings replay, health telemetry |
| Audio clock stabilization | Complete for prototype | Adaptive priming and bounded drop/repeat correction |
| Physical loopback benchmark | Pending | Requires external or virtual loopback harness |
| Converted-audio soak certification | Pending | Requires multi-hour matrix and acceptance thresholds |
| Installer and updater | Deferred | Python/CUDA packaging and signing not implemented |
| Built-in virtual microphone | Deferred | Windows driver development and signing not implemented |

## Completed acceptance criteria

### Application and UI

- React/TypeScript production build completes without type errors.
- Slow native operations run outside the Tauri UI thread.
- Loading, ready, running, recovering, and failed states are visible.
- Browser preview is clearly distinguished from native audio.
- Model, audio, and advanced controls have accessible labels.

### Model compatibility

- Metadata-only import does not deserialize `.pth` files.
- Trusted load uses a weights-only checkpoint policy.
- RVC v1/v2 configuration and weights are validated before inference.
- Optional FAISS indexes are dimension-checked and loaded once.
- RVC targets at 32, 40, and 48 kHz can feed the 48 kHz live path.
- Pitch ±50, retrieval, protection, speaker, RMVPE threshold, Chunk, and Extra reach the real worker.

### Native audio and failure handling

- Audio callbacks do not call Python or wait on subprocess I/O.
- Capture, output, and monitor use bounded queues.
- Missing converted samples produce silence instead of callback blocking.
- Playback adapts its safety target after underruns.
- Output and Monitor clocks expose independent correction counters.
- The Python worker can restart and restore a resident model/settings contract.

## Performance language

VC Next uses three different latency concepts:

| Term | Includes | Does not imply |
| --- | --- | --- |
| Model processing time | Feature, pitch, retrieval, generator, resample, stitch | Device or application latency |
| Configured buffering | Chunk, context, crossfade, priming, queue depth | Actual scheduler/device behavior |
| End-to-end loopback | Physical microphone/input to captured output impulse | Subjective quality or every application route |

Only the third is an end-to-end latency measurement.

## Historical engine measurements

The indexed low-latency RVC fixture on the RTX 4050 baseline averaged 107 ms across five 160 ms-hop requests, with a 93–125 ms range. All three named profiles met their selected hop deadlines in that targeted run.

This proves compute feasibility for that fixture. It does not establish total latency through Windows, a virtual cable, Discord/OBS, or another playback/capture application.

## Next acceptance milestone: measured live stability

The next milestone should produce a reproducible report containing:

### Loopback latency

- At least 100 detected impulses per route/profile
- P50, P95, minimum, maximum, and rejected detections
- Physical or virtual loopback topology
- Device names, sample rates, callback sizes, Chunk/Extra, and priming depth
- Separate passthrough and converted results

### Two-hour converted-audio soak

- Captured, processed, played, and dropped frame totals
- Input/output/monitor underruns and overruns
- Reprime count and maximum safety depth
- Clock drop/repeat corrections
- Worker restart count and last error
- Inference P50/P95/max and missed deadlines
- GPU memory and utilization samples
- Audible discontinuity log with timestamps

### Quality fixtures

- Clean speech and noisy-microphone inputs
- Low/high pitch, whispered/unvoiced, plosive, and sibilant coverage
- Boundaries scored with and without retrieval
- Blind comparison against the same checkpoint/settings in w-okada
- Original/converted recordings retained outside the repository

## Provisional targets

> [!IMPORTANT]
> The numbers below are engineering targets until the measurement milestone above is complete.

| Metric | Provisional target |
| --- | ---: |
| Balanced end-to-end loopback P50 | Near or below 150 ms on reference hardware |
| Low-latency end-to-end loopback P50 | Near or below 120 ms on a tuned route |
| P95 inference time | Below the selected Chunk deadline |
| Two-hour playback underruns | 0 after initial/recovery priming |
| Unhandled worker crashes | 0 |
| Non-finite output samples | 0 |
| Persistent queue growth | 0 |

Targets may change after the first trustworthy baseline.

## Roadmap

### Stage 1 — measurement and hardening

- Physical/virtual loopback harness
- Converted-audio soak runner and report export
- Device disconnect/reconnect recovery
- Automated per-model Chunk calibration
- Improved actionable diagnostics

### Stage 2 — distribution and routing

- Reproducible Python/CUDA bundle
- Signed Windows installer and updater
- Built-in or partnered virtual microphone strategy
- Crash reports with privacy-safe local export
- Persistent model library and presets

### Stage 3 — backend expansion

- Native ONNX/TensorRT RVC path
- AMD/Intel investigation
- Linux PipeWire/JACK and Apple Silicon feasibility
- Plugin/backend SDK

### Stage 4 — next-generation streaming research

- Causal feature encoder and F0 estimator
- Stateful lightweight vocoder
- Distillation from a larger offline teacher
- Training objectives for identity, intelligibility, and pitch continuity

The new streaming model is an independent research track; existing RVC compatibility remains supported.

## Reporting a useful test result

Include:

```text
VC Next commit:
Windows version:
GPU / VRAM / driver:
Input / Output / Monitor devices:
Device sample rate:
Checkpoint RVC version and target rate:
Index type and vector count:
Pitch / Index / Protect / F0:
Preset / Chunk / Extra:
Warm-up time:
Inference P50 / P95 / max:
Underruns / overruns / reprimes / corrections:
Worker restarts and last error:
Observed audio problem and timestamp:
```

## Related documents

- [Native audio](native-audio-spike.md)
- [Persistent live RVC](live-rvc-spike.md)
- [Architecture](architecture.md)

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
| Native Windows audio | Complete for prototype | Input/output/monitor streams, bounded queues, rate conversion, gains, high-pass, adaptive suppression, gate, echo control, limiter, telemetry |
| Python compatibility boundary | Complete | Versioned one-shot and persistent standard-I/O protocols |
| Trusted checkpoint loading | Complete | Weights-only validation and exact v1/v2 generator mapping |
| Offline CUDA conversion | Complete | Real RVC v2 fixture and validated output WAV |
| Persistent live RVC | Complete for prototype | Resident sessions, binary PCM, SOLA, FAISS, settings |
| 32/40/48 kHz checkpoint support | Complete for tested fixtures | Generator output normalized to 48 kHz live audio |
| Worker recovery | Complete for transport failures | Timeouts, restart, model/settings replay, health telemetry |
| Device-loss handling | Complete for prototype | Running-session endpoint watcher, actionable error, explicit stream restart |
| Audio clock stabilization | Complete for prototype | Adaptive priming and bounded drop/repeat correction |
| Per-model stream calibration | Complete for prototype | Measures three steady-state calls for all profiles, guards against p95/max spikes, and recommends a stable Chunk/Extra pair |
| Reference validation report | Complete for prototype | Runtime probe, real-model smoke, optional endpoint loopback, and optional converted-worker soak in one report |
| Physical loopback benchmark | Passthrough route validated; native idle route validated; speech loopback pending | CABLE-A shared-mode runs detected 105/105 and 100/100 impulses with zero callback warnings; the latest exact-count run measured P50/P95 247.7 ms. The native Rust/CPAL route also completed a five-second real-model idle run with zero output peak, zero XRuns, and zero missed inference deadlines. A physical converted-speech loopback and alternate-device certification remain |
| Converted-audio soak certification | 120-second worker soak passed; multi-hour/native speech soak pending | `live_route_validation.py` passed a 5-second real VoiceMeeter input 7 → CABLE-B output 13 run on the RTX 4050 reference system with 0 deadline misses, callback warnings, queue drops, or output underruns (Quality 250/600 ms; P50 145.3 ms, P95 224.5 ms, max 247.8 ms). The current Balanced worker run completed 600 realtime hops over 120 seconds with finite output, 0 deadline misses, P50 91.2 ms, P95 105.9 ms, and 154.4 ms max. The native route has an idle/noise-gate smoke result; the multi-hour native speech acceptance matrix remains to be run |
| Windows installer bundles | Complete for prototype | Optimized Tauri MSI/NSIS bundles launch the GUI host, place the staged module beside it as `engine-python`, and include the runtime manifest. An isolated release NSIS install was verified, the installed sidecar loaded the real paired checkpoint/index on CUDA, and the installed GUI stayed alive; signing and dependency installation remain |
| Production installer and updater | Deferred | Python/CUDA dependency bootstrap, signing, and update channel not implemented |
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
- RVC targets at 32, 40, and 48 kHz can feed the 48 kHz live path, and endpoint rates are resampled at the native boundary.
- Pitch ±50, retrieval, protection, speaker, RMVPE threshold, Chunk, and Extra reach the real worker.

### Native audio and failure handling

- Audio callbacks do not call Python or wait on subprocess I/O.
- Capture, output, and monitor use bounded queues.
- Missing converted samples produce silence instead of callback blocking.
- Playback adapts its safety target after underruns.
- Output and Monitor clocks expose independent correction counters.
- The Python worker can restart and restore a resident model/settings contract.
- Device disappearance is detected while a session is live; recovery remains explicit so a stale CPAL stream is never silently rebound to a different endpoint.
- Input cleanup runs in the native callback path: DC/high-pass filtering, adaptive stationary-noise suppression, noise gating, output limiting, and conservative far-end echo control.
- Calibration measures the loaded model without changing the user’s current stream profile and can apply a measured profile from the UI.

## Performance language

VC Next uses three different latency concepts:

| Term | Includes | Does not imply |
| --- | --- | --- |
| Model processing time | Feature, pitch, retrieval, generator, resample, stitch | Device or application latency |
| Configured buffering | Chunk, context, crossfade, priming, queue depth | Actual scheduler/device behavior |
| End-to-end loopback | Physical microphone/input to captured output impulse | Subjective quality or every application route |

Only the third is an end-to-end latency measurement.

## Historical engine measurements

The indexed low-latency RVC fixture on the RTX 4050 baseline averaged 107 ms across five 160 ms-hop requests, with a 93–125 ms range. A separate smoke run against a real w-okada RVC v2 checkpoint plus its sibling `IndexIVFFlat` index completed a 7.5 s load, returned finite CUDA audio, and processed three Balanced 200 ms hops in 110–122 ms each. These are local reference measurements, not a cross-machine guarantee.

The same live path now also passes a real RVC v1 checkpoint with its 256-dimensional `IndexIVFFlat` index: three Balanced hops completed in approximately 119–129 ms after a 6.7 s load, with the ContentVec `units9` head selected automatically and generator output resampled from 40 kHz to the fixed 48 kHz live path.

A 105-impulse CABLE-A shared-mode endpoint run detected every impulse with zero callback warnings. Repeating the exact-count run after the model-import hardening detected 100/100 impulses with zero warnings and measured a 247.7 ms P50/P95 passthrough delay. These are stable route baselines, not converted-voice latency results; the provisional 150 ms Balanced target therefore remains unproven for the application path.

The first real converted-route smoke now passes through the persistent worker: on the RTX 4050 reference system, the real e-girl v2 checkpoint and matching index were driven from VoiceMeeter input 7 to CABLE-B output 13 for 5 seconds using the Quality 12,000/28,800-frame profile. It completed 20 finite calls with zero deadline misses, callback warnings, input/output queue drops, or output underruns; worker timing was 145.3 ms P50, 224.5 ms P95, and 247.8 ms max. The first played sample arrived after 747.9 ms because the harness deliberately waits for a two-chunk safety prime; this is a startup/prime measurement, not a steady-state conversational-latency claim. The run validates the converted sidecar route, while the native Rust/CPAL route and multi-hour soak remain separate acceptance items.

A follow-up five-second empty-input route used the same checkpoint with
`-UsePackageDefaults`. It selected w-okada's canonical ContentVec embedder, the
matching `IndexIVFFlat` index, pitch +14, Index 0.30, Protect 0.50, and a
24,000-frame package Chunk. All ten calls were finite with zero callback
warnings, drops, underruns, or deadline misses; the measured output peak was
exactly 0.0 and all ten hops were suppressed by the idle-input gate. This is a
validated static-noise fix for the empty route, not yet a blind quality match.

The same model/settings were then exercised through the native CPAL/WASAPI
route with `native-route-validation`. The five-second run used VoiceMeeter Out
A2 as input and CABLE-B Input as output. It completed 501 native inference
calls with 0 missed deadlines, 0 queue drops, 0 underruns, and 0 input/output
peaks while the source endpoint was idle. The native report is now the baseline
for checking the desktop host; a non-zero speech fixture still needs to be
routed through a controlled loopback before declaring converted-speech quality
parity.

The current realtime worker soak then ran the same paired checkpoint for 120
wall-clock seconds (600 Balanced hops). It produced finite output for every hop,
kept the worker healthy, and recorded 0 deadline misses with P50 91.2 ms, P95
105.9 ms, and 154.4 ms maximum processing time. This strengthens the compute
stability evidence but is still shorter than the two-hour acceptance target and
does not replace a physical converted-speech loopback.

The same validation report loaded the real paired v2 checkpoint and ran ten converted-worker calls with finite output, a 106.5 ms P95 process time, and zero deadline misses. This is a short worker-soak gate, not a multi-hour converted-route certification.

A one-minute converted-worker soak on the same paired v2 checkpoint produced finite output for 300 calls with a 124.2 ms P50 and 158.1 ms P95, but one 203.1 ms call exceeded the 200 ms Balanced deadline. The safety queues tolerate isolated compute jitter, while sustained deadline margin still needs certification and tuning.

An intermediate 10,560-frame / 25,920-frame (220/540 ms) pair completed 273 calls in a 60-second simulated soak with zero misses (P95 140.6 ms, maximum 189.9 ms). A 60-second realtime run of that same pair remained finite but recorded 12 deadline misses, including a 467.5 ms scheduler spike, so it is not certified as a realtime default. The quality 12,000-frame / 28,800-frame (250/600 ms) profile completed 240 realtime calls over 60 seconds with zero misses (P95 159.1 ms, maximum 214.3 ms) and is the current stable reference profile for this fixture.

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

Use the soak runner's `--realtime` mode for a wall-clock acceptance run; the default mode is a faster simulated audio timeline and is useful for compute regression checks.

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
- One-command reference validation report combining runtime, real-model smoke, optional endpoint loopback, optional converted-worker soak, and optional duplex converted-route validation
- Physical/virtual device disconnect/reconnect test matrix (detection and explicit restart are implemented; certification is pending)
- Automated per-model Chunk calibration (implemented; broaden fixture coverage and persist benchmark reports)
- Improved actionable diagnostics

### Stage 2 — distribution and routing

- Reproducible Python/CUDA bundle
- Signed Windows installer and updater
- Built-in or partnered virtual microphone strategy
- Crash reports with privacy-safe local export
- Persistent model library and presets

### Stage 3 — backend expansion

- Native TensorRT optimization and broader ONNX fixture validation
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

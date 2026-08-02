# Persistent live RVC engine

The live RVC milestone connects a resident Python/CUDA model session to the native Rust capture → inference → playback pipeline without sending audio through HTTP, Socket.IO, or browser code.

[← Documentation index](README.md)

## Outcome

VC Next can:

- import and inspect an RVC checkpoint;
- pair it with an optional FAISS `.index`;
- discover or explicitly select ContentVec assets;
- load RVC v1/v2 generators targeting 32, 40, or 48 kHz;
- warm the complete model session on CUDA;
- retain the session between live audio hops;
- apply pitch, speaker, retrieval, protection, F0, Chunk, and Extra settings;
- stitch streaming output with persistent SOLA state;
- recover the Python process and replay model/settings after transport failure.

The app requires audio to be stopped during model load or unload. A checkpoint must reach **Loaded and warmed** before converted audio can start.

## End-to-end ownership

```mermaid
flowchart LR
    Mic["CPAL capture"] --> CQ["Capture queue"]
    CQ --> RI["Rust inference worker"]
    RI --> Acc["Selected Chunk accumulator"]
    Acc --> IO["Bounded sidecar I/O thread"]
    IO --> Py["Persistent Python CUDA session"]
    Py --> IO
    IO --> RI
    RI --> OQ["Output queues"]
    OQ --> Out["Converted output"]
    OQ --> Mon["Monitor"]
```

The Rust inference worker submits work asynchronously and polls completed responses. Neither capture nor playback waits for Python.

## Live wire protocol

Every live frame begins with a fixed 16-byte header:

| Field | Purpose |
| --- | --- |
| Magic | Rejects unrelated or corrupted streams |
| Kind | Distinguishes JSON control, audio, error, and shutdown frames |
| Request ID | Matches responses to requests |
| Payload length | Bounds allocation and validates complete reads |

Control payloads are compact JSON. Audio payloads are little-endian mono float32 samples. The host validates response kind, identity, and output frame count before exposing samples to the inference backend.

## Model load sequence

```mermaid
sequenceDiagram
    participant UI as Desktop UI
    participant Rust as Rust host
    participant Py as Python worker
    participant GPU as CUDA providers

    UI->>Rust: Load voice
    Rust->>Rust: Confirm audio is stopped
    Rust->>Py: load_model(paths + settings)
    Py->>Py: Weights-only schema validation
    Py->>GPU: Construct ContentVec + RMVPE + generator
    Py->>Py: Validate and reconstruct optional index
    Py->>GPU: Silent warm-up pass
    Py-->>Rust: Ready status + effective geometry
    Rust-->>UI: Loaded and warmed
```

Loading can take several seconds because CUDA contexts, ONNX providers, generator weights, and indexes are initialized. This work runs off the Tauri UI thread and has a longer explicit timeout than ordinary controls.

## Supported checkpoint rates

The native live path uses 48 kHz mono PCM. The RVC generator can target:

| Checkpoint target | Live behavior |
| ---: | --- |
| 32 kHz | Generator output is resampled to 48 kHz before stitching |
| 40 kHz | Generator output is resampled to 48 kHz before stitching |
| 48 kHz | Generator output proceeds directly to alignment |

Targeted GPU verification includes a 32 kHz RVC v2 checkpoint/index pair and a 40 kHz RVC v1 checkpoint/index pair. Both loaded with FAISS retrieval and returned a 48 kHz live status.

## Feature extraction and generation

For each trailing analysis window, Python performs:

```text
48 kHz live history
  → 16 kHz feature waveform
  ├── ContentVec content features
  └── RMVPE pitch and periodicity
  → optional FAISS retrieval blend
  → RVC pitch quantization and feature interpolation
  → v1/v2 PyTorch CUDA generator
  → optional 32/40 → 48 kHz output resample
  → SOLA candidate alignment and crossfade
  → one 48 kHz converted hop
```

All expensive model/session construction occurs during load. Per-hop work reuses the resident sessions and buffers.

## Named streaming profiles

| Preset | Chunk/hop | Analysis context | Crossfade | SOLA search |
| --- | ---: | ---: | ---: | ---: |
| Low latency | 7,680 frames / 160 ms | 19,200 / 400 ms | 1,440 / 30 ms | 480 / 10 ms |
| Balanced | 9,600 / 200 ms | 24,000 / 500 ms | 1,920 / 40 ms | 576 / 12 ms |
| Quality | 12,000 / 250 ms | 28,800 / 600 ms | 2,400 / 50 ms | 720 / 15 ms |

Changing a named mode resets Chunk and Extra to its defaults. Advanced settings can override them afterward.

## Custom Chunk and Extra

**Chunk** is the number of new 48 kHz samples collected before a model request. **Extra/context** is the retained analysis window used to provide past speech context.

The UI currently exposes Chunk values from 3,072 frames (64 ms) through 52,800 frames (1.1 s), including common w-okada-compatible choices such as 12,288 and 49,152. Extra choices range from 3,840 frames (80 ms) through 480,000 frames (10 s).

The engine applies safety rules:

- values must be whole frame counts within the worker's global bounds;
- crossfade and search are reduced when a small Chunk cannot hold the preset values;
- analysis is increased when Extra is too short for `Chunk + crossfade + search`;
- the effective values are reported back in worker status.

> [!CAUTION]
> A value appearing in the selector does not mean it is appropriate for every model. Very small chunks can miss their compute deadline or destabilize pitch and boundaries; very large context increases memory, compute, and response time.

## Stateful SOLA stitching

The generator produces an analysis-window waveform, not a phase-continuous live stream. The `SolaStitcher` therefore:

1. retains the previous overlap tail;
2. searches a bounded alignment region in the new candidate;
3. chooses the highest normalized correlation offset;
4. applies an equal-power crossfade;
5. emits exactly one Chunk-sized hop;
6. stores the new overlap state.

The first converted hop is silent while the overlap state primes. This is deliberate and avoids emitting an invalid partial boundary.

## FAISS retrieval

Retrieval is optional per model. During load, the worker verifies that the index dimension matches the checkpoint feature channels:

- RVC v1 commonly uses 256 channels;
- RVC v2 commonly uses 768 channels.

The index is reconstructed once, not once per hop. Each conversion then queries neighbors, calculates inverse-distance weights, blends retrieved features according to Index retrieval, and uses Protect ratio to preserve unvoiced consonants.

An index ratio above zero without a loaded index is rejected with an actionable error.

## Live settings

| Setting | Range/behavior | Requires restart? |
| --- | --- | --- |
| Pitch | −50 to +50 semitones | Applied before next session; stream state reset |
| Index retrieval | 0–100% | Requires a valid loaded index above zero |
| Protect ratio | 0–50% | Used during retrieval blending |
| Speaker ID | Checkpoint-defined | Validated against speaker embeddings |
| RMVPE threshold | 0.01–0.20 | Controls voiced/unvoiced detection |
| Preset | Quality/Balanced/Low latency | Updates default geometry |
| Chunk | Explicit frame choice | Changes request/output hop |
| Extra/context | Explicit frame choice | Changes retained analysis history |

Settings are persisted locally by model path. Imported library entries and the last selected voice are also restored locally; renaming or removing an entry never renames or deletes the source files.

## Worker deadlines and recovery

The native host distinguishes:

- handshake timeout;
- ordinary control timeout;
- model-load/warm-up timeout;
- live-audio response timeout;
- shutdown timeout.

After a pipe or transport failure:

```mermaid
stateDiagram-v2
    Healthy --> Recovering: Transport failure
    Recovering --> Healthy: Restart + reload + settings replay
    Recovering --> Failed: Replacement or replay fails
    Failed --> Healthy: User stops and loads again successfully
```

Rust remembers the last successful model-load parameters. Later settings calls are merged into that snapshot so the replacement worker receives the current model, index, embedder, pitch, retrieval, speaker, F0, and stream geometry.

While recovery runs, native audio remains alive and safely emits silence instead of blocking callbacks.

## Telemetry

The live status includes:

- model, index, ContentVec, and RMVPE paths;
- checkpoint version, target rate, precision, device, and speaker count;
- index type, dimension, and vector count;
- selected/effective Chunk, analysis, crossfade, and search frames;
- warm-up time and process-call count;
- last resample, content, pitch, retrieval, generator, stitch, and total times;
- last SOLA offset;
- provider names;
- worker state, restart count, and last worker error.

## Historical reference measurements

One indexed `mayaputri.pth` RVC v2 fixture on the RTX 4050 baseline produced:

| Measurement | Result |
| --- | ---: |
| Index | `IndexIVFFlat`, 34,823 × 768 |
| Low-latency geometry | 7,680 / 19,200 frames |
| Five indexed round trips | 93–125 ms |
| Five-pass average | 107 ms |
| Tested preset deadlines | All three returned within their selected hop |
| Six-hop boundary jumps | 0.0002–0.0050 peak amplitude |

These are per-request fixture measurements. They do not include every input/output buffer, device driver, virtual cable, or application buffer and therefore are not end-to-end latency claims.

## Verification

Automated Python tests cover protocol framing, stream geometry, pitch bounds, SOLA behavior, index helpers, and error handling. Rust tests cover dynamic Chunk propagation, transport framing, recovery state, and model/settings replay.

```powershell
.\engine-python\.venv\Scripts\python.exe -m unittest discover -s engine-python\tests -p "test_*.py" -v
cargo test --manifest-path src-tauri\Cargo.toml
```

## Current limitations

- The 160 ms named preset is still above the project's eventual responsiveness goal.
- Custom 64–150 ms Chunks are experimental and not automatically calibrated.
- RMVPE is the only connected live pitch extractor.
- RVC ONNX live inference is not connected.
- The first converted block is silent while state primes.
- Audio endpoints require matching default rates.
- Physical loopback latency and multi-hour converted-audio soak tests remain pending.
- Worker recovery preserves the process/model contract, but device disconnection still requires an audio refresh/restart.

## Next work

1. Add a repeatable physical-loopback measurement harness.
2. Run extended converted-audio soaks with worker restarts and device jitter.
3. Calibrate smaller Chunk values by model and hardware rather than exposing one universal recommendation.
4. Compare continuous rate correction with the current bounded sample-slip approach.
5. Add blind quality fixtures for pitch continuity, consonants, and chunk boundaries.

## Related documents

- [Architecture](architecture.md)
- [Python sidecar](python-sidecar.md)
- [Native audio](native-audio-spike.md)
- [Prototype targets](prototype-targets.md)

# Persistent live RVC spike

## Outcome

VC Next can now import a trusted RVC v2 checkpoint, load and warm it from the Tauri interface, retain the complete inference session on the RTX 4050, and connect that session to the native capture → inference → playback pipeline.

The UI does not start audio automatically. Model loading is only permitted while audio is stopped, and an imported checkpoint must reach **Loaded and warmed** before the main Start button is enabled.

## Transport and ownership

```text
CPAL capture callback
  -> lock-free native input queue
  -> Rust inference worker (480-frame chunks)
  -> selected streaming-hop accumulator
  -> bounded sidecar I/O thread
  -> framed float32 PCM over local stdio
  -> persistent Python CUDA session
  -> framed converted PCM response
  -> lock-free native output queue
  -> CPAL playback callback
```

The wire header is fixed at 16 bytes and identifies protocol magic, frame kind, request ID, and payload length. Control payloads are JSON; audio payloads are little-endian float32 mono samples. Payload size and response identity are validated on both sides. No network server, HTTP request, WebSocket, base64 encoding, or audio-callback Python call is involved.

## Streaming profiles

| Preset | Hop | Analysis | Crossfade | SOLA search |
| --- | ---: | ---: | ---: | ---: |
| Low latency | 160 ms | 400 ms | 30 ms | 10 ms |
| Balanced | 200 ms | 500 ms | 40 ms | 12 ms |
| Quality | 250 ms | 600 ms | 50 ms | 15 ms |

The selected geometry is reported by the Python worker, propagated through the Rust client, and applied before the native inference backend starts. RMVPE remains the only supported pitch extractor, but its voiced/unvoiced threshold is configurable from 0.01 to 0.20.

## Representative verification

- Checkpoint: local `mayaputri.pth` RVC v2 model
- Feature encoder: CUDA ContentVec ONNX
- Pitch extractor: CUDA RMVPE ONNX
- Generator: PyTorch CUDA FP16
- Input/output rate: 48 kHz mono
- Low-latency hop/window: 7,680/19,200 frames (160/400 ms)
- Balanced hop/window: 9,600/24,000 frames (200/500 ms)
- Quality hop/window: 12,000/28,800 frames (250/600 ms)
- Five indexed low-latency round trips: 93–125 ms, 107 ms average
- All three profiles returned finite samples and met their selected hop deadline
- Boundary jumps in the six-hop fixture: 0.0002–0.0050 peak amplitude
- Desktop state: **Loaded and warmed**, worker **Resident**

The checkpoint and feature assets stay outside this repository and are not redistributed.

## Current limitations

- The first converted block is silent while the pipeline primes.
- A 160 ms hop is still above the final latency target.
- Sibling FAISS indexes, retrieval strength, standard RVC unvoiced protection, pitch, RMVPE threshold, streaming preset, and speaker ID are applied when audio starts.
- Model settings are persisted locally by model path and restored when that voice is imported again.
- Input and output devices still require matching sample rates.
- Physical loopback latency and extended live-audio soak testing are pending.
- The framed worker now has bounded control/audio deadlines, a dedicated pipe reader, and supervised restart with model/settings replay after transport failure.

The next engine pass is physical loopback measurement, extended soak testing, continuous resampling where needed, and smaller stable hops.

# Prototype architecture

## Boundary

```text
React UI
   │ typed commands and events
Tauri host
   ├── device and hardware discovery
   ├── lifecycle and diagnostics
   └── native audio core
          ├── capture and output workers
          ├── lock-free ring buffers
          ├── resampling and chunk stitching
          └── inference adapter
                 ├── PyTorch RVC compatibility
                 └── ONNX/CUDA optimized path
```

The UI never owns the real-time audio loop. It renders state and sends control commands. Audio capture, buffering, inference scheduling, and playback remain independent of browser rendering and network transports.

## First engine seam

The initial inference contract will expose:

- `prepare(model, device, precision)`
- `process(audio_chunk, persistent_state)`
- `reset()`
- `inspect_capabilities()`
- `collect_stage_timings()`

That contract lets the prototype use a compatibility backend first, then add a native ONNX or future causal streaming backend without rebuilding the product UI.

## Real-time rules

- No allocation, file I/O, logging, or model loading on audio callback threads.
- Fixed-capacity single-producer/single-consumer buffers between real-time stages.
- Capture and playback continue safely if inference misses its deadline.
- All latency values distinguish measured loopback latency from inferred processing time.
- Backends must prove output compatibility before an optimized model is cached.

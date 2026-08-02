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

Potentially slow native operations never execute on the Windows UI thread. Audio-device discovery, Python runtime probing, checkpoint inspection, model loading and CUDA warm-up, live-worker control, and audio stream startup/shutdown are dispatched to Tauri's blocking worker pool. The React client renders immediately from fallback state, initializes independent services concurrently, and exposes an explicit operation state while a model is being loaded.

## First engine seam

The initial inference contract will expose:

- `prepare(model, device, precision)`
- `process(audio_chunk, persistent_state)`
- `reset()`
- `inspect_capabilities()`
- `collect_stage_timings()`

That contract lets the prototype use a compatibility backend first, then add a native ONNX or future causal streaming backend without rebuilding the product UI.

The inference contract is implemented in `src-tauri/src/inference.rs`. It runs either a no-op backend or the live RVC adapter on a dedicated worker thread using fixed 480-frame native chunks. Capture and playback own separate bounded queues, and telemetry records worker calls, processing time, missed chunk deadlines, processed frames, and dropped output frames.

The Python compatibility process lives under `engine-python`. Lightweight runtime probes use versioned JSON lines. Live conversion uses a separate persistent worker and a framed binary standard-I/O protocol: JSON controls lifecycle, while float32 PCM remains binary. The sidecar does not open a network port. A bounded Rust I/O worker owns the subprocess, and the native inference thread only submits or polls work; neither audio callback waits on Python.

## Implemented host commands

- `get_audio_devices()` enumerates active WASAPI input and output endpoints.
- `start_audio_engine(input_device_id, output_device_id)` starts the native capture → inference worker → playback pipeline.
- `get_audio_engine_status()` reports buffer depth, peaks, frame counts, and xruns.
- `stop_audio_engine()` deterministically drops both streams.
- `load_live_rvc_model(model_path, pitch_shift)` starts the worker, loads the checkpoint and feature models, and completes CUDA warm-up while audio is stopped.
- `set_live_rvc_settings(pitch_shift)` updates live-safe worker settings.
- `get_live_rvc_status()` reports resident model, chunk size, provider, and processing telemetry.
- `unload_live_rvc_model()` releases the persistent model session while audio is stopped.

The no-op backend remains an explicit fallback. The live RVC backend accumulates 480-frame native chunks into a 9,600-frame request, exchanges it asynchronously with the persistent Python process, and fans converted mono output to the selected output channels. Python retains a 24,000-frame trailing analysis window and emits one 9,600-frame hop through stateful SOLA stitching. The first hop primes the overlap state with silence.

## Real-time rules

- No allocation, file I/O, logging, or model loading on audio callback threads.
- Fixed-capacity single-producer/single-consumer buffers between real-time stages.
- Capture and playback continue safely if inference misses its deadline.
- All latency values distinguish measured loopback latency from inferred processing time.
- Backends must prove output compatibility before an optimized model is cached.

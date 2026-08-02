# Native audio spike

## Purpose

This checkpoint proves that VC Next can own its local Windows audio path without browser AudioWorklets, Socket.IO, base64 PCM transport, or Python audio queues.

## Implemented

- Native input and output enumeration through CPAL's WASAPI host.
- Default-device detection and format reporting.
- Explicit input/output selection in the Tauri interface.
- Separate CPAL capture and playback streams.
- Fixed-capacity `ArrayQueue<f32>` between callbacks.
- Mono downmix on capture and channel fan-out on playback.
- Validated input/output gain in the native callbacks.
- A smoothed envelope-based input noise gate with a transparent off setting.
- Live captured/played frame counts, queue depth, peaks, overruns, underruns, and callback errors.
- Start, status, stop, and device-discovery Tauri commands.
- Browser preview that never claims to capture native audio.
- Separate bounded capture and playback queues around a dedicated inference worker.
- A backend-neutral inference contract with `prepare`, `process`, `reset`, and capability inspection.
- A no-op fallback plus a persistent Python RVC backend behind the same inference contract.
- Worker telemetry for processing time, processed frames, missed deadlines, and dropped output frames.
- Adaptive playback priming that starts at a bounded 20 ms target, raises the safety depth after an underrun, and settles back after a stable session.
- Bounded sample-slip drift correction for the main output and optional monitor route, with drop/repeat counters exposed in diagnostics.
- Optional monitor playback stream with independent gain, queue, peak, underrun, and recovery telemetry.

## Safety

Passthrough sends microphone audio directly to the selected output. Use headphones during testing to avoid acoustic feedback. Device selections are locked while streams are active.

## Deliberate limitations

- Input and output must currently have matching default sample rates.
- The current drift correction is bounded sample-slip correction; a continuous high-quality resampler is still a later pass.
- Startup priming adapts between 960 and 4,800 frames based on observed underruns.
- Only common Windows `f32`, `i16`, and `u16` sample formats are accepted.
- The current stream uses shared-mode device defaults rather than an exclusive/event-driven tuning pass.
- Bypass remains deferred; unloaded sessions use explicit passthrough and loaded RVC sessions use converted output.
- Live RVC loading, pitch extraction, feature encoding, FAISS retrieval, consonant protection, and generation are connected.

## Next measurement pass

1. Record callback sizes and scheduling variance per device.
2. Replace sample-slip correction with a controlled resampler where long-session measurements justify it.
3. Add impulse/loopback measurement for real P50/P95 end-to-end latency.
4. Run a two-hour passthrough and converted-audio soak test and record xruns.
5. Tune the connected 160 ms stateful SOLA profile toward smaller stable hops.

# Prototype targets

## Reference system

- Windows 11
- NVIDIA GeForce RTX 4050 Laptop GPU
- 6,141 MiB reported VRAM
- NVIDIA driver 610.62

## Phase 0 acceptance

- React interface builds without warnings or type errors.
- Tauri desktop scaffold has a typed system-profile command.
- The upstream engine boundary and licensing policy are documented.
- Latency is represented as a per-stage budget, never as an unmeasured claim.

## Engine-spike measurements

The next milestone will record:

- Physical or virtual loopback end-to-end latency, P50 and P95
- Audio callback block size and scheduling variance
- Pitch extraction, content encoding, retrieval, generation, stitching, and resampling time
- CPU/GPU utilization and GPU memory
- Missed inference deadlines and buffer underruns during a two-hour soak test
- Dry versus converted output alignment

The initial balanced-mode engineering goal is a stable result near 100 ms on the reference machine. It remains a target until loopback measurements prove it.

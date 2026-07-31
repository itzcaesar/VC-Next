# VC Next

VC Next is a Windows-first feasibility prototype for a modern, local, low-latency RVC voice changer. It is a new application architecture, not a permanent fork of w-okada/voice-changer.

## Phase 0 target

- Reference platform: Windows 11
- Reference GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM
- UI: React 19, TypeScript, Vite
- Desktop host: Tauri 2
- Engine direction: native audio core with swappable RVC compatibility backends

The current interface is intentionally backed by prototype data. The displayed latency values are budgets, not benchmark results.

## Run the interface

```powershell
npm install
npm run dev
```

The browser build works without Rust. Building the desktop host requires the Windows Tauri prerequisites and a Rust toolchain.

## Documents

- [Architecture](docs/architecture.md)
- [Upstream assessment](docs/upstream-assessment.md)
- [Prototype targets](docs/prototype-targets.md)

## Upstream policy

The w-okada repository is retained separately as a reference checkout. Any source adapted into this project must be recorded with its origin and license. Proprietary or separately licensed components are excluded unless explicitly reviewed.

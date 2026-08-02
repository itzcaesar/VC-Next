# Offline RVC compatibility spike

## Outcome

VC Next completed a real offline RVC v2 conversion on the Windows 11 / NVIDIA GeForce RTX 4050 Laptop GPU reference machine. This proves the selected PyTorch, CUDA, ONNX Runtime, ContentVec, RMVPE, and generator components can execute together in the isolated Python 3.11 environment.

## Pipeline

```text
16 kHz mono speech
  -> ContentVec layer 12 (ONNX Runtime CUDA)
  -> RMVPE pitch (ONNX Runtime CUDA)
  -> feature upsampling and pitch quantization
  -> RVC v2 FP16 generator (PyTorch CUDA)
  -> 48 kHz mono PCM WAV
```

The generator definitions under `engine-python/vc_next_sidecar/rvc_compat/infer_pack` are the minimal MIT-licensed compatibility set adapted from w-okada commit `f1caf8e7c39fd0d6866202be27bf142790191a51`. Exact provenance and consolidated upstream notices are stored beside the code.

## Representative fixture

- Checkpoint: local `mayaputri.pth` RVC v2 model
- Checkpoint size: 57,583,493 bytes
- Target rate: 48 kHz
- Pitch enabled: yes
- State dictionary: exact inference match after removing the unused training-only posterior encoder
- Content encoder: local `contentvec-f.onnx`, layer-12 768-channel output
- Pitch extractor: local `rmvpe_20231006.onnx`
- Retrieval index: deliberately disabled for the first proof

The checkpoint and model assets remain outside the repository and are not redistributed.

## Measured first-pass result

- Input: 3.00 seconds at 16 kHz
- Output: 2.98 seconds at 48 kHz
- Content extraction: 358.9 ms
- Pitch extraction: 444.9 ms
- Generator: 946.1 ms
- Generator headroom: 3.15x real time
- Output peak: 0.733
- Output RMS: 0.141
- Output validation: finite, mono, PCM 16-bit WAV

The one-shot total was 8.01 seconds because it includes Python imports, ONNX session construction, checkpoint loading, network construction, and CUDA warm-up. Those operations must move to a persistent worker and occur once per model, outside the real-time path.

## Follow-on work after the live connection

The persistent sidecar, bounded binary transport, and first stateful SOLA streaming pass are now complete. Remaining work is:

1. Reduce the current 200 ms hop without destabilizing feature and pitch extraction.
2. Add cancellation, automatic restart, and underrun recovery.
3. Optional FAISS retrieval was added and validated later through the persistent streaming worker.
4. Compare physical loopback latency and blind audio quality against w-okada.

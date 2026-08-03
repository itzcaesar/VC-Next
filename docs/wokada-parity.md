# w-okada compatibility audit

VC Next keeps the RVC model ecosystem while replacing the browser transport and
real-time host. This note records the processing boundaries that are intentionally
matched to w-okada's RVC v2 path and the checks used when a voice sounds different.

## Reference implementation

The audit is based on the upstream files in the w-okada checkout at commit
`f1caf8e7c39fd0d6866202be27bf142790191a51` (the upstream `master` tip used for
this audit):

- `server/voice_changer/RVC/RVCr2.py`
- `server/voice_changer/RVC/pipeline/Pipeline.py`
- `server/voice_changer/RVC/pitchExtractor/RMVPEOnnxPitchExtractor.py`
- `server/voice_changer/VoiceChanger.py`

The upstream files are used as compatibility references. VC Next does not bundle
the checkout or copy its browser/server transport into the desktop application.
The reference checkout is kept outside this repository; only the compatibility
rules and test evidence below belong to VC Next.

## Matched processing rules

| Boundary | w-okada | VC Next |
| --- | --- | --- |
| Live input | 48 kHz device audio | 48 kHz native audio engine |
| Feature-rate audio | Per-hop `resampy` `kaiser_fast` to 16 kHz | Same filter and per-hop v2 feature history |
| RVC v2 hop | 160 samples at 16 kHz | Same rounded `convertSize` geometry |
| Extra/context | `extraConvertSize` front context before the current output candidate | Same explicit front context; effective retained window is derived from `convertSize` rounding |
| Content features | ContentVec (or Fairseq HuBERT fallback), then 2× interpolation | ContentVec by default; explicit Fairseq HuBERT `.pt/.pth` is also supported |
| RMVPE front context | Analyze the post-front tail; restore zero F0 frames | Same trim-and-restore boundary |
| Retrieval | FAISS nearest vector, `k=1` | Same; index dimension is validated before load |
| Retrieval front restore | Reuse the rolling post-inference feature buffer before the current tail | Same rolling buffer, including zero rows for each new live hop |
| Protection | Preserve source features where `pitchf < 1` | Same mask and ratio range `0..0.5` |
| Pitch shift | Multiply F0 by `2 ** (semitones / 12)` | Same |
| Pitch coarse bins | 50–1100 Hz mel mapping, rounded 1–255 | Same |
| RMVPE threshold | `0.30` in the upstream ONNX extractor | `0.30` default; advanced range `0.01..0.99` |
| Loudness | `sqrt(mean(crop²))`, output scaled by `sqrt(vol)` | Same crop and gain rule |
| Output stitching | 4096-frame equal-power crossfade + 12 ms SOLA | Native `SolaStitcher` with the same geometry |

The RMVPE threshold was a material parity bug: VC Next previously defaulted to
`0.03`, which changes voiced/unvoiced decisions even when every model and index
file is identical. Existing stored settings with the old untouched default are
migrated to `0.30` on first launch.

The retrieval-front rule was the second material parity issue found by the raw
window comparison. The live worker had been restoring the current ContentVec
frames, while w-okada restores the preceding `feature_buffer` rows. VC Next now
keeps that post-interpolation buffer between hops and appends the same zero
feature rows before each retrieval query. On the RTX 4050 reference checkpoint,
the first voiced window now matches a seeded upstream generator window at
`0.99997` correlation; the remaining difference is expected half/quantized
output rounding.

## Reproducible checks

Run the complete compatibility and runtime tests from the repository root:

```powershell
npm run release:check -- -SkipModelSmoke
```

For a real checkpoint and paired index, use the native speech fixture. The
following uses the same settings as the app's Balanced profile:

```powershell
.\scripts\run-native-speech-loopback.ps1 `
  -ModelPath 'C:\path\voice.pth' `
  -IndexPath 'C:\path\voice.index' `
  -FixturePath '.\outputs\e-girl-speech-current.wav' `
  -Seconds 60 -PitchShift 14 -IndexRatio 0.5 -ProtectRatio 0.5 `
  -ChunkFrames 9600 -ExtraFrames 24000 -Preset balanced
```

The report must show a healthy worker, zero missed inference deadlines, and zero
output underruns/reprimes. The latest RTX 4050 reference run with the real
`e-girl_e350_s42700.pth` and its paired FAISS index met those conditions for 120
seconds (600 Balanced hops). A separate idle USB-microphone run produced zero
output and monitor peaks while the native silence backstop suppressed the
decoder.

The current machine's virtual-cable endpoints do not deliver the speech fixture
back to the selected capture endpoint, so a device-level converted-speech
comparison is intentionally not marked as passed. Use the WAV comparison tool
once both recordings exist:

```powershell
python engine-python/tools/compare_audio.py `
  --reference .\reference-w-okada.wav `
  --candidate .\reference-vc-next.wav `
  --report .\outputs\audio-comparison.json
```

This separates model/output differences from Windows routing differences; it
does not claim parity when the recordings were made with different input,
model, index, or processing settings.

## When the sound still differs

Check the following in order:

1. Confirm the same feature asset (`contentvec-f.onnx` or the explicit Fairseq
   `hubert_base.pt`) and RMVPE asset are selected. A different
   Hubert/ContentVec variant changes the voice even with the same checkpoint.
2. Confirm pitch shift, index ratio, protection, speaker ID, and RMVPE threshold.
3. Confirm the paired `.index` file is loaded and its dimension matches the
   checkpoint (v2 models normally use 768 channels).
4. Compare Chunk and Extra/context. Smaller windows reduce latency but alter the
   amount of context available to ContentVec, RMVPE, and the generator.
5. Compare the raw converted WAV before device playback. If the files match but
   the live route does not, the remaining difference is in device resampling,
   gain, or virtual-cable routing rather than RVC inference.

This separation keeps model-quality changes measurable and prevents audio-driver
artifacts from being mistaken for neural-model differences.

## Fairseq HuBERT comparison on the RTX 4050 reference machine

The same real `e-girl_e350_s42700.pth` checkpoint, paired 768-dimensional
`.index`, RMVPE asset, and input fixture were run through both feature backends.
After warm-up, both paths produced finite output and met the Balanced 500 ms
chunk deadline. The direct feature comparison over the same 54-frame window
reported cosine similarity `0.999999`, mean absolute feature delta `0.000299`,
and identical feature RMS (`0.316219`). This indicates that the earlier quality
difference was not caused by a missing HuBERT fallback alone; remaining audible
differences should be investigated through pitch settings, index/protection,
Chunk/Extra geometry, stitching, and device routing.

Fairseq initialization was slower on this machine (about 3.6 s versus 3.3 s for
the ONNX feature pipeline), so ContentVec ONNX remains the recommended default.

To remove generator randomness from the output check, the smoke harness now
accepts a diagnostic `--seed`. With seed `777`, the same 72,000-frame fixture
produced a best-lag correlation of `0.9992513`, RMSE `0.00022851`, MAE
`0.00004496`, and a gain ratio of `0.9997887` between ContentVec and Fairseq
HuBERT. That is strong evidence that the remaining unseeded waveform variation
is the RVC latent-noise sampler rather than a feature-path mismatch.

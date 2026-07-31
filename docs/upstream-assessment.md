# Upstream assessment

Reference checkout: `w-okada/voice-changer` at commit `f1caf8e` on `master`.

## Reusable areas requiring license tracking

- RVC model-slot parsing and model loading
- RVC inference and ONNX-export conventions
- Pitch-extractor adapters, especially RMVPE and FCPE integration
- Existing model settings and compatibility behavior
- SOLA/crossfade behavior as a comparison baseline

## Areas to replace

- Browser AudioWorklet ownership of the local audio path
- Socket.IO and base64 audio transport for local processing
- Python queue-based capture and playback scheduling
- UI-specific engine state and undocumented control behavior
- Runtime allocation and CPU/GPU transfers in critical inference paths
- Packaging and dependency installation experience

## Files inspected

- `server/voice_changer/VoiceChangerV2.py`
- `server/voice_changer/Local/ServerDevice.py`
- `server/voice_changer/RVC/pipeline/Pipeline.py`
- `server/voice_changer/RVC/pitchExtractor/`
- `server/voice_changer/RVC/inferencer/`
- `server/restapi/MMVC_Rest_VoiceChanger.py`
- `client/lib/worklet/src/voice-changer-worklet-processor.ts`
- `client/demo/package.json`
- `client/lib/package.json`

## License rule

The upstream repository includes MIT notices, but individual model families and bundled dependencies may have separate terms. Code is not copied into VC Next until its file-level origin and dependency licenses are recorded.

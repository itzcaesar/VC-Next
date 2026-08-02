# RVC compatibility source provenance

The files under `infer_pack/` were copied from:

- Repository: `https://github.com/w-okada/voice-changer.git`
- Commit: `f1caf8e7c39fd0d6866202be27bf142790191a51`
- Source path: `server/voice_changer/RVC/inferencer/rvc_models/infer_pack/`
- Imported files: `models.py`, `modules.py`, `attentions.py`, `commons.py`, and `transforms.py`

They are used only for PyTorch RVC checkpoint architecture compatibility. Browser transport, Python audio queues, Socket.IO, model management, and the upstream application shell were not imported.

The upstream repository's consolidated MIT notices are preserved in `UPSTREAM_LICENSE`. Local changes must remain minimal and should be documented here.

## Local changes

- Removed constructor debug prints from `models.py`.
- Model loading is implemented separately in `loader.py` with a weights-only checkpoint policy, inference-only module removal, exact state-dictionary validation, and explicit device/precision selection.

## Streaming behavior reference

`vc_next_sidecar/streaming.py` is a new, isolated implementation of synchronized overlap-add behavior using `server/voice_changer/VoiceChangerV2.py` from the same upstream commit as its compatibility reference. It retains the upstream 12 ms normalized alignment-search concept and equal-power overlap profile, but is implemented around VC Next's fixed float32 hop contract and does not import upstream transport, queues, settings, or application state.

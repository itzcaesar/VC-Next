# Upstream assessment and reuse policy

VC Next uses w-okada as a compatibility reference while building a separate desktop product architecture.

[← Documentation index](README.md)

## Reference snapshot

| Item | Value |
| --- | --- |
| Repository | `https://github.com/w-okada/voice-changer.git` |
| Branch inspected | `master` |
| Commit | `f1caf8e7c39fd0d6866202be27bf142790191a51` |
| Primary purpose | RVC model compatibility and behavior reference |

The reference checkout is kept outside the VC Next repository. It is not used as the application worktree or Git base.

## Strategy

VC Next follows a hybrid approach:

1. **Study** upstream model formats, settings, and established RVC behavior.
2. **Reuse selectively** only where source provenance and license terms are clear.
3. **Record exact imports** by repository, commit, source path, destination path, and local modification.
4. **Replace product architecture** where the upstream design conflicts with native desktop audio, bounded transport, diagnostics, or maintainability goals.
5. **Exclude uncertain assets** until their terms are reviewed.

## Reuse decision matrix

| Area | Decision | VC Next treatment |
| --- | --- | --- |
| RVC v1/v2 generator definitions | Selective adaptation | Minimal inference-compatible subset with provenance and upstream notice |
| Checkpoint/config conventions | Compatibility reference | New strict weights-only loader and validation |
| FAISS index behavior | Behavioral compatibility | New loading, validation, reconstruction, and blend path |
| Pitch and protect settings | Behavioral compatibility | New typed settings and validation |
| SOLA alignment concept | Reference and reimplementation | New stateful implementation for a fixed float32 hop contract |
| RMVPE/ContentVec asset layout | File-layout compatibility | Discovery of w-okada-style module paths |
| Browser AudioWorklet audio loop | Replace | Native CPAL/WASAPI callbacks |
| Socket.IO/base64 local PCM | Replace | Framed binary standard I/O |
| Python capture/playback queues | Replace | Rust-owned bounded queues and native streams |
| Upstream UI and model slots | Replace | New React/Tauri desktop studio and typed state |
| Packaging/update flow | Replace | Future VC Next-specific installer/update design |
| Proprietary model families | Exclude by default | Require separate legal and technical review |

## Imported compatibility source

The files under `engine-python/vc_next_sidecar/rvc_compat/infer_pack/` originate from:

```text
server/voice_changer/RVC/inferencer/rvc_models/infer_pack/
```

Imported filenames:

- `models.py`
- `modules.py`
- `attentions.py`
- `commons.py`
- `transforms.py`

They are used only to construct RVC-compatible inference networks. The upstream application shell, web transport, audio ownership, settings stores, model-slot UI, and server lifecycle were not imported with them.

The authoritative local records are:

- [PROVENANCE.md](../engine-python/vc_next_sidecar/rvc_compat/PROVENANCE.md)
- [UPSTREAM_LICENSE](../engine-python/vc_next_sidecar/rvc_compat/UPSTREAM_LICENSE)

## Files inspected during architecture assessment

| Upstream path | Reason for inspection |
| --- | --- |
| `server/voice_changer/VoiceChangerV2.py` | Chunk history, SOLA behavior, and engine state |
| `server/voice_changer/Local/ServerDevice.py` | Device/audio ownership and queues |
| `server/voice_changer/RVC/pipeline/Pipeline.py` | RVC feature, pitch, retrieval, and generator flow |
| `server/voice_changer/RVC/pitchExtractor/` | RMVPE/FCPE adapter conventions |
| `server/voice_changer/RVC/inferencer/` | Model architecture and load conventions |
| `server/restapi/MMVC_Rest_VoiceChanger.py` | REST/server control boundary |
| `client/lib/worklet/src/voice-changer-worklet-processor.ts` | Browser audio and local transport behavior |
| `client/demo/package.json` | Frontend application dependencies |
| `client/lib/package.json` | Shared client library dependencies |

Inspection does not imply that code was copied.

## What VC Next implements independently

### Desktop product layer

- Tauri lifecycle and command surface
- React voice-studio interface
- native model/index/embedder picker workflow
- model operation state and recovery UX
- persistent local model library, settings, and diagnostics presentation

### Native audio and scheduling

- CPAL/WASAPI input, output, and monitor routes
- fixed-capacity queues and inference scheduling
- gain and noise gate
- adaptive playback priming
- independent output/monitor clock correction
- audio and inference telemetry

### Sidecar transport and supervision

- versioned JSON-line discovery protocol
- framed binary float32 live protocol
- request identities and payload validation
- bounded control/audio deadlines
- subprocess restart, model replay, and settings replay

### Compatibility glue

- safe first-pass file inspection
- weights-only trusted checkpoint validation
- exact state-dictionary validation
- 32/40/48 kHz live output handling
- ContentVec/RMVPE discovery
- FAISS dimension validation and partial-result handling
- custom stream-geometry validation

## License and asset rules

> [!IMPORTANT]
> A permissive license on one repository does not automatically cover third-party checkpoints, indexes, feature models, vocoders, datasets, characters, voices, or bundled components.

Before adapting or redistributing anything, record:

1. repository and exact commit;
2. source and destination paths;
3. license text and copyright notice;
4. modifications made locally;
5. transitive components or assets it expects;
6. redistribution and commercial-use restrictions;
7. model/voice consent or provenance where applicable.

VC Next does not commit or redistribute user checkpoints, `.index` files, ContentVec, RMVPE, recordings, or w-okada model bundles.

## Adding another upstream-derived file

Use this checklist before committing:

```text
[ ] The file is technically necessary rather than merely convenient.
[ ] The upstream repository and exact commit are recorded.
[ ] The file-level license and required notices are understood.
[ ] A clean-room or new implementation was considered.
[ ] The destination is isolated from newly authored code where practical.
[ ] Local modifications are documented.
[ ] Tests cover the compatibility behavior being preserved.
[ ] No checkpoint, dataset, recording, or separately licensed asset is included.
```

If any item is uncertain, do not import the source until it is reviewed.

## Long-term boundary

RVC compatibility is one backend, not the application architecture. A future native or causal engine should use the same model/session controls, audio device layer, telemetry, and recovery contracts without depending on the upstream web server or UI.

## Related documents

- [Architecture](architecture.md)
- [Python sidecar](python-sidecar.md)
- [Offline RVC proof](offline-rvc-spike.md)

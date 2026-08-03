import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export type ConversionMode = "quality" | "balanced" | "latency";

export interface SystemProfile {
  os: string;
  gpu: string;
  vramMb: number;
  driverVersion: string;
  source: "prototype-baseline" | "native-probe";
}

export interface AudioDevice {
  id: string;
  name: string;
  isDefault: boolean;
  channels: number;
  sampleRate: number;
  sampleFormat: string;
}

export interface AudioDeviceSnapshot {
  inputs: AudioDevice[];
  outputs: AudioDevice[];
  defaultInputId: string | null;
  defaultOutputId: string | null;
  backend: string;
  source: "native-probe" | "browser-preview";
}

export interface AudioEngineStatus {
  state: "stopped" | "passthrough" | "rvc" | "preview";
  inputDeviceId: string | null;
  outputDeviceId: string | null;
  monitorDeviceId: string | null;
  inputDeviceName: string | null;
  outputDeviceName: string | null;
  monitorDeviceName: string | null;
  sampleRate: number | null;
  inferenceSampleRate: number;
  inputChannels: number | null;
  outputChannels: number | null;
  monitorChannels: number | null;
  bufferCapacityFrames: number;
  bufferedFrames: number;
  captureBufferedFrames: number;
  capturedFrames: number;
  processedFrames: number;
  playedFrames: number;
  monitorBufferedFrames: number;
  monitorPlayedFrames: number;
  underruns: number;
  overruns: number;
  monitorUnderruns: number;
  monitorOverruns: number;
  primeTargetFrames: number;
  monitorPrimeTargetFrames: number;
  reprimes: number;
  monitorReprimes: number;
  driftDroppedFrames: number;
  driftRepeatedFrames: number;
  monitorDriftDroppedFrames: number;
  monitorDriftRepeatedFrames: number;
  inferenceBackend: string;
  inferenceStateful: boolean;
  inferenceChunkFrames: number;
  inferenceCalls: number;
  lastInferenceMicros: number;
  maxInferenceMicros: number;
  missedInferenceDeadlines: number;
  droppedInferenceFrames: number;
  inferenceSilenceSuppressedCalls: number;
  inputPeak: number;
  outputPeak: number;
  monitorPeak: number;
  inputGainDb: number;
  outputGainDb: number;
  monitorGainDb: number;
  noiseGateDb: number;
  noiseSuppressionStrength: number;
  echoControlStrength: number;
  highPassEnabled: boolean;
  lastError: string | null;
}

export interface AudioRouteTestResult {
  outputDeviceName: string;
  monitorDeviceName: string | null;
  durationMs: number;
  outputFrames: number;
  monitorFrames: number;
  outputPeak: number;
  monitorPeak: number;
  outputError: string | null;
  monitorError: string | null;
}

export interface AudioProcessingSettings {
  inputGainDb: number;
  outputGainDb: number;
  monitorGainDb: number;
  noiseGateDb: number;
  noiseSuppressionStrength: number;
  echoControlStrength: number;
  highPassEnabled: boolean;
}

export interface InferenceRuntimeProbe {
  source: "python-sidecar" | "browser-preview";
  protocolVersion: number;
  engineVersion: string;
  platform: string;
  python: {
    version: string;
    executable: string;
    sidecarCompatible: boolean;
    rvcEnvironmentCompatible: boolean;
    recommendedVersion: string;
  };
  packages: Record<string, string | null>;
  torchRuntime: {
    imported: boolean;
    cudaAvailable: boolean;
    cudaVersion: string | null;
    deviceName: string | null;
    deviceCapability: number[] | null;
    error: string | null;
  };
  onnxRuntime: {
    imported: boolean;
    availableProviders: string[];
    cudaProviderAvailable: boolean;
    error: string | null;
  };
  capabilities: string[];
  readyForRvc: boolean;
  blockers: string[];
  optionalMissing: string[];
}

export interface ModelPreset {
  id: string;
  name: string;
  initials: string;
  format: "RVC v1" | "RVC v2" | "RVC ONNX";
  sampleRate: number | null;
  sourcePath?: string;
  indexPaths?: string[];
  recommendedIndexPath?: string | null;
  embedderPath?: string | null;
  /** True when the user explicitly selected the embedder in the package dialog. */
  embedderExplicit?: boolean;
  pairingNote?: string;
  modelDefaults?: ModelDefaults;
}

/** Safe settings imported from a w-okada-style params.json package file. */
export interface ModelDefaults {
  pitchShift?: number;
  indexRatio?: number;
  protectRatio?: number;
  chunkFrames?: number;
  extraFrames?: number;
  embedder?: string;
  pitchEstimator?: string;
  recommendedIndex?: string | null;
}

export interface RvcModelSettings {
  pitchShift: number;
  indexRatio: number;
  protectRatio: number;
  speakerId: number;
  indexPath: string | null;
  contentvecPath: string | null;
  f0Threshold: number;
  streamingPreset: ConversionMode;
  chunkFrames: number;
  extraFrames: number;
}

export interface ModelInspectionResult {
  path: string;
  name: string;
  extension: ".pth" | ".onnx" | ".index";
  role: "rvc-checkpoint" | "onnx-model" | "faiss-index";
  container: string;
  sizeBytes: number;
  siblingIndexes: string[];
  recommendedIndex?: string | null;
  packageComplete?: boolean;
  pairingNote?: string;
  modelDefaults?: ModelDefaults;
  safeInspectionOnly: boolean;
  checkpointLoaded: boolean;
  warnings: string[];
}

export interface TrustedCheckpointInspection extends ModelInspectionResult {
  rvcVersion: string;
  targetSampleRate: number;
  usesPitch: boolean;
  speakerCount: number | null;
  weightKeyCount: number;
  configLength: number;
  loadPolicy: "torch-weights-only";
}

export interface LiveRvcStatus {
  state: "empty" | "ready";
  protocolVersion: number;
  modelPath: string | null;
  contentvecPath?: string | null;
  featurePath?: string | null;
  featureBackend?: "contentvec-onnx" | "fairseq-hubert" | string | null;
  rmvpePath?: string | null;
  indexPath?: string | null;
  indexLoaded?: boolean;
  indexDimension?: number | null;
  indexVectorCount?: number;
  indexType?: string | null;
  indexNeighbors?: number;
  sampleRate: number;
  chunkFrames: number;
  chunkMilliseconds: number;
  analysisFrames: number;
  analysisMilliseconds: number;
  extraFrames?: number;
  extraMilliseconds?: number;
  crossfadeFrames: number;
  crossfadeMilliseconds: number;
  solaSearchFrames: number;
  solaSearchMilliseconds: number;
  silenceFrontFrames?: number;
  silenceFrontFeatureFrames?: number;
  generatorConvertFrames?: number;
  streamPrimed: boolean;
  rvcVersion?: string | null;
  targetSampleRate?: number | null;
  speakerCount?: number | null;
  precision?: string | null;
  device?: string | null;
  backend?: "pytorch" | "onnx" | string | null;
  generatorProviders?: string[];
  pitchShift: number;
  speakerId?: number;
  indexRatio?: number;
  protectRatio?: number;
  f0Method?: string;
  f0Threshold?: number;
  streamingPreset?: ConversionMode;
  warmupMs?: number;
  processCalls: number;
  lastProcessMs: number;
  lastResampleMs?: number;
  lastContentMs?: number;
  lastPitchMs?: number;
  lastRetrievalMs?: number;
  lastGeneratorMs?: number;
  lastStitchMs?: number;
  lastSolaOffsetFrames?: number;
  silenceSuppressedCalls?: number;
  lastInputRms?: number;
  lastInputPeak?: number;
  maxInputRms?: number;
  maxInputPeak?: number;
  lastInputVolume?: number;
  lastOutputGain?: number;
  silenceGateRms?: number;
  silenceGatePeak?: number;
  silenceGateMode?: "rms" | "rms-and-peak" | "rms+activity";
  providers?: string[];
  workerState?: "stopped" | "healthy" | "recovering" | "failed";
  workerRestarts?: number;
  lastWorkerError?: string | null;
}

export interface LiveCalibrationMeasurement {
  preset: ConversionMode;
  chunkFrames: number;
  extraFrames?: number;
  analysisFrames: number;
  processMs: number;
  maxProcessMs?: number;
  sampleCount?: number;
  deadlineMs: number;
  headroomMs: number;
  stable: boolean;
}

export interface LiveCalibrationResult {
  sampleRate: number;
  recommendedPreset: ConversionMode;
  restoredPreset: ConversionMode;
  profiles: LiveCalibrationMeasurement[];
  message: string;
}

export const FALLBACK_PROFILE: SystemProfile = {
  os: "Windows 11",
  gpu: "NVIDIA GeForce RTX 4050 Laptop GPU",
  vramMb: 6141,
  driverVersion: "610.62",
  source: "prototype-baseline",
};

export const FALLBACK_AUDIO_DEVICES: AudioDeviceSnapshot = {
  inputs: [
    { id: "preview-input", name: "Default Windows input", isDefault: true, channels: 1, sampleRate: 48_000, sampleFormat: "f32" },
  ],
  outputs: [
    { id: "preview-output", name: "Choose output in desktop build", isDefault: true, channels: 2, sampleRate: 48_000, sampleFormat: "f32" },
  ],
  defaultInputId: "preview-input",
  defaultOutputId: "preview-output",
  backend: "Browser preview",
  source: "browser-preview",
};

export const STOPPED_ENGINE_STATUS: AudioEngineStatus = {
  state: "stopped",
  inputDeviceId: null,
  outputDeviceId: null,
  monitorDeviceId: null,
  inputDeviceName: null,
  outputDeviceName: null,
  monitorDeviceName: null,
  sampleRate: null,
  inferenceSampleRate: 48_000,
  inputChannels: null,
  outputChannels: null,
  monitorChannels: null,
  bufferCapacityFrames: 96_000,
  bufferedFrames: 0,
  captureBufferedFrames: 0,
  capturedFrames: 0,
  processedFrames: 0,
  playedFrames: 0,
  monitorBufferedFrames: 0,
  monitorPlayedFrames: 0,
  underruns: 0,
  overruns: 0,
  monitorUnderruns: 0,
  monitorOverruns: 0,
  primeTargetFrames: 1920,
  monitorPrimeTargetFrames: 1920,
  reprimes: 0,
  monitorReprimes: 0,
  driftDroppedFrames: 0,
  driftRepeatedFrames: 0,
  monitorDriftDroppedFrames: 0,
  monitorDriftRepeatedFrames: 0,
  inferenceBackend: "Not running",
  inferenceStateful: false,
  inferenceChunkFrames: 480,
  inferenceCalls: 0,
  lastInferenceMicros: 0,
  maxInferenceMicros: 0,
  missedInferenceDeadlines: 0,
  droppedInferenceFrames: 0,
  inferenceSilenceSuppressedCalls: 0,
  inputPeak: 0,
  outputPeak: 0,
  monitorPeak: 0,
  inputGainDb: 0,
  outputGainDb: 0,
  monitorGainDb: -6,
  noiseGateDb: -80,
  noiseSuppressionStrength: 0,
  echoControlStrength: 0,
  highPassEnabled: false,
  lastError: null,
};

export const EMPTY_LIVE_RVC_STATUS: LiveRvcStatus = {
  state: "empty",
  protocolVersion: 1,
  modelPath: null,
  contentvecPath: null,
  featurePath: null,
  featureBackend: "contentvec-onnx",
  sampleRate: 48_000,
  chunkFrames: 9_600,
  chunkMilliseconds: 200,
  analysisFrames: 24_000,
  analysisMilliseconds: 500,
  extraFrames: 24_000,
  extraMilliseconds: 500,
  crossfadeFrames: 4_096,
  crossfadeMilliseconds: 4096 / 48,
  solaSearchFrames: 576,
  solaSearchMilliseconds: 12,
  silenceFrontFrames: 0,
  silenceFrontFeatureFrames: 0,
  generatorConvertFrames: 28_672,
  streamPrimed: false,
  pitchShift: 0,
  speakerId: 0,
  indexPath: null,
  indexLoaded: false,
  indexDimension: null,
  indexVectorCount: 0,
  indexType: null,
  indexNeighbors: 0,
  indexRatio: 0,
  protectRatio: 0.5,
  f0Method: "RMVPE",
  // w-okada's RMVPE ONNX extractor uses a 0.30 periodicity threshold.
  f0Threshold: 0.30,
  streamingPreset: "balanced",
  processCalls: 0,
  lastProcessMs: 0,
  workerState: "stopped",
  workerRestarts: 0,
  lastWorkerError: null,
};

export const FALLBACK_INFERENCE_RUNTIME: InferenceRuntimeProbe = {
  source: "browser-preview",
  protocolVersion: 1,
  engineVersion: "0.1.0",
  platform: "Browser preview",
  python: {
    version: "Desktop only",
    executable: "",
    sidecarCompatible: false,
    rvcEnvironmentCompatible: false,
    recommendedVersion: "3.11",
  },
  packages: {},
  torchRuntime: {
    imported: false,
    cudaAvailable: false,
    cudaVersion: null,
    deviceName: null,
    deviceCapability: null,
    error: null,
  },
  onnxRuntime: {
    imported: false,
    availableProviders: [],
    cudaProviderAvailable: false,
    error: null,
  },
  capabilities: [],
  readyForRvc: false,
  blockers: ["Open the Tauri desktop app to probe the Python runtime."],
  optionalMissing: [],
};

export const MODEL_PRESETS: ModelPreset[] = [
  { id: "reference-rvc", name: "Reference RVC voice", initials: "RV", format: "RVC v2", sampleRate: 40000 },
  { id: "reference-onnx", name: "Reference ONNX voice", initials: "OX", format: "RVC ONNX", sampleRate: 48000 },
];

export async function getSystemProfile(): Promise<SystemProfile> {
  try {
    return await invoke<SystemProfile>("get_system_profile");
  } catch {
    return FALLBACK_PROFILE;
  }
}

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function getAudioDevices(): Promise<AudioDeviceSnapshot> {
  if (!isTauriRuntime()) return FALLBACK_AUDIO_DEVICES;
  return invoke<AudioDeviceSnapshot>("get_audio_devices");
}

export async function startAudioEngine(
  inputDeviceId: string,
  outputDeviceId: string,
  monitorDeviceId: string | null,
  processing: AudioProcessingSettings,
): Promise<AudioEngineStatus> {
  if (!isTauriRuntime()) {
    const input = FALLBACK_AUDIO_DEVICES.inputs.find((device) => device.id === inputDeviceId) ?? FALLBACK_AUDIO_DEVICES.inputs[0];
    const output = FALLBACK_AUDIO_DEVICES.outputs.find((device) => device.id === outputDeviceId) ?? FALLBACK_AUDIO_DEVICES.outputs[0];
    return {
      ...STOPPED_ENGINE_STATUS,
      state: "preview",
      inputDeviceId,
      outputDeviceId,
      monitorDeviceId,
      inputDeviceName: input.name,
      outputDeviceName: output.name,
      monitorDeviceName: monitorDeviceId
        ? (FALLBACK_AUDIO_DEVICES.outputs.find((device) => device.id === monitorDeviceId)?.name ?? null)
        : null,
      sampleRate: input.sampleRate,
      inputChannels: input.channels,
      outputChannels: output.channels,
      monitorChannels: monitorDeviceId ? 2 : null,
      monitorBufferedFrames: 0,
      monitorPlayedFrames: 0,
      monitorUnderruns: 0,
      monitorOverruns: 0,
      monitorPeak: 0,
      ...processing,
    };
  }
  return invoke<AudioEngineStatus>("start_audio_engine", {
    inputDeviceId,
    outputDeviceId,
    monitorDeviceId,
    ...processing,
  });
}

export async function stopAudioEngine(): Promise<AudioEngineStatus> {
  if (!isTauriRuntime()) return STOPPED_ENGINE_STATUS;
  return invoke<AudioEngineStatus>("stop_audio_engine");
}

export async function restartAudioEngine(): Promise<AudioEngineStatus> {
  if (!isTauriRuntime()) throw new Error("Audio recovery requires the Tauri desktop app.");
  return invoke<AudioEngineStatus>("restart_audio_engine");
}

export async function getAudioEngineStatus(): Promise<AudioEngineStatus> {
  if (!isTauriRuntime()) return STOPPED_ENGINE_STATUS;
  return invoke<AudioEngineStatus>("get_audio_engine_status");
}

export async function testAudioRoutes(
  outputDeviceId: string,
  monitorDeviceId: string | null,
  durationMs = 800,
): Promise<AudioRouteTestResult> {
  if (!isTauriRuntime()) {
    return {
      outputDeviceName: "Browser preview output",
      monitorDeviceName: monitorDeviceId ? "Browser preview monitor" : null,
      durationMs,
      outputFrames: Math.round(durationMs * 48),
      monitorFrames: monitorDeviceId ? Math.round(durationMs * 48) : 0,
      outputPeak: 0.08,
      monitorPeak: monitorDeviceId ? 0.08 : 0,
      outputError: null,
      monitorError: null,
    };
  }
  return invoke<AudioRouteTestResult>("test_audio_routes", {
    outputDeviceId,
    monitorDeviceId,
    durationMs,
  });
}

export async function probeInferenceRuntime(): Promise<InferenceRuntimeProbe> {
  if (!isTauriRuntime()) return FALLBACK_INFERENCE_RUNTIME;
  return invoke<InferenceRuntimeProbe>("probe_inference_runtime");
}

export async function openRuntimeSetup(): Promise<string> {
  if (!isTauriRuntime()) throw new Error("Runtime setup requires the Tauri desktop app.");
  return invoke<string>("open_runtime_setup");
}

export async function inspectRvcModel(path: string): Promise<ModelInspectionResult> {
  if (!isTauriRuntime()) throw new Error("Model inspection requires the Tauri desktop app.");
  return invoke<ModelInspectionResult>("inspect_rvc_model", { path });
}

export async function discoverRvcModels(path: string): Promise<string[]> {
  if (!isTauriRuntime()) throw new Error("Model folder discovery requires the Tauri desktop app.");
  return invoke<string[]>("discover_rvc_models", { path });
}

export async function inspectTrustedRvcCheckpoint(path: string): Promise<TrustedCheckpointInspection> {
  if (!isTauriRuntime()) throw new Error("Checkpoint inspection requires the Tauri desktop app.");
  return invoke<TrustedCheckpointInspection>("inspect_trusted_rvc_checkpoint", { path });
}

export async function loadLiveRvcModel(modelPath: string, settings: RvcModelSettings): Promise<LiveRvcStatus> {
  if (!isTauriRuntime()) throw new Error("Live model loading requires the Tauri desktop app.");
  return invoke<LiveRvcStatus>("load_live_rvc_model", { modelPath, ...settings });
}

export async function setLiveRvcSettings(settings: RvcModelSettings): Promise<LiveRvcStatus> {
  if (!isTauriRuntime()) return EMPTY_LIVE_RVC_STATUS;
  const { indexPath: _indexPath, contentvecPath: _contentvecPath, ...liveSettings } = settings;
  return invoke<LiveRvcStatus>("set_live_rvc_settings", liveSettings);
}

export async function getLiveRvcStatus(): Promise<LiveRvcStatus> {
  if (!isTauriRuntime()) return EMPTY_LIVE_RVC_STATUS;
  return invoke<LiveRvcStatus>("get_live_rvc_status");
}

export async function calibrateLiveRvc(): Promise<LiveCalibrationResult> {
  if (!isTauriRuntime()) throw new Error("Stream calibration requires the Tauri desktop app.");
  return invoke<LiveCalibrationResult>("calibrate_live_rvc");
}

export async function unloadLiveRvcModel(): Promise<LiveRvcStatus> {
  if (!isTauriRuntime()) return EMPTY_LIVE_RVC_STATUS;
  return invoke<LiveRvcStatus>("unload_live_rvc_model");
}

export async function chooseAndInspectRvcModel(): Promise<ModelInspectionResult | null> {
  const packageSelection = await chooseAndInspectRvcPackage();
  return packageSelection?.inspection ?? null;
}

export interface RvcModelPackageSelection {
  inspection: ModelInspectionResult;
  indexPath: string | null;
  contentvecPath: string | null;
}

export async function chooseAndInspectRvcPackage(): Promise<RvcModelPackageSelection | null> {
  if (!isTauriRuntime()) return null;
  const selected = await open({
    multiple: false,
    directory: false,
    title: "Select the RVC model checkpoint (.pth or .onnx)",
    filters: [{ name: "RVC voice models", extensions: ["pth", "onnx"] }],
  });
  if (typeof selected !== "string") return null;
  const inspection = await inspectRvcModel(selected);
  const selectedIndex = await open({
    multiple: false,
    directory: false,
    title: "Select the matching FAISS index (Cancel to auto-detect or skip)",
    filters: [{ name: "RVC retrieval indexes", extensions: ["index"] }],
  });
  const selectedEmbedder = await open({
    multiple: false,
    directory: false,
    title: "Select a feature embedder (Cancel to auto-discover)",
    filters: [{ name: "ContentVec or HuBERT embedders", extensions: ["onnx", "pt", "pth"] }],
  });
  return {
    inspection,
    indexPath: typeof selectedIndex === "string" ? selectedIndex : null,
    contentvecPath: typeof selectedEmbedder === "string" ? selectedEmbedder : null,
  };
}

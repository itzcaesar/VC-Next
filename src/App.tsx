import { useEffect, useMemo, useRef, useState, type ChangeEvent, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";
import {
  FALLBACK_AUDIO_DEVICES,
  FALLBACK_INFERENCE_RUNTIME,
  FALLBACK_PROFILE,
  EMPTY_LIVE_RVC_STATUS,
  MODEL_PRESETS,
  STOPPED_ENGINE_STATUS,
  getAudioDevices,
  getAudioEngineStatus,
  getLiveRvcStatus,
  calibrateLiveRvc,
  getSystemProfile,
  inspectRvcModel,
  discoverRvcModels,
  probeInferenceRuntime,
  openRuntimeSetup,
  getRuntimeSetupCommand,
  loadLiveRvcModel,
  unloadLiveRvcModel,
  setLiveRvcSettings,
  startAudioEngine,
  restartAudioEngine,
  stopAudioEngine,
  testAudioRoutes,
  type AudioDevice,
  type AudioDeviceSnapshot,
  type AudioEngineStatus,
  type AudioRouteTestResult,
  type AudioProcessingSettings,
  type ConversionMode,
  type InferenceRuntimeProbe,
  type LiveRvcStatus,
  type LiveCalibrationResult,
  type ModelPreset,
  type ModelInspectionResult,
  type RvcModelSettings,
  type SystemProfile,
} from "./lib/engine";

const modeLabels: Record<ConversionMode, string> = {
  quality: "Quality",
  balanced: "Balanced",
  latency: "Low latency",
};

const modeDescriptions: Record<ConversionMode, string> = {
  quality: "More context and overlap for the cleanest conversion.",
  balanced: "A practical default for voice chat and streaming.",
  latency: "Smaller buffers for the quickest response.",
};

const streamProfiles: Record<ConversionMode, { hop: number; extra: number; overlap: number; search: number }> = {
  quality: { hop: 250, extra: 600, overlap: 50, search: 15 },
  balanced: { hop: 200, extra: 500, overlap: 40, search: 12 },
  latency: { hop: 160, extra: 400, overlap: 30, search: 10 },
};

type ModelPackageBusy = "checkpoint" | "folder" | "index" | "embedder" | "cover" | "adding" | null;
type ModelLoadPhase = "preparing" | "loading" | "finalizing" | null;
type PeakTone = "idle" | "signal" | "hot" | "clip";
type SidebarName = "library" | "session";
type LiveLogTone = "info" | "success" | "warning" | "error";
type ModelScope = "all" | "recent" | "favorites";
type ModelSort = "recent" | "name" | "format" | "folder";

interface LiveLogEntry {
  id: number;
  time: string;
  tone: LiveLogTone;
  message: string;
}

interface CoverImageAsset {
  dataUrl: string;
  name: string;
  type: string;
}

interface ModelOrganization {
  folder: string;
  tags: string[];
}

type LibraryModel = ModelPreset & {
  coverImage?: CoverImageAsset | null;
  folder?: string;
  tags?: string[];
};

interface ModelPackageDraft {
  checkpointPath: string | null;
  inspection: ModelInspectionResult | null;
  indexPath: string | null;
  indexCandidates: string[];
  embedderPath: string | null;
  coverImage: CoverImageAsset | null;
}

const EMPTY_MODEL_PACKAGE: ModelPackageDraft = {
  checkpointPath: null,
  inspection: null,
  indexPath: null,
  indexCandidates: [],
  embedderPath: null,
  coverImage: null,
};

const COVER_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "gif", "bmp"];

const CHUNK_OPTIONS = [
  3_072, 3_840, 4_800, 5_760, 7_200, 7_680, 9_600, 10_560, 12_000, 12_288,
  14_400, 16_800, 19_200, 21_600, 24_000, 28_800, 33_600, 38_400,
  43_200, 48_000, 49_152, 52_800,
];

const EXTRA_OPTIONS = [
  3_840, 4_096, 7_680, 16_320, 24_000, 25_920, 32_640, 65_280, 131_040, 144_000,
  192_000, 240_000, 288_000, 336_000, 384_000, 432_000, 480_000,
];

function frameDurationLabel(frames: number) {
  const seconds = frames / 48_000;
  return `${frames.toLocaleString()} · ${seconds < 1 ? `${Math.round(seconds * 1_000)} ms` : `${Number(seconds.toFixed(3))} sec`}`;
}

type IconName =
  | "activity"
  | "close"
  | "dots"
  | "github"
  | "headset"
  | "info"
  | "library"
  | "microphone"
  | "moon"
  | "play"
  | "plus"
  | "search"
  | "settings"
  | "speaker"
  | "stop"
  | "sun";

const iconPaths: Record<IconName, string> = {
  activity: "M3 12h4l2.5-6 5 12 2.5-6H21",
  close: "M6 6l12 12M18 6 6 18",
  dots: "M5 12h.01M12 12h.01M19 12h.01",
  github: "M12 3a9 9 0 0 0-2.85 17.54c.45.08.62-.2.62-.44v-1.54c-2.53.56-3.06-1.08-3.06-1.08-.41-1.05-1-1.33-1-1.33-.82-.57.06-.56.06-.56.91.07 1.39.96 1.39.96.81 1.42 2.13 1.01 2.65.77.08-.6.32-1.01.58-1.24-2.02-.23-4.15-1.04-4.15-4.62 0-1.02.36-1.85.95-2.5-.1-.23-.41-1.18.09-2.46 0 0 .78-.25 2.55.95a8.5 8.5 0 0 1 4.64 0c1.77-1.2 2.55-.95 2.55-.95.5 1.28.19 2.23.09 2.46.59.65.95 1.48.95 2.5 0 3.59-2.13 4.39-4.16 4.62.33.29.62.84.62 1.7v2.52c0 .24.16.52.62.43A9 9 0 0 0 12 3Z",
  headset: "M4 14v-2a8 8 0 0 1 16 0v2M4 14h3v6H6a2 2 0 0 1-2-2v-4Zm16 0h-3v6h1a2 2 0 0 0 2-2v-4Z",
  info: "M12 10v6M12 7.5h.01M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z",
  library: "M4 5h16v14H4zM4 9h16M8 5v14",
  microphone: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3ZM5 12a7 7 0 0 0 14 0M12 19v3M8 22h8",
  moon: "M20 15.2A8.5 8.5 0 0 1 8.8 4 8.5 8.5 0 1 0 20 15.2Z",
  play: "m9 6 9 6-9 6V6Z",
  plus: "M12 5v14M5 12h14",
  search: "m21 21-4.35-4.35M19 11a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z",
  settings: "M4 7h10M18 7h2M4 17h2M10 17h10M14 4v6M6 14v6",
  speaker: "M5 9v6h4l5 4V5L9 9H5Zm12.5 1a3 3 0 0 1 0 4M19.5 7a7 7 0 0 1 0 10",
  stop: "M7 7h10v10H7z",
  sun: "M12 3v2m0 14v2M3 12h2m14 0h2M5.64 5.64l1.42 1.42m9.88 9.88 1.42 1.42m0-12.72-1.42 1.42M7.06 16.94l-1.42 1.42M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
};

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg aria-hidden="true" className="icon" width={size} height={size} viewBox="0 0 24 24">
      <path d={iconPaths[name]} />
    </svg>
  );
}

function LevelMeter({ active, output = false, peak = 0 }: { active: boolean; output?: boolean; peak?: number }) {
  const tone = peakTone(peak);
  return (
    <div className={`level-meter ${active ? "active" : ""} tone-${tone}`} aria-label={`${active ? "Signal active" : "No signal"}${tone === "clip" ? " · clipping" : tone === "hot" ? " · hot" : ""}`}>
      {Array.from({ length: 20 }, (_, index) => (
        <i key={index} className={`${index > 16 ? "peak" : ""} ${tone === "clip" && index > 15 ? "clip" : ""}`} style={{ animationDelay: `${-(index + (output ? 4 : 0)) * 36}ms` }} />
      ))}
    </div>
  );
}

function Waveform({ active }: { active: boolean }) {
  const heights = [16, 24, 35, 20, 44, 63, 36, 72, 48, 28, 54, 82, 46, 68, 31, 56, 77, 41, 25, 64, 88, 58, 36, 71, 49, 21, 43, 62, 38, 75, 52, 29, 68, 84, 46, 34, 57, 73, 40, 22, 48, 66, 33, 54, 79, 45, 27, 60, 72, 39, 19, 51, 69, 43, 30, 58, 81, 48, 35, 63, 42, 24, 47, 70];
  return (
    <div className={`waveform ${active ? "active" : ""}`} aria-hidden="true">
      <div className="waveform-center" />
      {heights.map((height, index) => <i key={index} style={{ height: `${height}%`, animationDelay: `${-index * 29}ms` }} />)}
    </div>
  );
}

function deviceSummary(device: AudioDevice | undefined) {
  if (!device) return "Unavailable";
  const channelLabel = device.channels === 1 ? "Mono" : `${device.channels} channels`;
  return `${device.sampleRate.toLocaleString()} Hz · ${channelLabel} · ${device.sampleFormat}`;
}

function peakDb(peak: number) {
  return peak > 0.00001 ? `${(20 * Math.log10(peak)).toFixed(1)}` : "−∞";
}

function peakTone(peak: number): PeakTone {
  if (peak >= 0.98) return "clip";
  if (peak >= 0.75) return "hot";
  if (peak > 0.001) return "signal";
  return "idle";
}

function sameWindowsPath(left: string | null | undefined, right: string | null | undefined) {
  return Boolean(left && right && left.toLocaleLowerCase() === right.toLocaleLowerCase());
}

const MODEL_SETTINGS_STORAGE_KEY = "vc-next:model-settings:v2";
const LEGACY_MODEL_SETTINGS_STORAGE_KEY = "vc-next:model-settings:v1";
const AUDIO_SETTINGS_STORAGE_KEY = "vc-next:audio-settings:v1";
const AUDIO_SETTINGS_STORAGE_VERSION_KEY = "vc-next:audio-settings:version";
const THEME_STORAGE_KEY = "vc-next:theme:v1";
const DEVICE_SETTINGS_STORAGE_KEY = "vc-next:device-settings:v1";
const MODEL_LIBRARY_STORAGE_KEY = "vc-next:model-library:v1";
const MODEL_SELECTION_STORAGE_KEY = "vc-next:model-selection:v1";
const MODEL_FAVORITES_STORAGE_KEY = "vc-next:model-favorites:v1";
const MODEL_RECENTS_STORAGE_KEY = "vc-next:model-recents:v1";
const MODEL_ORGANIZATION_STORAGE_KEY = "vc-next:model-organization:v1";
const CALIBRATION_STORAGE_KEY = "vc-next:calibration:v1";
const SIDEBAR_LAYOUT_STORAGE_KEY = "vc-next:sidebar-layout:v2";
const LEGACY_SIDEBAR_LAYOUT_STORAGE_KEY = "vc-next:sidebar-layout:v1";

const DEFAULT_LIBRARY_WIDTH = 262;
const DEFAULT_SESSION_WIDTH = 295;
const MIN_LIBRARY_WIDTH = DEFAULT_LIBRARY_WIDTH;
const MAX_LIBRARY_WIDTH = 400;
const MIN_SESSION_WIDTH = DEFAULT_SESSION_WIDTH;
const MAX_SESSION_WIDTH = 320;
const DESKTOP_STUDIO_MIN_WIDTH = 560;
const COMPACT_STUDIO_MIN_WIDTH = 500;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

function loadStoredSidebarLayout() {
  const fallback = { libraryWidth: DEFAULT_LIBRARY_WIDTH, sessionWidth: DEFAULT_SESSION_WIDTH };
  try {
    const currentValue = window.localStorage.getItem(SIDEBAR_LAYOUT_STORAGE_KEY);
    const legacyValue = window.localStorage.getItem(LEGACY_SIDEBAR_LAYOUT_STORAGE_KEY);
    const value = currentValue ?? legacyValue;
    if (!value) return fallback;
    const parsed: unknown = JSON.parse(value);
    if (typeof parsed !== "object" || parsed === null) return fallback;
    const candidate = parsed as Record<string, unknown>;
    const legacyDefault = !currentValue && candidate.libraryWidth === 248;
    return {
      libraryWidth: legacyDefault
        ? fallback.libraryWidth
        : typeof candidate.libraryWidth === "number" && Number.isFinite(candidate.libraryWidth)
          ? clamp(candidate.libraryWidth, MIN_LIBRARY_WIDTH, MAX_LIBRARY_WIDTH)
        : fallback.libraryWidth,
      sessionWidth: typeof candidate.sessionWidth === "number" && Number.isFinite(candidate.sessionWidth)
        ? clamp(candidate.sessionWidth, MIN_SESSION_WIDTH, MAX_SESSION_WIDTH)
        : fallback.sessionWidth,
    };
  } catch {
    return fallback;
  }
}

function hasStaleAutoDiscoveredRinna(model: ModelPreset) {
  const embedderHint = (model.modelDefaults?.embedder ?? "").toLocaleLowerCase();
  return !model.embedderExplicit
    && embedderHint.includes("hubert_base_l12")
    && (model.embedderPath ?? "").toLocaleLowerCase().includes("rinna_hubert");
}

function defaultModelSettings(model: ModelPreset): RvcModelSettings {
  const defaults = model.modelDefaults ?? {};
  const staleAutoDiscoveredRinna = hasStaleAutoDiscoveredRinna(model);
  return {
    pitchShift: defaults.pitchShift ?? 0,
    indexRatio: model.indexPaths?.length ? defaults.indexRatio ?? 0.5 : 0,
    protectRatio: defaults.protectRatio ?? 0.33,
    speakerId: 0,
    indexPath: model.recommendedIndexPath ?? defaults.recommendedIndex ?? model.indexPaths?.[0] ?? null,
    // Older imports could persist Rinna Hubert while params.json requested
    // hubert_base_l12. Let the corrected resolver choose canonical ContentVec
    // unless the user explicitly selected an embedder.
    contentvecPath: staleAutoDiscoveredRinna ? null : model.embedderPath ?? null,
    f0Threshold: 0.30,
    streamingPreset: "balanced",
    chunkFrames: defaults.chunkFrames ?? 9_600,
    extraFrames: defaults.extraFrames ?? 24_000,
  };
}

function isLegacyPrototypeSettings(settings: RvcModelSettings, model: ModelPreset) {
  const defaultIndexRatio = model.indexPaths?.length ? 0.5 : 0;
  const defaultEmbedderPath = defaultModelSettings(model).contentvecPath;
  return settings.pitchShift === 0
    && settings.indexRatio === defaultIndexRatio
    && settings.protectRatio === 0.33
    && settings.speakerId === 0
    && settings.indexPath === (model.recommendedIndexPath ?? model.indexPaths?.[0] ?? null)
    && settings.contentvecPath === defaultEmbedderPath
    && settings.f0Threshold === 0.30
    && settings.streamingPreset === "balanced"
    && settings.chunkFrames === 9_600
    && settings.extraFrames === 24_000;
}

const DEFAULT_AUDIO_SETTINGS: AudioProcessingSettings = {
  inputGainDb: 0,
  outputGainDb: 0,
  monitorGainDb: -6,
  noiseGateDb: -80,
  // Keep the compatibility path neutral by default. Users can opt into
  // suppression for noisy rooms, but it should not reshape a clean voice.
  noiseSuppressionStrength: 0,
  echoControlStrength: 0,
  highPassEnabled: false,
};

function loadStoredAudioSettings(): AudioProcessingSettings {
  try {
    const value = window.localStorage.getItem(AUDIO_SETTINGS_STORAGE_KEY);
    const parsed: unknown = value ? JSON.parse(value) : null;
    const candidate = parsed && typeof parsed === "object"
      ? parsed as Partial<AudioProcessingSettings>
      : {};
    // v1 shipped with a non-zero suppression default. Migrate that untouched
    // default once, while preserving any deliberate user adjustment.
    const isLegacyDefault = window.localStorage.getItem(AUDIO_SETTINGS_STORAGE_VERSION_KEY) === null
      && candidate.noiseSuppressionStrength === 0.35
      && candidate.echoControlStrength === 0;
    const settings = {
      ...DEFAULT_AUDIO_SETTINGS,
      ...candidate,
      ...(isLegacyDefault ? { noiseSuppressionStrength: 0 } : {}),
    };
    window.localStorage.setItem(AUDIO_SETTINGS_STORAGE_VERSION_KEY, "2");
    return settings;
  } catch {
    return DEFAULT_AUDIO_SETTINGS;
  }
}

function loadStoredModelSettings(): Record<string, RvcModelSettings> {
  try {
    const currentValue = window.localStorage.getItem(MODEL_SETTINGS_STORAGE_KEY);
    const value = currentValue ?? window.localStorage.getItem(LEGACY_MODEL_SETTINGS_STORAGE_KEY);
    if (!value) return {};
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== "object") return {};
    // v1 used 0.03 as the RMVPE default.  Migrate that untouched default to
    // the w-okada-compatible 0.30 threshold while retaining all other user
    // settings.  The versioned key prevents this from repeating.
    const migrated = Object.fromEntries(
      Object.entries(parsed).map(([id, settings]) => {
        if (settings && typeof settings === "object" && (settings as Record<string, unknown>).f0Threshold === 0.03) {
          return [id, { ...(settings as Record<string, unknown>), f0Threshold: 0.30 }];
        }
        return [id, settings];
      }),
    ) as Record<string, RvcModelSettings>;
    if (!currentValue) window.localStorage.setItem(MODEL_SETTINGS_STORAGE_KEY, JSON.stringify(migrated));
    return migrated;
  } catch {
    return {};
  }
}

function isStoredCalibrationResult(value: unknown): value is LiveCalibrationResult {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  if (typeof candidate.sampleRate !== "number"
    || !["quality", "balanced", "latency"].includes(String(candidate.recommendedPreset))
    || !["quality", "balanced", "latency"].includes(String(candidate.restoredPreset))
    || typeof candidate.message !== "string"
    || !Array.isArray(candidate.profiles)) return false;
  return candidate.profiles.every((profile) => {
    if (typeof profile !== "object" || profile === null) return false;
    const item = profile as Record<string, unknown>;
    return ["quality", "balanced", "latency"].includes(String(item.preset))
      && ["chunkFrames", "analysisFrames", "processMs", "deadlineMs", "headroomMs"].every((key) => typeof item[key] === "number")
      && (item.extraFrames === undefined || typeof item.extraFrames === "number")
      && typeof item.stable === "boolean";
  });
}

function loadStoredCalibrations(): Record<string, LiveCalibrationResult> {
  try {
    const value = window.localStorage.getItem(CALIBRATION_STORAGE_KEY);
    if (!value) return {};
    const parsed: unknown = JSON.parse(value);
    if (typeof parsed !== "object" || parsed === null) return {};
    return Object.fromEntries(
      Object.entries(parsed).filter(([, result]) => isStoredCalibrationResult(result)),
    ) as Record<string, LiveCalibrationResult>;
  } catch {
    return {};
  }
}

function isCoverImageAsset(value: unknown): value is CoverImageAsset {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.dataUrl === "string"
    && candidate.dataUrl.startsWith("data:image/")
    && typeof candidate.name === "string"
    && typeof candidate.type === "string";
}

function isStoredModelPreset(value: unknown): value is LibraryModel {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.id === "string"
    && typeof candidate.name === "string"
    && typeof candidate.initials === "string"
    && (candidate.format === "RVC v1" || candidate.format === "RVC v2" || candidate.format === "RVC ONNX")
    && (candidate.sampleRate === null || typeof candidate.sampleRate === "number")
    && typeof candidate.sourcePath === "string"
    && (candidate.coverImage === undefined || candidate.coverImage === null || isCoverImageAsset(candidate.coverImage))
    && (candidate.indexPaths === undefined || (Array.isArray(candidate.indexPaths) && candidate.indexPaths.every((path) => typeof path === "string")));
}

function loadStoredImportedModels(): LibraryModel[] {
  try {
    const value = window.localStorage.getItem(MODEL_LIBRARY_STORAGE_KEY);
    if (!value) return [];
    const parsed: unknown = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isStoredModelPreset)
      .filter((model, index, models) => models.findIndex((candidate) => sameWindowsPath(candidate.id, model.id)) === index)
      .map((model) => ({
        ...model,
        indexPaths: model.indexPaths?.filter(Boolean) ?? [],
        recommendedIndexPath: model.recommendedIndexPath ?? null,
        embedderPath: model.embedderPath ?? null,
        embedderExplicit: model.embedderExplicit ?? false,
        coverImage: model.coverImage ?? null,
      }));
  } catch {
    return [];
  }
}

function loadStoredModelId() {
  try {
    return window.localStorage.getItem(MODEL_SELECTION_STORAGE_KEY) || MODEL_PRESETS[0].id;
  } catch {
    return MODEL_PRESETS[0].id;
  }
}

function loadStoredStringList(key: string) {
  try {
    const value = window.localStorage.getItem(key);
    if (!value) return [];
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function normalizeModelFolder(value: string) {
  return value.trim().replace(/\s+/g, " ").slice(0, 40);
}

function normalizeModelTags(value: string | string[]) {
  const values = Array.isArray(value) ? value : value.split(/[\n,]/);
  const tags: string[] = [];
  values.forEach((item) => {
    const tag = item.trim().replace(/\s+/g, " ").slice(0, 24);
    if (!tag || tags.some((current) => current.toLocaleLowerCase() === tag.toLocaleLowerCase())) return;
    tags.push(tag);
  });
  return tags.slice(0, 8);
}

function loadStoredModelOrganization(): Record<string, ModelOrganization> {
  try {
    const value = window.localStorage.getItem(MODEL_ORGANIZATION_STORAGE_KEY);
    if (!value) return {};
    const parsed: unknown = JSON.parse(value);
    if (typeof parsed !== "object" || parsed === null) return {};
    return Object.fromEntries(Object.entries(parsed).flatMap(([id, organization]) => {
      if (typeof organization !== "object" || organization === null) return [];
      const candidate = organization as Record<string, unknown>;
      const folder = typeof candidate.folder === "string" ? normalizeModelFolder(candidate.folder) : "";
      const tags = Array.isArray(candidate.tags) ? normalizeModelTags(candidate.tags.filter((tag): tag is string => typeof tag === "string")) : [];
      return [[id, { folder, tags } satisfies ModelOrganization]];
    }));
  } catch {
    return {};
  }
}

function modelInitials(name: string) {
  return name.replace(/[^a-z0-9]/gi, "").slice(0, 2).toUpperCase() || "VC";
}

function loadStoredTheme(): "dark" | "light" {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function loadStoredDeviceSelection(): { inputDeviceId: string; outputDeviceId: string; monitorDeviceId: string } {
  const empty = { inputDeviceId: "", outputDeviceId: "", monitorDeviceId: "" };
  try {
    const value = window.localStorage.getItem(DEVICE_SETTINGS_STORAGE_KEY);
    if (!value) return empty;
    const parsed = JSON.parse(value);
    return {
      inputDeviceId: typeof parsed?.inputDeviceId === "string" ? parsed.inputDeviceId : "",
      outputDeviceId: typeof parsed?.outputDeviceId === "string" ? parsed.outputDeviceId : "",
      monitorDeviceId: typeof parsed?.monitorDeviceId === "string" ? parsed.monitorDeviceId : "",
    };
  } catch {
    return empty;
  }
}

function windowsFileName(path: string | null | undefined) {
  if (!path) return "None";
  return path.split(/[\\/]/).pop() || path;
}

function CoverImage({ coverImage, className, alt = "" }: { coverImage?: CoverImageAsset | null; className: string; alt?: string }) {
  const [failedDataUrl, setFailedDataUrl] = useState<string | null>(null);
  if (!coverImage || coverImage.dataUrl === failedDataUrl) return null;
  return <img className={className} src={coverImage.dataUrl} alt={alt} onError={() => setFailedDataUrl(coverImage.dataUrl)} />;
}

function readCoverImage(file: File): Promise<CoverImageAsset> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error("The selected cover image could not be read."));
        return;
      }
      resolve({ dataUrl: reader.result, name: file.name, type: file.type || "image/*" });
    };
    reader.onerror = () => reject(new Error("The selected cover image could not be read."));
    reader.readAsDataURL(file);
  });
}

function isSupportedCoverImage(file: File) {
  const extension = file.name.toLocaleLowerCase().split(".").pop() || "";
  return COVER_IMAGE_EXTENSIONS.includes(extension) && (file.type.startsWith("image/") || extension === "gif");
}

function App() {
  const [profile, setProfile] = useState<SystemProfile>(FALLBACK_PROFILE);
  const [devices, setDevices] = useState<AudioDeviceSnapshot>(FALLBACK_AUDIO_DEVICES);
  const [engineStatus, setEngineStatus] = useState<AudioEngineStatus>(STOPPED_ENGINE_STATUS);
  const [inferenceRuntime, setInferenceRuntime] = useState<InferenceRuntimeProbe>(FALLBACK_INFERENCE_RUNTIME);
  const [liveRvcStatus, setLiveRvcStatus] = useState<LiveRvcStatus>(EMPTY_LIVE_RVC_STATUS);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [calibrations, setCalibrations] = useState<Record<string, LiveCalibrationResult>>(loadStoredCalibrations);
  const [running, setRunning] = useState(false);
  const [engineBusy, setEngineBusy] = useState(false);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [inputDeviceId, setInputDeviceId] = useState(() => loadStoredDeviceSelection().inputDeviceId || FALLBACK_AUDIO_DEVICES.defaultInputId || "");
  const [outputDeviceId, setOutputDeviceId] = useState(() => loadStoredDeviceSelection().outputDeviceId || FALLBACK_AUDIO_DEVICES.defaultOutputId || "");
  const [monitorDeviceId, setMonitorDeviceId] = useState(() => loadStoredDeviceSelection().monitorDeviceId);
  const [modelId, setModelId] = useState(loadStoredModelId);
  const [modelSettings, setModelSettings] = useState<Record<string, RvcModelSettings>>(loadStoredModelSettings);
  const [audioSettings, setAudioSettings] = useState<AudioProcessingSettings>(loadStoredAudioSettings);
  const [importedModels, setImportedModels] = useState<LibraryModel[]>(loadStoredImportedModels);
  const [modelPackageOpen, setModelPackageOpen] = useState(false);
  const [modelPackageBusy, setModelPackageBusy] = useState<ModelPackageBusy>(null);
  const [modelPackage, setModelPackage] = useState<ModelPackageDraft>(() => ({ ...EMPTY_MODEL_PACKAGE, indexCandidates: [] }));
  const [modelFolderCandidates, setModelFolderCandidates] = useState<string[]>([]);
  const [modelPackageName, setModelPackageName] = useState("");
  const coverInputRef = useRef<HTMLInputElement>(null);
  const editCoverInputRef = useRef<HTMLInputElement>(null);
  const [modelLoadBusy, setModelLoadBusy] = useState(false);
  const [modelLoadPhase, setModelLoadPhase] = useState<ModelLoadPhase>(null);
  const [modelLoadProgress, setModelLoadProgress] = useState(0);
  const [modelLoadStartedAt, setModelLoadStartedAt] = useState<number | null>(null);
  const [modelLoadElapsedMs, setModelLoadElapsedMs] = useState(0);
  const [modelUnloadBusy, setModelUnloadBusy] = useState(false);
  const [modelUnloadConfirmOpen, setModelUnloadConfirmOpen] = useState(false);
  const [deviceRefreshBusy, setDeviceRefreshBusy] = useState(false);
  const [routeTestBusy, setRouteTestBusy] = useState(false);
  const [routeTestResult, setRouteTestResult] = useState<AudioRouteTestResult | null>(null);
  const [runtimeRefreshBusy, setRuntimeRefreshBusy] = useState(false);
  const [startupBusy, setStartupBusy] = useState(true);
  const [modelQuery, setModelQuery] = useState("");
  const [modelDrawerOpen, setModelDrawerOpen] = useState(false);
  const [modelMenuId, setModelMenuId] = useState<string | null>(null);
  const [modelEditorId, setModelEditorId] = useState<string | null>(null);
  const [modelRemovalId, setModelRemovalId] = useState<string | null>(null);
  const [modelDraftName, setModelDraftName] = useState("");
  const [modelDraftCoverImage, setModelDraftCoverImage] = useState<CoverImageAsset | null>(null);
  const [modelEditorCoverBusy, setModelEditorCoverBusy] = useState(false);
  const [favoriteModelIds, setFavoriteModelIds] = useState<string[]>(loadStoredStringList(MODEL_FAVORITES_STORAGE_KEY));
  const [recentModelIds, setRecentModelIds] = useState<string[]>(loadStoredStringList(MODEL_RECENTS_STORAGE_KEY));
  const [modelOrganization, setModelOrganization] = useState<Record<string, ModelOrganization>>(loadStoredModelOrganization);
  const [modelScope, setModelScope] = useState<ModelScope>("all");
  const [modelSort, setModelSort] = useState<ModelSort>("recent");
  const [modelFolderFilter, setModelFolderFilter] = useState("");
  const [modelTagFilter, setModelTagFilter] = useState("");
  const [modelOrganizationId, setModelOrganizationId] = useState<string | null>(null);
  const [modelOrganizationFolder, setModelOrganizationFolder] = useState("");
  const [modelOrganizationTags, setModelOrganizationTags] = useState("");
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [audioProcessingOpen, setAudioProcessingOpen] = useState(false);
  const [telemetryModalOpen, setTelemetryModalOpen] = useState(false);
  const [liveLogs, setLiveLogs] = useState<LiveLogEntry[]>([]);
  const liveLogIdRef = useRef(0);
  const deviceWatchSignatureRef = useRef("");
  const audioErrorRecoverySignatureRef = useRef("");
  const recoveryInFlightRef = useRef(false);
  const recoveryAttemptRef = useRef(0);
  const [quickSetupOpen, setQuickSetupOpen] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [libraryWidth, setLibraryWidth] = useState(() => loadStoredSidebarLayout().libraryWidth);
  const [sessionWidth, setSessionWidth] = useState(() => loadStoredSidebarLayout().sessionWidth);
  const [resizingSidebar, setResizingSidebar] = useState<SidebarName | null>(null);
  const sidebarResizeRef = useRef<{ sidebar: SidebarName; startX: number; startWidth: number } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(loadStoredTheme);
  const [activeTab, setActiveTab] = useState<"live" | "about">("live");

  function appendLiveLog(message: string, tone: LiveLogTone = "info") {
    const entry: LiveLogEntry = {
      id: ++liveLogIdRef.current,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      tone,
      message,
    };
    setLiveLogs((current) => [entry, ...current].slice(0, 50));
  }

  function sidebarWidthBounds(sidebar: SidebarName) {
    const compactLayout = window.innerWidth <= 1180;
    if (sidebar === "library") {
      const available = compactLayout
        ? MAX_LIBRARY_WIDTH
        : window.innerWidth - sessionWidth - DESKTOP_STUDIO_MIN_WIDTH;
      return {
        minimum: MIN_LIBRARY_WIDTH,
        maximum: Math.max(MIN_LIBRARY_WIDTH, Math.min(MAX_LIBRARY_WIDTH, available)),
      };
    }
    const available = window.innerWidth
      - (compactLayout ? 0 : libraryWidth)
      - (compactLayout ? COMPACT_STUDIO_MIN_WIDTH : DESKTOP_STUDIO_MIN_WIDTH);
    return {
      minimum: MIN_SESSION_WIDTH,
      maximum: Math.max(MIN_SESSION_WIDTH, Math.min(MAX_SESSION_WIDTH, available)),
    };
  }

  function setSidebarWidth(sidebar: SidebarName, nextWidth: number) {
    const bounds = sidebarWidthBounds(sidebar);
    const width = clamp(nextWidth, bounds.minimum, bounds.maximum);
    if (sidebar === "library") setLibraryWidth(width);
    else setSessionWidth(width);
  }

  function beginSidebarResize(sidebar: SidebarName, event: ReactPointerEvent<HTMLButtonElement>) {
    if (event.button !== 0) return;
    event.preventDefault();
    sidebarResizeRef.current = {
      sidebar,
      startX: event.clientX,
      startWidth: sidebar === "library" ? libraryWidth : sessionWidth,
    };
    setResizingSidebar(sidebar);
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function handleSidebarKeyDown(sidebar: SidebarName, event: ReactKeyboardEvent<HTMLButtonElement>) {
    const bounds = sidebarWidthBounds(sidebar);
    const currentWidth = sidebar === "library" ? libraryWidth : sessionWidth;
    if (event.key === "Home") {
      event.preventDefault();
      setSidebarWidth(sidebar, bounds.minimum);
    } else if (event.key === "End") {
      event.preventDefault();
      setSidebarWidth(sidebar, bounds.maximum);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const direction = sidebar === "library"
        ? event.key === "ArrowRight" ? 16 : -16
        : event.key === "ArrowLeft" ? 16 : -16;
      setSidebarWidth(sidebar, currentWidth + direction);
    }
  }

  function resetSidebarWidth(sidebar: SidebarName) {
    setSidebarWidth(sidebar, sidebar === "library" ? DEFAULT_LIBRARY_WIDTH : DEFAULT_SESSION_WIDTH);
  }

  useEffect(() => {
    if (!resizingSidebar) return;
    const handlePointerMove = (event: globalThis.PointerEvent) => {
      const activeResize = sidebarResizeRef.current;
      if (!activeResize) return;
      const delta = event.clientX - activeResize.startX;
      const nextWidth = activeResize.startWidth + (activeResize.sidebar === "library" ? delta : -delta);
      setSidebarWidth(activeResize.sidebar, nextWidth);
    };
    const stopResize = () => {
      sidebarResizeRef.current = null;
      setResizingSidebar(null);
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResize);
    window.addEventListener("pointercancel", stopResize);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResize);
      window.removeEventListener("pointercancel", stopResize);
    };
  }, [resizingSidebar, libraryWidth, sessionWidth]);

  useEffect(() => {
    let active = true;
    async function initializeDesktop() {
      const [profileResult, runtimeResult, liveResult, devicesResult] = await Promise.allSettled([
        getSystemProfile(),
        probeInferenceRuntime(),
        getLiveRvcStatus(),
        getAudioDevices(),
      ]);
      if (!active) return;

      if (profileResult.status === "fulfilled") setProfile(profileResult.value);
      if (runtimeResult.status === "fulfilled") setInferenceRuntime(runtimeResult.value);
      if (liveResult.status === "fulfilled") setLiveRvcStatus(liveResult.value);
      if (devicesResult.status === "fulfilled") {
        const snapshot = devicesResult.value;
        const storedDevices = loadStoredDeviceSelection();
        setDevices(snapshot);
        setInputDeviceId((current) => snapshot.inputs.some((device) => device.id === current)
          ? current
          : snapshot.inputs.some((device) => device.id === storedDevices.inputDeviceId)
            ? storedDevices.inputDeviceId
            : snapshot.defaultInputId ?? snapshot.inputs[0]?.id ?? "");
        setOutputDeviceId((current) => snapshot.outputs.some((device) => device.id === current)
          ? current
          : snapshot.outputs.some((device) => device.id === storedDevices.outputDeviceId)
            ? storedDevices.outputDeviceId
            : snapshot.defaultOutputId ?? snapshot.outputs[0]?.id ?? "");
        setMonitorDeviceId((current) => snapshot.outputs.some((device) => device.id === current)
          ? current
          : snapshot.outputs.some((device) => device.id === storedDevices.monitorDeviceId)
            ? storedDevices.monitorDeviceId
            : "");
      }

      const firstFailure = [runtimeResult, liveResult, devicesResult]
        .find((result) => result.status === "rejected");
      if (firstFailure?.status === "rejected") setEngineError(String(firstFailure.reason));
      setStartupBusy(false);
      appendLiveLog(
        firstFailure ? "Startup completed with local runtime warnings" : "Local runtime and audio devices ready",
        firstFailure ? "warning" : "success",
      );
    }
    void initializeDesktop();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!running || engineBusy || !["passthrough", "rvc"].includes(engineStatus.state)) return;
    const interval = window.setInterval(() => {
        getAudioEngineStatus()
        .then((status) => {
          setEngineStatus(status);
          if (status.lastError) setEngineError(status.lastError);
          if (status.state === "stopped") {
            setRunning(false);
            appendLiveLog("Audio streams stopped unexpectedly", "error");
          }
        })
        .catch((error: unknown) => setEngineError(String(error)));
    }, 500);
    return () => window.clearInterval(interval);
  }, [running, engineBusy, engineStatus.state]);

  useEffect(() => {
    if (!running || !["passthrough", "rvc"].includes(engineStatus.state)) {
      deviceWatchSignatureRef.current = "";
      recoveryAttemptRef.current = 0;
      return;
    }
    let active = true;
    let inFlight = false;
    const checkDeviceAvailability = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const snapshot = await getAudioDevices();
        if (!active) return;
        setDevices(snapshot);
        const missing: string[] = [];
        if (inputDeviceId && !snapshot.inputs.some((device) => device.id === inputDeviceId)) missing.push("microphone");
        if (outputDeviceId && !snapshot.outputs.some((device) => device.id === outputDeviceId)) missing.push("output");
        if (monitorDeviceId && !snapshot.outputs.some((device) => device.id === monitorDeviceId)) missing.push("monitor");
        const signature = missing.join(",");
        const previousSignature = deviceWatchSignatureRef.current;
        if (signature === previousSignature) return;
        deviceWatchSignatureRef.current = signature;
        if (missing.length) {
          const routes = missing.join(", ");
          const message = `The selected ${routes} device${missing.length > 1 ? "s are" : " is"} unavailable. Reconnect it, then use Restart audio.`;
          setEngineError(message);
          appendLiveLog(`Audio route unavailable · ${routes}`, "error");
        } else if (previousSignature) {
          // CPAL streams keep their original endpoint handles after a Windows device
          // disappears. Recreate them once the exact selected route is visible again;
          // this avoids leaving the user with a green-looking but silent session.
          appendLiveLog("Audio devices detected again · reconnecting the session", "warning");
          setNotice("Audio devices are back. Reconnecting the audio session…");
          if (!engineBusy && !recoveryInFlightRef.current && recoveryAttemptRef.current < 3) {
            recoveryInFlightRef.current = true;
            recoveryAttemptRef.current += 1;
            void recoverAudioSession();
          }
        }
      } catch {
        // Device enumeration can fail transiently while Windows rebuilds an endpoint.
      } finally {
        inFlight = false;
      }
    };
    void checkDeviceAvailability();
    const interval = window.setInterval(checkDeviceAvailability, 2_500);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [running, engineBusy, engineStatus.state, inputDeviceId, outputDeviceId, monitorDeviceId]);

  useEffect(() => {
    const callbackError = engineStatus.lastError?.trim() ?? "";
    if (!running || !["passthrough", "rvc"].includes(engineStatus.state)) {
      audioErrorRecoverySignatureRef.current = "";
      return;
    }
    if (!callbackError) {
      audioErrorRecoverySignatureRef.current = "";
      return;
    }
    // CPAL can report a stream callback failure while Windows still lists the
    // endpoint. The device watcher cannot detect that case, so recover once
    // per distinct error after a short grace period. Keep the existing retry
    // ceiling shared with device-loss recovery so a broken route does not
    // create an endless restart loop.
    if (
      engineBusy
      || recoveryInFlightRef.current
      || audioErrorRecoverySignatureRef.current === callbackError
      || recoveryAttemptRef.current >= 3
    ) return;
    audioErrorRecoverySignatureRef.current = callbackError;
    appendLiveLog(`Audio callback error · ${callbackError}`, "error");
    setNotice("Audio callback stopped responding. Reconnecting the route…");
    const timer = window.setTimeout(() => {
      if (engineBusy || recoveryInFlightRef.current || !running) return;
      recoveryInFlightRef.current = true;
      recoveryAttemptRef.current += 1;
      void recoverAudioSession();
    }, 700);
    return () => window.clearTimeout(timer);
  }, [running, engineBusy, engineStatus.lastError, engineStatus.state]);

  useEffect(() => {
    if (!running || engineStatus.state !== "rvc") return;
    let active = true;
    let inFlight = false;
    const refreshWorkerHealth = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const status = await getLiveRvcStatus();
        if (!active) return;
        setLiveRvcStatus(status);
        if (status.workerState === "failed") {
          setEngineError(status.lastWorkerError ?? "The RVC worker could not recover.");
        }
      } catch {
        // Audio status remains authoritative while a worker restart is in progress.
      } finally {
        inFlight = false;
      }
    };
    void refreshWorkerHealth();
    const interval = window.setInterval(refreshWorkerHealth, 2_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [running, engineStatus.state]);

  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(null), 3200);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  useEffect(() => {
    if (!modelLoadBusy || modelLoadStartedAt === null) return;
    const updateElapsed = () => setModelLoadElapsedMs(Math.max(0, Date.now() - modelLoadStartedAt));
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 250);
    return () => window.clearInterval(interval);
  }, [modelLoadBusy, modelLoadStartedAt]);

  useEffect(() => {
    if (!modelDrawerOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setModelDrawerOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [modelDrawerOpen]);

  const availableModels = useMemo<LibraryModel[]>(
    () => [...MODEL_PRESETS, ...importedModels].map((model) => ({
      ...model,
      folder: modelOrganization[model.id]?.folder ?? "",
      tags: modelOrganization[model.id]?.tags ?? [],
    })),
    [importedModels, modelOrganization],
  );
  const selectedModel = useMemo<LibraryModel>(
    () => availableModels.find((model) => model.id === modelId) ?? MODEL_PRESETS[0],
    [availableModels, modelId],
  );
  const modelDefaults = defaultModelSettings(selectedModel);
  const selectedSettings = {
    ...modelDefaults,
    ...modelSettings[selectedModel.id],
  };
  const calibrationResult = calibrations[selectedModel.id] ?? null;
  if (selectedSettings.indexPath && !selectedModel.indexPaths?.some((path) => sameWindowsPath(path, selectedSettings.indexPath))) {
    selectedSettings.indexPath = modelDefaults.indexPath;
  }
  const mode = selectedSettings.streamingPreset;
  const presetProfile = streamProfiles[mode];
  const streamProfile = {
    ...presetProfile,
    hop: selectedSettings.chunkFrames / 48,
    extra: selectedSettings.extraFrames / 48,
  };
  const pitch = selectedSettings.pitchShift;
  const indexRate = Math.round(selectedSettings.indexRatio * 100);
  const protection = Math.round(selectedSettings.protectRatio * 100);
  const audioProcessingSummary = `${audioSettings.inputGainDb > 0 ? "+" : ""}${audioSettings.inputGainDb} dB in · ${audioSettings.outputGainDb > 0 ? "+" : ""}${audioSettings.outputGainDb} dB out · NS ${audioSettings.noiseSuppressionStrength <= 0 ? "off" : `${Math.round(audioSettings.noiseSuppressionStrength * 100)}%`} · echo ${audioSettings.echoControlStrength <= 0 ? "off" : `${Math.round(audioSettings.echoControlStrength * 100)}%`} · HP ${audioSettings.highPassEnabled ? "on" : "off"}`;
  const modelFolders = useMemo(() => [...new Set(availableModels.map((model) => model.folder).filter(Boolean) as string[])].sort((left, right) => left.localeCompare(right)), [availableModels]);
  const modelTags = useMemo(() => [...new Set(availableModels.flatMap((model) => model.tags ?? []))].sort((left, right) => left.localeCompare(right)), [availableModels]);
  const orderedModels = useMemo(() => {
    const originalOrder = new Map(availableModels.map((model, index) => [model.id, index]));
    const recentOrder = new Map(recentModelIds.map((id, index) => [id, index]));
    const scopedModels = availableModels.filter((model) => {
      const matchesScope = modelScope === "all"
        || (modelScope === "recent" && recentOrder.has(model.id))
        || (modelScope === "favorites" && favoriteModelIds.includes(model.id));
      const matchesFolder = !modelFolderFilter
        || modelFolderFilter === "__uncategorized__" && !model.folder
        || model.folder === modelFolderFilter;
      const matchesTag = !modelTagFilter || model.tags?.some((tag) => tag.toLocaleLowerCase() === modelTagFilter.toLocaleLowerCase());
      return matchesScope && matchesFolder && matchesTag;
    });
    return [...scopedModels].sort((left, right) => {
      if (modelSort === "name") return left.name.localeCompare(right.name) || (originalOrder.get(left.id) ?? 0) - (originalOrder.get(right.id) ?? 0);
      if (modelSort === "format") return left.format.localeCompare(right.format) || left.name.localeCompare(right.name);
      if (modelSort === "folder") return (left.folder || "\uFFFF").localeCompare(right.folder || "\uFFFF") || left.name.localeCompare(right.name);
      const leftRecent = recentOrder.get(left.id);
      const rightRecent = recentOrder.get(right.id);
      if (leftRecent !== undefined || rightRecent !== undefined) {
        if (leftRecent === undefined) return 1;
        if (rightRecent === undefined) return -1;
        if (leftRecent !== rightRecent) return leftRecent - rightRecent;
      }
      const favoriteOrder = Number(favoriteModelIds.includes(left.id)) - Number(favoriteModelIds.includes(right.id));
      if (favoriteOrder !== 0) return -favoriteOrder;
      return (originalOrder.get(left.id) ?? 0) - (originalOrder.get(right.id) ?? 0);
    });
  }, [availableModels, favoriteModelIds, modelFolderFilter, modelScope, modelSort, modelTagFilter, recentModelIds]);
  const filteredModels = useMemo(() => {
    const query = modelQuery.trim().toLocaleLowerCase();
    if (!query) return orderedModels;
    return orderedModels.filter((model) => `${model.name} ${model.format} ${model.folder ?? ""} ${(model.tags ?? []).join(" ")}`.toLocaleLowerCase().includes(query));
  }, [orderedModels, modelQuery]);
  const modelLibraryLocked = running || modelLoadBusy || modelUnloadBusy || modelPackageOpen || Boolean(modelPackageBusy);
  const modelBeingEdited = importedModels.find((model) => model.id === modelEditorId);
  const modelBeingRemoved = importedModels.find((model) => model.id === modelRemovalId);
  const modelBeingOrganized = availableModels.find((model) => model.id === modelOrganizationId);
  const inputDevice = devices.inputs.find((device) => device.id === inputDeviceId);
  const outputDevice = devices.outputs.find((device) => device.id === outputDeviceId);
  const monitorDevice = devices.outputs.find((device) => device.id === monitorDeviceId);
  const selectedModelLoaded = liveRvcStatus.state === "ready"
    && sameWindowsPath(liveRvcStatus.modelPath, selectedModel.sourcePath);
  // Retrieval indexes are loaded together with the checkpoint. The resident
  // worker cannot swap one in place, so surface a pending change instead of
  // letting the UI drift away from the active voice.
  const selectedModelNeedsReload = selectedModelLoaded
    && !sameWindowsPath(selectedSettings.indexPath, liveRvcStatus.indexPath ?? null);
  const selectedModelCanLoad = selectedModel.format.startsWith("RVC") && Boolean(selectedModel.sourcePath);
  const selectedModelIsPreviewOnly = selectedModel.format === "RVC ONNX" && !selectedModel.sourcePath;
  const voiceNeedsLoad = selectedModelCanLoad && !selectedModelLoaded;
  const voiceNeedsReload = selectedModelCanLoad && selectedModelLoaded && selectedModelNeedsReload;
  const onnxCpuFallback = selectedModelLoaded && liveRvcStatus.backend === "onnx" && liveRvcStatus.device === "cpu";
  const selectedModelHasIndex = Boolean(selectedSettings.indexPath && selectedModel.indexPaths?.some((path) => sameWindowsPath(path, selectedSettings.indexPath)));
  const audioReady = Boolean(inputDevice && outputDevice);
  const sampleRateDifference = Boolean(
    inputDevice && outputDevice && (
      inputDevice.sampleRate !== outputDevice.sampleRate
      || Boolean(monitorDevice && monitorDevice.sampleRate !== inputDevice.sampleRate)
    ),
  );
  const conversionReady = selectedModelLoaded && !selectedModelNeedsReload;
  const signalActive = running;
  const workerRecovering = running && liveRvcStatus.workerState === "recovering";
  const hasInputSignal = signalActive && ["passthrough", "rvc"].includes(engineStatus.state) && engineStatus.inputPeak > 0.001;
  const hasOutputSignal = signalActive && ["passthrough", "rvc"].includes(engineStatus.state) && engineStatus.outputPeak > 0.001;
  const hasMonitorSignal = signalActive && Boolean(engineStatus.monitorDeviceId) && engineStatus.monitorPeak > 0.001;
  // A live input with a completely idle output is different from a quiet
  // microphone. Surface that distinction so a disconnected virtual cable or
  // stale Windows endpoint does not look like a healthy conversion session.
  const outputRouteStalled = signalActive
    && !workerRecovering
    && engineStatus.capturedFrames >= 4_800
    && engineStatus.inputPeak > 0.01
    && engineStatus.outputPeak <= 0.0005
    && (engineStatus.state === "passthrough" || engineStatus.inferenceCalls > 0);
  // A virtual cable can open successfully while delivering an all-zero
  // stream. Keep that separate from an output stall so users are told to fix
  // the selected microphone/VoiceMeeter bus instead of chasing model settings.
  const inputRouteSilent = signalActive
    && !workerRecovering
    && engineStatus.capturedFrames >= 96_000
    && engineStatus.inputPeak <= 0.0005
    && (engineStatus.state === "passthrough" || engineStatus.inferenceSilenceSuppressedCalls > 0);
  const inputPeakTone = peakTone(engineStatus.inputPeak);
  const outputPeakTone = peakTone(engineStatus.outputPeak);
  const clippingMessage = inputPeakTone === "clip" && outputPeakTone === "clip"
    ? "Input and output are clipping. Lower the input or output gain."
    : inputPeakTone === "clip"
      ? "Input is clipping. Lower input gain or move the microphone farther away."
      : "Output is clipping. Lower output gain or the source level.";
  const engineLabel = workerRecovering
    ? "Recovering voice engine"
    : !running
    ? "Ready to start"
    : engineStatus.state === "rvc"
      ? "Live RVC conversion"
      : engineStatus.state === "passthrough"
      ? "Native pipeline"
      : "Browser preview";
  const engineTone = !running ? "ready" : engineStatus.state === "preview" ? "preview" : "live";
  const completedSetupSteps = [audioReady, conversionReady, running].filter(Boolean).length;
  const sessionStatusLabel = modelLoadBusy
    ? "Loading voice"
    : modelUnloadBusy
      ? "Unloading voice"
    : modelPackageBusy
      ? modelPackageBusy === "adding" ? "Adding model" : modelPackageBusy === "cover" ? "Choosing cover" : "Choosing files"
      : startupBusy
        ? "Starting"
        : workerRecovering
          ? "Recovering"
        : !running
          ? "Ready"
            : engineStatus.state === "preview"
              ? "Preview"
              : "Audio live";
  const startBlockedReason = startupBusy
    ? "Still checking local devices and the inference runtime."
    : !audioReady
      ? "Choose a microphone and output first."
      : modelLoadBusy
          ? "Wait for the voice model to finish loading."
          : voiceNeedsReload
            ? "Reload the selected voice to apply the new retrieval index."
            : voiceNeedsLoad
              ? "Load the selected voice before starting conversion."
            : null;
  const startDisabled = running
    ? engineBusy
    : engineBusy || startupBusy || modelLoadBusy || !audioReady || voiceNeedsLoad || voiceNeedsReload;
  const startButtonLabel = running
    ? "Stop audio"
    : startupBusy
      ? "Preparing…"
      : modelLoadBusy
        ? "Loading voice…"
        : !audioReady
          ? "Choose audio"
          : voiceNeedsReload
            ? "Reload voice first"
            : voiceNeedsLoad
              ? "Load voice first"
            : "Start audio";
  const modelLoadStageLabel = modelLoadPhase === "preparing"
    ? "Preparing the local voice worker"
    : modelLoadPhase === "loading"
      ? "Loading the checkpoint"
      : "Finalizing the voice session";
  const modelLoadStageDetail = modelLoadPhase === "preparing"
    ? "Checking the selected files and starting the local worker."
    : modelLoadPhase === "loading"
      ? "Loading weights and configuring the selected retrieval settings."
      : "Warming the voice and preparing it for live conversion.";
  const modelLoadElapsedLabel = `${(modelLoadElapsedMs / 1_000).toFixed(1)} s elapsed`;
  const modelLoadSlow = modelLoadElapsedMs >= 15_000;
  const modelStatusFor = (model: ModelPreset) => {
    const loaded = liveRvcStatus.state === "ready" && sameWindowsPath(liveRvcStatus.modelPath, model.sourcePath);
    if (loaded && model.id === selectedModel.id && selectedModelNeedsReload) return { label: "Reload required", tone: "warning" };
    if (loaded) return { label: "Ready", tone: "ready" };
    if (!model.sourcePath) return { label: "Preview", tone: "preview" };
    if (model.format === "RVC ONNX") return { label: "ONNX · needs load", tone: "pending" };
    if (!model.indexPaths?.length) return { label: "No index · optional", tone: "neutral" };
    return { label: "Needs load", tone: "pending" };
  };
  const selectedModelStatus = modelStatusFor(selectedModel);
  const navigatorWithMemory = typeof navigator !== "undefined"
    ? navigator as Navigator & { deviceMemory?: number }
    : null;
  const cpuSummary = navigatorWithMemory?.hardwareConcurrency
    ? `${navigatorWithMemory.hardwareConcurrency} logical cores`
    : "Unavailable";
  const memorySummary = navigatorWithMemory?.deviceMemory
    ? `${navigatorWithMemory.deviceMemory} GB estimate`
    : "Not exposed";
  const cudaSummary = inferenceRuntime.torchRuntime.cudaAvailable
    ? inferenceRuntime.torchRuntime.cudaVersion ? `CUDA ${inferenceRuntime.torchRuntime.cudaVersion}` : "Available"
    : "Unavailable";
  const pythonSummary = inferenceRuntime.python.version || "Unavailable";
  const processLatencyMs = liveRvcStatus.processCalls > 0
    ? liveRvcStatus.lastProcessMs
    : engineStatus.lastInferenceMicros > 0 ? engineStatus.lastInferenceMicros / 1_000 : null;
  const maxInferenceMs = engineStatus.maxInferenceMicros > 0 ? engineStatus.maxInferenceMicros / 1_000 : null;
  const bufferedLatencyMs = engineStatus.sampleRate && engineStatus.bufferedFrames > 0
    ? (engineStatus.bufferedFrames / engineStatus.sampleRate) * 1_000
    : null;
  const xrunCount = engineStatus.underruns + engineStatus.overruns + engineStatus.monitorUnderruns + engineStatus.monitorOverruns;
  const performanceLabel = !running ? "Idle" : xrunCount > 0 ? "Attention" : "Live";
  const performanceTone = !running ? "neutral" : xrunCount > 0 ? "warning" : "success";

  useEffect(() => {
    const handleStartShortcut = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key !== "Enter") return;
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName)) return;
      event.preventDefault();
      if (startDisabled) {
        if (startBlockedReason) setNotice(startBlockedReason);
        return;
      }
      void toggleEngine();
    };
    document.addEventListener("keydown", handleStartShortcut);
    return () => document.removeEventListener("keydown", handleStartShortcut);
  }, [startBlockedReason, startDisabled]);

  useEffect(() => {
    if (!telemetryModalOpen) return;
    const handleTelemetryEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setTelemetryModalOpen(false);
    };
    document.addEventListener("keydown", handleTelemetryEscape);
    return () => document.removeEventListener("keydown", handleTelemetryEscape);
  }, [telemetryModalOpen]);

  function updateSelectedSettings(patch: Partial<RvcModelSettings>) {
    setModelSettings((current) => ({
      ...current,
      [selectedModel.id]: { ...selectedSettings, ...patch },
    }));
  }

  function resetSelectedModelSettings() {
    setModelSettings((current) => ({ ...current, [selectedModel.id]: defaultModelSettings(selectedModel) }));
    setNotice("Voice controls reset to their defaults");
  }

  function resetAudioSettings() {
    setAudioSettings(DEFAULT_AUDIO_SETTINGS);
    setNotice("Audio processing reset to its defaults");
  }

  function closeModelEditor() {
    setModelEditorId(null);
    setModelDraftName("");
    setModelDraftCoverImage(null);
    setModelEditorCoverBusy(false);
  }

  function closeModelOrganization() {
    setModelOrganizationId(null);
    setModelOrganizationFolder("");
    setModelOrganizationTags("");
  }

  function beginOrganizeModel(model: LibraryModel) {
    if (modelLibraryLocked) {
      setNotice("Stop audio and finish any model operation before organizing the library");
      return;
    }
    const organization = modelOrganization[model.id];
    setModelMenuId(null);
    setModelOrganizationId(model.id);
    setModelOrganizationFolder(organization?.folder ?? "");
    setModelOrganizationTags((organization?.tags ?? []).join(", "));
  }

  function toggleOrganizationTag(tag: string) {
    const tags = normalizeModelTags(modelOrganizationTags);
    const nextTags = tags.some((current) => current.toLocaleLowerCase() === tag.toLocaleLowerCase())
      ? tags.filter((current) => current.toLocaleLowerCase() !== tag.toLocaleLowerCase())
      : [...tags, tag];
    setModelOrganizationTags(nextTags.join(", "));
  }

  function saveModelOrganization() {
    const target = modelBeingOrganized;
    if (!target) {
      closeModelOrganization();
      return;
    }
    const folder = normalizeModelFolder(modelOrganizationFolder);
    const tags = normalizeModelTags(modelOrganizationTags);
    setModelOrganization((current) => {
      const next = { ...current };
      if (!folder && tags.length === 0) delete next[target.id];
      else next[target.id] = { folder, tags };
      return next;
    });
    closeModelOrganization();
    setNotice(`${target.name} organization saved`);
  }

  function beginEditModel(model: LibraryModel) {
    if (!model.sourcePath) return;
    if (modelLibraryLocked) {
      setNotice("Stop audio and finish any model operation before editing the library");
      return;
    }
    setModelMenuId(null);
    setModelEditorId(model.id);
    setModelDraftName(model.name);
    setModelDraftCoverImage(model.coverImage ?? null);
  }

  function saveModelEdit() {
    if (modelEditorCoverBusy) return;
    const target = modelBeingEdited;
    const nextName = modelDraftName.trim().replace(/\s+/g, " ");
    if (!target) {
      closeModelEditor();
      return;
    }
    if (!nextName) {
      setNotice("Enter a name for this voice");
      return;
    }
    if (importedModels.some((model) => model.id !== target.id && model.name.toLocaleLowerCase() === nextName.toLocaleLowerCase())) {
      setNotice("A voice with that name is already in the library");
      return;
    }
    setImportedModels((current) => current.map((model) => model.id === target.id
      ? { ...model, name: nextName, initials: modelInitials(nextName), coverImage: modelDraftCoverImage }
      : model));
    closeModelEditor();
    setNotice(`${nextName} updated`);
  }

  function requestRemoveModel(model: ModelPreset) {
    if (!model.sourcePath) return;
    if (modelLibraryLocked) {
      setNotice("Stop audio and finish any model operation before changing the library");
      return;
    }
    if (selectedModelLoaded && sameWindowsPath(model.sourcePath, selectedModel.sourcePath)) {
      setNotice("Choose another voice before removing the loaded model");
      return;
    }
    setModelMenuId(null);
    setModelRemovalId(model.id);
  }

  function removeModelFromLibrary() {
    const target = modelBeingRemoved;
    if (!target) {
      setModelRemovalId(null);
      return;
    }
    if (modelLibraryLocked || (selectedModelLoaded && sameWindowsPath(target.sourcePath, selectedModel.sourcePath))) {
      setModelRemovalId(null);
      setNotice("Choose another voice and stop audio before removing the loaded model");
      return;
    }
    setImportedModels((current) => current.filter((model) => model.id !== target.id));
    setFavoriteModelIds((current) => current.filter((id) => id !== target.id));
    setRecentModelIds((current) => current.filter((id) => id !== target.id));
    setModelOrganization((current) => {
      const next = { ...current };
      delete next[target.id];
      return next;
    });
    setModelSettings((current) => {
      const next = { ...current };
      delete next[target.id];
      return next;
    });
    if (modelId === target.id) setModelId(MODEL_PRESETS[0].id);
    setModelRemovalId(null);
    setNotice(`${target.name} removed from your library; the source file was kept`);
  }

  async function refreshDevices() {
    if (deviceRefreshBusy || running) return;
    setDeviceRefreshBusy(true);
    setEngineError(null);
    try {
      const snapshot = await getAudioDevices();
      setDevices(snapshot);
      setInputDeviceId((current) => snapshot.inputs.some((device) => device.id === current) ? current : snapshot.defaultInputId ?? snapshot.inputs[0]?.id ?? "");
      setOutputDeviceId((current) => snapshot.outputs.some((device) => device.id === current) ? current : snapshot.defaultOutputId ?? snapshot.outputs[0]?.id ?? "");
      setMonitorDeviceId((current) => snapshot.outputs.some((device) => device.id === current) ? current : "");
      setNotice("Audio devices refreshed");
      appendLiveLog("Audio devices refreshed", "success");
    } catch (error) {
      setEngineError(String(error));
      appendLiveLog(`Device refresh failed · ${String(error)}`, "error");
    } finally {
      setDeviceRefreshBusy(false);
    }
  }

  async function runRouteTest() {
    if (routeTestBusy || running || !outputDeviceId) return;
    setRouteTestBusy(true);
    setEngineError(null);
    try {
      const result = await testAudioRoutes(outputDeviceId, monitorDeviceId || null);
      setRouteTestResult(result);
      if (result.outputError) {
        setEngineError(`Output route test failed: ${result.outputError}`);
        appendLiveLog(`Output route test failed · ${result.outputError}`, "error");
      } else if (result.monitorError) {
        setNotice("Output callback passed; monitor route needs attention");
        appendLiveLog(`Output callback passed · monitor warning: ${result.monitorError}`, "warning");
      } else {
        setNotice("Output and monitor callbacks passed the test tone");
        appendLiveLog("Output and monitor callbacks passed the test tone", "success");
      }
    } catch (error) {
      setRouteTestResult(null);
      setEngineError(String(error));
      appendLiveLog(`Audio route test failed · ${String(error)}`, "error");
    } finally {
      setRouteTestBusy(false);
    }
  }

  async function refreshRuntime() {
    if (runtimeRefreshBusy || running || startupBusy) return;
    setRuntimeRefreshBusy(true);
    setEngineError(null);
    try {
      const runtime = await probeInferenceRuntime();
      setInferenceRuntime(runtime);
      setNotice(runtime.readyForRvc ? "RVC runtime is ready" : "Runtime checked · setup still needs attention");
      appendLiveLog(
        runtime.readyForRvc ? "RVC runtime checked · ready" : `Runtime checked · ${runtime.blockers.length} blocker${runtime.blockers.length === 1 ? "" : "s"}`,
        runtime.readyForRvc ? "success" : "warning",
      );
    } catch (error) {
      setEngineError(String(error));
      appendLiveLog(`Runtime check failed · ${String(error)}`, "error");
    } finally {
      setRuntimeRefreshBusy(false);
    }
  }

  async function copyRuntimeSetupCommand() {
    let command = "npm run runtime:setup";
    try {
      command = await getRuntimeSetupCommand();
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard access is unavailable.");
      await navigator.clipboard.writeText(command);
      setNotice("Runtime setup command copied to the clipboard");
      appendLiveLog("Runtime setup command copied", "info");
    } catch (error) {
      setEngineError(`Copy failed. Run this command in PowerShell to install the local CUDA runtime: ${command}`);
      appendLiveLog(`Could not prepare runtime setup command · ${String(error)}`, "error");
    }
  }

  async function launchRuntimeSetup() {
    if (runtimeRefreshBusy || startupBusy || running) return;
    setRuntimeRefreshBusy(true);
    setEngineError(null);
    try {
      const script = await openRuntimeSetup();
      setNotice("Runtime setup opened in PowerShell");
      appendLiveLog(`Runtime setup opened · ${windowsFileName(script)}`, "info");
    } catch (error) {
      setEngineError(String(error));
      appendLiveLog(`Could not open runtime setup · ${String(error)}`, "error");
    } finally {
      setRuntimeRefreshBusy(false);
    }
  }

  function handleChecklistStep(step: "audio" | "voice" | "session") {
    if (step === "audio") {
      document.querySelector<HTMLElement>(".setup-panel")?.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    if (step === "voice") {
      if (selectedModelIsPreviewOnly) {
        setNotice("Import an ONNX or .pth voice to enable live conversion");
        return;
      }
      if (selectedModelCanLoad) {
        void loadSelectedModel();
      } else {
        void importModel();
      }
      return;
    }
    if (startDisabled) {
      if (startBlockedReason) setNotice(startBlockedReason);
      return;
    }
    void toggleEngine();
  }

  function selectModel(id: string) {
    const model = availableModels.find((item) => item.id === id);
    setModelId(id);
    setModelMenuId(null);
    setModelDrawerOpen(false);
    setRecentModelIds((current) => [id, ...current.filter((recentId) => recentId !== id)].slice(0, 5));
    if (model) appendLiveLog(`Selected voice · ${model.name}`);
  }

  function toggleFavoriteModel(id: string) {
    setFavoriteModelIds((current) => current.includes(id) ? current.filter((favoriteId) => favoriteId !== id) : [id, ...current]);
    setNotice(favoriteModelIds.includes(id) ? "Voice removed from favorites" : "Voice pinned to favorites");
  }

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_SETTINGS_STORAGE_KEY, JSON.stringify(modelSettings));
    } catch {
      // Settings persistence is best-effort; inference remains fully local either way.
    }
  }, [modelSettings]);

  useEffect(() => {
    try {
      window.localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(calibrations));
    } catch {
      // Calibration history is best-effort; the current session remains usable.
    }
  }, [calibrations]);

  useEffect(() => {
    try {
      window.localStorage.setItem(AUDIO_SETTINGS_STORAGE_KEY, JSON.stringify(audioSettings));
    } catch {
      // Session audio settings are best-effort and remain local.
    }
  }, [audioSettings]);

  useEffect(() => {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Theme persistence is best-effort.
    }
  }, [theme]);

  useEffect(() => {
    try {
      window.localStorage.setItem(DEVICE_SETTINGS_STORAGE_KEY, JSON.stringify({ inputDeviceId, outputDeviceId, monitorDeviceId }));
    } catch {
      // Device preference persistence is best-effort; unavailable devices are revalidated on startup.
    }
  }, [inputDeviceId, outputDeviceId, monitorDeviceId]);

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_LIBRARY_STORAGE_KEY, JSON.stringify(importedModels));
    } catch {
      // Imported model persistence is best-effort; source files remain untouched.
    }
  }, [importedModels]);

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_SELECTION_STORAGE_KEY, modelId);
    } catch {
      // Selection persistence is best-effort.
    }
  }, [modelId]);

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_FAVORITES_STORAGE_KEY, JSON.stringify(favoriteModelIds));
    } catch {
      // Library preferences are best-effort and remain available for this session.
    }
  }, [favoriteModelIds]);

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_RECENTS_STORAGE_KEY, JSON.stringify(recentModelIds));
    } catch {
      // Recent voice ordering is best-effort.
    }
  }, [recentModelIds]);

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_ORGANIZATION_STORAGE_KEY, JSON.stringify(modelOrganization));
    } catch {
      // Model organization is best-effort and remains available for this session.
    }
  }, [modelOrganization]);

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_LAYOUT_STORAGE_KEY, JSON.stringify({ libraryWidth, sessionWidth }));
    } catch {
      // Sidebar layout persistence is best-effort.
    }
  }, [libraryWidth, sessionWidth]);

  useEffect(() => {
    if (completedSetupSteps === 3) setQuickSetupOpen(false);
  }, [completedSetupSteps]);

  useEffect(() => {
    if (!modelMenuId && !modelEditorId && !modelRemovalId && !modelOrganizationId && !modelPackageOpen && !modelUnloadConfirmOpen) return;
    const closeModelOverlays = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setModelMenuId(null);
      closeModelEditor();
      closeModelOrganization();
      setModelRemovalId(null);
      setModelDraftName("");
      if (modelPackageOpen && !modelPackageBusy) closeModelPackage();
      if (modelUnloadConfirmOpen && !modelUnloadBusy) setModelUnloadConfirmOpen(false);
    };
    document.addEventListener("keydown", closeModelOverlays);
    return () => document.removeEventListener("keydown", closeModelOverlays);
  }, [modelMenuId, modelEditorId, modelRemovalId, modelOrganizationId, modelPackageOpen, modelPackageBusy, modelUnloadConfirmOpen, modelUnloadBusy]);

  async function toggleEngine() {
    if (engineBusy) return;
    const wasRunning = running;
    setEngineBusy(true);
    setEngineError(null);
    appendLiveLog(wasRunning ? "Stopping audio session" : selectedModelLoaded ? `Starting RVC session · ${selectedModel.name}` : "Starting passthrough session");
    try {
      if (!wasRunning && conversionReady) {
        setLiveRvcStatus(await setLiveRvcSettings(selectedSettings));
      }
      const status = wasRunning
        ? await stopAudioEngine()
        : await startAudioEngine(inputDeviceId, outputDeviceId, monitorDeviceId || null, audioSettings);
      setEngineStatus(status);
      setRunning(status.state !== "stopped");
      appendLiveLog(
        status.state === "stopped" ? "Audio session stopped" : `Audio session live · ${status.state === "rvc" ? "RVC conversion" : "Passthrough"}`,
        status.state === "stopped" ? "info" : "success",
      );
      if (wasRunning) {
        getLiveRvcStatus().then(setLiveRvcStatus).catch(() => undefined);
      }
    } catch (error) {
      setRunning(false);
      setEngineStatus(STOPPED_ENGINE_STATUS);
      setEngineError(String(error));
      appendLiveLog(`Audio session failed · ${String(error)}`, "error");
    } finally {
      setEngineBusy(false);
    }
  }

  async function recoverAudioSession() {
    if (engineBusy || !running) return;
    setEngineBusy(true);
    setEngineError(null);
    appendLiveLog("Restarting native audio streams", "warning");
    try {
      const status = await restartAudioEngine();
      setEngineStatus(status);
      setRunning(status.state !== "stopped");
      if (status.state === "stopped") {
        throw new Error(status.lastError ?? "The audio session could not be restarted.");
      }
      recoveryAttemptRef.current = 0;
      appendLiveLog("Native audio streams recovered", "success");
      setNotice("Audio route recovered");
    } catch (error) {
      setRunning(false);
      setEngineStatus(STOPPED_ENGINE_STATUS);
      setEngineError(String(error));
      appendLiveLog(`Audio recovery failed · ${String(error)}`, "error");
    } finally {
      recoveryInFlightRef.current = false;
      setEngineBusy(false);
    }
  }

  async function exportDiagnostics() {
    const report = {
      generatedAt: new Date().toISOString(),
      application: "VC Next",
      profile: {
        os: profile.os,
        gpu: profile.gpu,
        vramMb: profile.vramMb,
        driverVersion: profile.driverVersion,
      },
      runtime: {
        source: inferenceRuntime.source,
        platform: inferenceRuntime.platform,
        pythonVersion: inferenceRuntime.python.version,
        cudaAvailable: inferenceRuntime.torchRuntime.cudaAvailable,
        cudaVersion: inferenceRuntime.torchRuntime.cudaVersion,
        deviceName: inferenceRuntime.torchRuntime.deviceName,
        readyForRvc: inferenceRuntime.readyForRvc,
        blockers: inferenceRuntime.blockers,
      },
      devices: {
        input: inputDevice ? { name: inputDevice.name, sampleRate: inputDevice.sampleRate, channels: inputDevice.channels } : null,
        output: outputDevice ? { name: outputDevice.name, sampleRate: outputDevice.sampleRate, channels: outputDevice.channels } : null,
        monitor: monitorDevice ? { name: monitorDevice.name, sampleRate: monitorDevice.sampleRate, channels: monitorDevice.channels } : null,
        backend: devices.backend,
      },
      voice: {
        name: selectedModel.name,
        format: selectedModel.format,
        sampleRate: selectedModel.sampleRate,
        retrievalIndexSelected: selectedModelHasIndex,
        loaded: selectedModelLoaded,
        settings: selectedSettings,
      },
      audioProcessing: audioSettings,
      routeTest: routeTestResult,
      engine: engineStatus,
      liveRvc: {
        ...liveRvcStatus,
        modelPath: liveRvcStatus.modelPath ? windowsFileName(liveRvcStatus.modelPath) : null,
        contentvecPath: liveRvcStatus.contentvecPath ? windowsFileName(liveRvcStatus.contentvecPath) : null,
        rmvpePath: liveRvcStatus.rmvpePath ? windowsFileName(liveRvcStatus.rmvpePath) : null,
        indexPath: liveRvcStatus.indexPath ? windowsFileName(liveRvcStatus.indexPath) : null,
      },
      events: liveLogs.slice(0, 50),
    };
    const contents = JSON.stringify(report, null, 2);
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(contents);
        setNotice("Diagnostic report copied to the clipboard");
        return;
      }
    } catch {
      // Fall through to a local download when clipboard access is unavailable.
    }
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([contents], { type: "application/json" }));
    link.download = `vc-next-diagnostics-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
    setNotice("Diagnostic report downloaded");
  }

  async function loadSelectedModel() {
    if (modelLoadBusy || modelUnloadBusy || running || !selectedModel.sourcePath || !selectedModelCanLoad) return;
    setModelLoadBusy(true);
    setModelLoadStartedAt(Date.now());
    setModelLoadElapsedMs(0);
    setModelLoadPhase("preparing");
    setModelLoadProgress(14);
    setEngineError(null);
    appendLiveLog(`Loading voice · ${selectedModel.name}`);
    try {
      await new Promise<void>((resolve) => window.setTimeout(resolve, 90));
      setModelLoadPhase("loading");
      setModelLoadProgress(48);
      let settingsForLoad = selectedSettings;
      let modelForLoad: ModelPreset = selectedModel;
      if (hasStaleAutoDiscoveredRinna(selectedModel)) {
        modelForLoad = { ...selectedModel, embedderPath: null, embedderExplicit: false };
        settingsForLoad = { ...selectedSettings, contentvecPath: null };
        setModelSettings((current) => ({ ...current, [selectedModel.id]: settingsForLoad }));
        setNotice("Updated the voice to w-okada's canonical ContentVec asset");
      }
      // Older library entries were created before params.json compatibility was
      // added. Inspect once at load time so those entries receive the same
      // model-specific defaults as newly imported packages.
      if (!selectedModel.modelDefaults) {
        try {
          const inspection = await inspectRvcModel(selectedModel.sourcePath);
          if (inspection.modelDefaults && Object.keys(inspection.modelDefaults).length > 0) {
            modelForLoad = {
              ...selectedModel,
              modelDefaults: inspection.modelDefaults,
              recommendedIndexPath: selectedModel.recommendedIndexPath ?? inspection.recommendedIndex ?? null,
              indexPaths: selectedModel.indexPaths?.length ? selectedModel.indexPaths : inspection.siblingIndexes,
            };
            if (hasStaleAutoDiscoveredRinna(modelForLoad)) {
              // Migrate entries created by the old resolver without overriding
              // an embedder the user explicitly selected in the package dialog.
              modelForLoad = { ...modelForLoad, embedderPath: null, embedderExplicit: false };
              settingsForLoad = { ...selectedSettings, contentvecPath: null };
              setModelSettings((current) => ({ ...current, [selectedModel.id]: settingsForLoad }));
              setNotice("Updated the voice to w-okada's canonical ContentVec asset");
            }
            if (isLegacyPrototypeSettings(selectedSettings, selectedModel)) {
              settingsForLoad = defaultModelSettings(modelForLoad);
              setModelSettings((current) => ({ ...current, [selectedModel.id]: settingsForLoad }));
              setNotice("Applied the voice package's w-okada settings");
            }
            setImportedModels((current) => current.map((model) => (
              sameWindowsPath(model.id, selectedModel.id) ? modelForLoad as LibraryModel : model
            )));
          }
        } catch {
          // The load call below retains the authoritative error if inspection
          // cannot read a stale or moved package.
        }
      }
      const status = await loadLiveRvcModel(selectedModel.sourcePath, settingsForLoad);
      setModelLoadPhase("finalizing");
      setModelLoadProgress(86);
      setLiveRvcStatus(status);
      setNotice(`${selectedModel.name} is loaded and ready`);
      appendLiveLog(`Voice loaded · ${selectedModel.name}`, "success");
      setImportedModels((current) => current.map((model) => (
        sameWindowsPath(model.sourcePath, status.modelPath)
          ? {
              ...model,
              format: status.rvcVersion === "v1" ? "RVC v1" : status.rvcVersion === "v2" ? "RVC v2" : model.format,
              sampleRate: status.targetSampleRate ?? model.sampleRate,
            }
          : model
      )));
      setModelLoadProgress(100);
      await new Promise<void>((resolve) => window.setTimeout(resolve, 240));
    } catch (error) {
      setEngineError(String(error));
      appendLiveLog(`Voice load failed · ${String(error)}`, "error");
    } finally {
      setModelLoadBusy(false);
      setModelLoadStartedAt(null);
      setModelLoadPhase(null);
      setModelLoadProgress(0);
    }
  }

  async function calibrateSelectedModel() {
    if (calibrationBusy || modelLoadBusy || modelUnloadBusy || running || !selectedModelLoaded || selectedModelNeedsReload) return;
    setCalibrationBusy(true);
    setEngineError(null);
    appendLiveLog(`Calibrating stream profiles · ${selectedModel.name}`);
    try {
      const result = await calibrateLiveRvc();
      setCalibrations((current) => ({ ...current, [selectedModel.id]: result }));
      appendLiveLog(`Stream calibration complete · recommend ${result.recommendedPreset}`, "success");
      setNotice(`Calibration recommends ${modeLabels[result.recommendedPreset]}`);
    } catch (error) {
      setEngineError(String(error));
      appendLiveLog(`Stream calibration failed · ${String(error)}`, "error");
    } finally {
      setCalibrationBusy(false);
    }
  }

  function applyCalibration(preset: ConversionMode) {
    const measurement = calibrationResult?.profiles.find((profile) => profile.preset === preset);
    if (!measurement) return;
    updateSelectedSettings({
      streamingPreset: preset,
      chunkFrames: measurement.chunkFrames,
      extraFrames: measurement.extraFrames ?? measurement.analysisFrames,
    });
    setNotice(`${modeLabels[preset]} stream settings applied`);
  }

  function requestUnloadSelectedModel() {
    if (modelUnloadBusy || modelLoadBusy || !selectedModelLoaded) return;
    if (running) {
      setNotice("Stop audio before unloading the voice");
      return;
    }
    setModelUnloadConfirmOpen(true);
  }

  async function unloadSelectedModel() {
    if (modelUnloadBusy || modelLoadBusy || running || !selectedModelLoaded) return;
    setModelUnloadBusy(true);
    setEngineError(null);
    appendLiveLog(`Unloading voice · ${selectedModel.name}`);
    try {
      const status = await unloadLiveRvcModel();
      setLiveRvcStatus(status);
      setModelUnloadConfirmOpen(false);
      setNotice(`${selectedModel.name} unloaded from memory`);
      appendLiveLog(`Voice unloaded · ${selectedModel.name}`, "success");
    } catch (error) {
      setEngineError(String(error));
      appendLiveLog(`Voice unload failed · ${String(error)}`, "error");
    } finally {
      setModelUnloadBusy(false);
    }
  }

  function importModel() {
    if (running || modelLoadBusy || modelPackageOpen || modelPackageBusy) return;
    setModelMenuId(null);
    setModelDrawerOpen(false);
    setModelPackage({ ...EMPTY_MODEL_PACKAGE, indexCandidates: [] });
    setModelFolderCandidates([]);
    setModelPackageName("");
    setModelPackageBusy(null);
    setEngineError(null);
    setModelPackageOpen(true);
  }

  function closeModelPackage() {
    if (modelPackageBusy) return;
    setModelPackageOpen(false);
    setModelPackage({ ...EMPTY_MODEL_PACKAGE, indexCandidates: [] });
    setModelFolderCandidates([]);
    setModelPackageName("");
  }

  async function inspectCheckpointPath(selected: string) {
    const inspection = await inspectRvcModel(selected);
    if (!(inspection.extension === ".pth" || inspection.extension === ".onnx") || !["rvc-checkpoint", "onnx-model"].includes(inspection.role)) {
      throw new Error("Choose a valid .pth or exported RVC .onnx model.");
    }
    const indexCandidates = [inspection.recommendedIndex, ...inspection.siblingIndexes]
      .filter((path): path is string => Boolean(path))
      .filter((path, index, paths) => paths.findIndex((candidate) => sameWindowsPath(candidate, path)) === index);
    setModelPackage({
      checkpointPath: selected,
      inspection,
      indexPath: inspection.recommendedIndex ?? indexCandidates[0] ?? null,
      indexCandidates,
      embedderPath: null,
      coverImage: null,
    });
    setModelFolderCandidates([]);
    setModelPackageName(inspection.name.replace(/\.(pth|onnx)$/i, ""));
  }

  async function selectModelCheckpoint() {
    if (modelPackageBusy) return;
    setModelPackageBusy("checkpoint");
    setEngineError(null);
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        title: "Select an RVC model (.pth or .onnx)",
        filters: [{ name: "RVC voice models", extensions: ["pth", "onnx"] }],
      });
      if (typeof selected !== "string") return;
      await inspectCheckpointPath(selected);
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelPackageBusy(null);
    }
  }

  async function selectModelFolder() {
    if (modelPackageBusy) return;
    setModelPackageBusy("folder");
    setEngineError(null);
    try {
      const selected = await open({
        multiple: false,
        directory: true,
        title: "Choose a w-okada model folder",
      });
      if (typeof selected !== "string") return;
      const candidates = await discoverRvcModels(selected);
      if (candidates.length === 1) {
        await inspectCheckpointPath(candidates[0]);
      } else {
        setModelFolderCandidates(candidates);
        setNotice(`Found ${candidates.length} voice models. Choose the checkpoint to import.`);
      }
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelPackageBusy(null);
    }
  }

  async function chooseModelFolderCandidate(path: string) {
    if (modelPackageBusy) return;
    setModelPackageBusy("checkpoint");
    setEngineError(null);
    try {
      await inspectCheckpointPath(path);
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelPackageBusy(null);
    }
  }

  async function selectModelIndex() {
    if (modelPackageBusy || !modelPackage.inspection) return;
    setModelPackageBusy("index");
    setEngineError(null);
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        title: "Select the matching FAISS index (.index)",
        filters: [{ name: "RVC retrieval indexes", extensions: ["index"] }],
      });
      if (typeof selected !== "string") return;
      if (!selected.toLocaleLowerCase().endsWith(".index")) {
        throw new Error("Choose a valid .index retrieval file.");
      }
      setModelPackage((current) => ({
        ...current,
        indexPath: selected,
        indexCandidates: [selected, ...current.indexCandidates]
          .filter((path, index, paths) => paths.findIndex((candidate) => sameWindowsPath(candidate, path)) === index),
      }));
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelPackageBusy(null);
    }
  }

  async function selectModelEmbedder() {
    if (modelPackageBusy || !modelPackage.inspection) return;
    setModelPackageBusy("embedder");
    setEngineError(null);
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        title: "Select a feature embedder (.onnx or .pt)",
        filters: [{ name: "ContentVec or HuBERT embedders", extensions: ["onnx", "pt", "pth"] }],
      });
      if (typeof selected !== "string") return;
      const extension = selected.toLocaleLowerCase();
      if (!(extension.endsWith(".onnx") || extension.endsWith(".pt") || extension.endsWith(".pth"))) {
        throw new Error("Choose a valid ContentVec (.onnx) or Fairseq HuBERT (.pt/.pth) embedder file.");
      }
      setModelPackage((current) => ({ ...current, embedderPath: selected }));
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelPackageBusy(null);
    }
  }

  async function handleModelCoverChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file || modelPackageBusy) return;
    if (!isSupportedCoverImage(file)) {
      setEngineError("Choose a PNG, JPG, WEBP, BMP, or GIF cover image.");
      return;
    }
    setModelPackageBusy("cover");
    setEngineError(null);
    try {
      const coverImage = await readCoverImage(file);
      setModelPackage((current) => ({ ...current, coverImage }));
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelPackageBusy(null);
    }
  }

  async function handleEditCoverChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file || modelEditorCoverBusy) return;
    if (!isSupportedCoverImage(file)) {
      setEngineError("Choose a PNG, JPG, WEBP, BMP, or GIF icon image.");
      return;
    }
    setModelEditorCoverBusy(true);
    setEngineError(null);
    try {
      setModelDraftCoverImage(await readCoverImage(file));
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelEditorCoverBusy(false);
    }
  }

  function addModelFromPackage() {
    if (modelPackageBusy || !modelPackage.inspection || !modelPackage.checkpointPath) return;
    const displayName = modelPackageName.trim().replace(/\s+/g, " ");
    if (!displayName) {
      setNotice("Give the voice a display name before adding it");
      return;
    }
    setModelPackageBusy("adding");
    const inspection = modelPackage.inspection;
    const indexPaths = [modelPackage.indexPath, ...modelPackage.indexCandidates]
      .filter((path): path is string => Boolean(path))
      .filter((path, index, paths) => paths.findIndex((candidate) => sameWindowsPath(candidate, path)) === index);
    const model: LibraryModel = {
      id: inspection.path,
      name: displayName,
      initials: modelInitials(displayName),
      format: inspection.extension === ".onnx" ? "RVC ONNX" : "RVC v2",
      sampleRate: null,
      sourcePath: inspection.path,
      indexPaths,
      recommendedIndexPath: modelPackage.indexPath ?? inspection.recommendedIndex ?? null,
      embedderPath: modelPackage.embedderPath,
      embedderExplicit: Boolean(modelPackage.embedderPath),
      modelDefaults: inspection.modelDefaults,
      coverImage: modelPackage.coverImage,
      pairingNote: modelPackage.indexPath
        ? "The selected .index file will be loaded with this checkpoint."
        : inspection.pairingNote,
    };
    const previousModel = importedModels.find((item) => sameWindowsPath(item.id, model.id));
    const alreadyImported = importedModels.some((item) => sameWindowsPath(item.id, model.id));
    setImportedModels((current) => [model, ...current.filter((item) => !sameWindowsPath(item.id, model.id))]);
    // A re-import can be the first time this package's params.json is seen. In
    // that case replace stale prototype defaults while preserving any settings
    // the user has already customized after metadata was applied.
    if (!previousModel?.modelDefaults && model.modelDefaults) {
      setModelSettings((current) => ({ ...current, [model.id]: defaultModelSettings(model) }));
    }
    selectModel(model.id);
    setModelPackageBusy(null);
    setModelPackageOpen(false);
    setModelPackage({ ...EMPTY_MODEL_PACKAGE, indexCandidates: [] });
    setModelFolderCandidates([]);
    setModelPackageName("");
    setNotice(alreadyImported
      ? `${displayName} updated in your library`
      : indexPaths.length
        ? `${displayName} added with its matching retrieval index`
        : `${displayName} added without a retrieval index`);
  }

  return (
    <div
      className="app-frame"
      data-theme={theme}
      aria-busy={modelLoadBusy || modelUnloadBusy || startupBusy || Boolean(modelPackageBusy)}
      style={{ "--library-width": `${libraryWidth}px`, "--session-width": `${sessionWidth}px` } as CSSProperties}
    >
      <header className="app-header" data-tauri-drag-region>
        <div className="brand-block">
          <span className="brand-mark"><img className="brand-icon" src="/vc-next-icon.png" alt="" /></span>
          <span><strong>VC Next</strong><small>Local voice studio</small></span>
        </div>

        <nav className="primary-nav" aria-label="Main navigation">
          <button className={activeTab === "live" ? "active" : ""} aria-current={activeTab === "live" ? "page" : undefined} onClick={() => setActiveTab("live")}><Icon name="activity" />Live</button>
          <button className={activeTab === "about" ? "active" : ""} aria-current={activeTab === "about" ? "page" : undefined} onClick={() => { setActiveTab("about"); setModelDrawerOpen(false); }}><Icon name="info" />About</button>
        </nav>

        <div className="header-actions">
          <button className="compact-voice-button header-voice-button" aria-expanded={modelDrawerOpen} aria-controls="voice-library" onClick={() => setModelDrawerOpen(true)}>
            <Icon name="library" size={15} />
            <span>Voices</span>
          </button>
          <button className="icon-button theme-toggle" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={() => setTheme((value) => value === "dark" ? "light" : "dark")}>
            <Icon name={theme === "dark" ? "sun" : "moon"} />
            <span>Theme</span>
          </button>
          <button className={`start-button ${running ? "stop" : ""}`} onClick={toggleEngine} aria-pressed={running} aria-keyshortcuts="Control+Enter" title={startBlockedReason ?? "Start or stop audio with Ctrl+Enter"} disabled={startDisabled}>
            <Icon name={running ? "stop" : "play"} />
            {engineBusy ? "Working…" : startButtonLabel}
          </button>
        </div>
      </header>

      {notice && <div className="toast" role="status" aria-live="polite">{notice}</div>}
      {modelDrawerOpen && <button className="model-drawer-shade" aria-label="Close voice library" onClick={() => setModelDrawerOpen(false)} />}

      <div
        className={`workspace ${activeTab === "about" ? "workspace-hidden" : ""} ${modelDrawerOpen ? "library-open" : ""} ${resizingSidebar ? "is-resizing" : ""}`}
      >
        <button
          type="button"
          className="sidebar-resize-handle library-resize-handle"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize voice library sidebar"
          aria-valuemin={sidebarWidthBounds("library").minimum}
          aria-valuemax={sidebarWidthBounds("library").maximum}
          aria-valuenow={Math.round(libraryWidth)}
          title="Drag to resize voice library · Double-click to reset"
          onPointerDown={(event) => beginSidebarResize("library", event)}
          onDoubleClick={() => resetSidebarWidth("library")}
          onKeyDown={(event) => handleSidebarKeyDown("library", event)}
        />
        <aside className={`model-panel ${modelDrawerOpen ? "open" : ""}`} id="voice-library">
          <div className="panel-heading">
            <div><span className="eyebrow">Voice models</span><h2>Your library <span className="library-count" aria-label={`${availableModels.length} models`}>{availableModels.length}</span></h2></div>
            <button className="icon-button subtle" aria-label="Add model" onClick={importModel} disabled={modelLibraryLocked}><Icon name="plus" /></button>
          </div>

          <label className="search-field">
            <Icon name="search" size={17} />
            <input placeholder="Search your models" aria-label="Search models" value={modelQuery} onChange={(event) => setModelQuery(event.target.value)} />
            {modelQuery && <button type="button" className="search-clear" aria-label="Clear model search" onMouseDown={(event) => event.preventDefault()} onClick={() => setModelQuery("")}><Icon name="close" size={14} /></button>}
          </label>

          <div className="model-library-controls" aria-label="Organize voice library">
            <div className="model-library-control-row">
              <select aria-label="Filter voices" value={modelScope} onChange={(event) => setModelScope(event.target.value as ModelScope)}>
                <option value="all">All voices</option>
                <option value="recent">Recently used</option>
                <option value="favorites">Favorites</option>
              </select>
              <select aria-label="Sort voices" value={modelSort} onChange={(event) => setModelSort(event.target.value as ModelSort)}>
                <option value="recent">Recently used</option>
                <option value="name">Name A–Z</option>
                <option value="format">Format</option>
                <option value="folder">Folder</option>
              </select>
            </div>
            <div className="model-library-control-row">
              <select aria-label="Filter by folder" value={modelFolderFilter} onChange={(event) => setModelFolderFilter(event.target.value)}>
                <option value="">All folders</option>
                {modelFolders.map((folder) => <option key={folder} value={folder}>{folder}</option>)}
                {availableModels.some((model) => !model.folder) && <option value="__uncategorized__">Unsorted</option>}
              </select>
              <select aria-label="Filter by tag" value={modelTagFilter} onChange={(event) => setModelTagFilter(event.target.value)}>
                <option value="">All tags</option>
                {modelTags.map((tag) => <option key={tag} value={tag}>#{tag}</option>)}
              </select>
            </div>
          </div>

          <div className="model-list" aria-label="Local voice models" role="listbox">
            {filteredModels.length > 0 ? filteredModels.map((model, index) => (
              <div key={model.id} role="option" aria-selected={modelId === model.id} className={`model-row ${modelId === model.id ? "selected" : ""}`}>
                <button type="button" className="model-select-button" onClick={() => selectModel(model.id)} disabled={modelLibraryLocked}>
                  <span className={`model-art art-${(index % 4) + 1}`}>
                    <span className="model-art-initials">{model.initials}</span>
                    <CoverImage coverImage={model.coverImage} className="model-art-image" />
                  </span>
                  <span className="model-copy"><strong>{model.name}</strong><small>{model.format} · {model.sampleRate ? `${model.sampleRate / 1000} kHz` : "Needs validation"}</small>{Boolean(model.folder || (model.tags ?? []).length > 0) ? <span className="model-organization-summary" title={[model.folder || "Unsorted", ...(model.tags ?? []).map((tag) => `#${tag}`)].join(" · ")}>{model.folder || "Unsorted"}{model.tags?.slice(0, 2).map((tag) => ` · #${tag}`)}{(model.tags?.length ?? 0) > 2 ? " · …" : ""}</span> : null}</span>
                  <span className={`model-status ${modelStatusFor(model).tone}`}>{modelStatusFor(model).label}</span>
                </button>
                <button type="button" className={`model-favorite-button ${favoriteModelIds.includes(model.id) ? "active" : ""}`} aria-label={`${favoriteModelIds.includes(model.id) ? "Remove" : "Add"} ${model.name} ${favoriteModelIds.includes(model.id) ? "from" : "to"} favorites`} aria-pressed={favoriteModelIds.includes(model.id)} title={favoriteModelIds.includes(model.id) ? "Remove from favorites" : "Add to favorites"} onClick={() => toggleFavoriteModel(model.id)} disabled={modelLibraryLocked}>{favoriteModelIds.includes(model.id) ? "★" : "☆"}</button>
                <button type="button" className={`model-row-action icon-button subtle ${modelMenuId === model.id ? "active" : ""}`} aria-label={`Manage ${model.name}`} aria-expanded={modelMenuId === model.id} onClick={() => setModelMenuId((current) => current === model.id ? null : model.id)} disabled={modelLibraryLocked}><Icon name="dots" size={16} /></button>
                {modelMenuId === model.id && <div className="model-row-menu" role="menu">
                  <button type="button" role="menuitem" onClick={() => beginOrganizeModel(model)}>Organize tags &amp; folder</button>
                  {model.sourcePath && <button type="button" role="menuitem" onClick={() => beginEditModel(model)}>Rename &amp; cover</button>}
                  {model.sourcePath && <button type="button" className="danger-menu-item" role="menuitem" onClick={() => requestRemoveModel(model)}>Remove from library</button>}
                </div>}
              </div>
            )) : (
              <div className="model-empty">
                <strong>No voices found</strong>
                <small>Try a different name or format.</small>
                {modelQuery && <button type="button" className="text-button" onClick={() => setModelQuery("")}>Clear search</button>}
              </div>
            )}
          </div>

          <div className="library-spacer" />
          <button className="drop-model-button" onClick={importModel} disabled={modelLibraryLocked}>
            <span className="drop-icon"><Icon name="plus" /></span>
            <span><strong>{modelPackageBusy ? "Preparing model package…" : modelLibraryLocked ? "Stop audio to edit library" : "Import a voice model"}</strong><small>Model · optional .index · optional embedder</small></span>
          </button>
          <p className="library-note">Reference profiles are fixed. Imported entries persist locally; removing one never deletes its source file.</p>
        </aside>

        <main className="studio-scroll">
          <div className="studio-content">
            <section className="voice-overview">
              <div className="voice-art-large">
                <span className="voice-art-initials">{selectedModel.initials}</span>
                <CoverImage coverImage={selectedModel.coverImage} className="voice-art-image" />
                <span className="voice-art-status" />
              </div>
              <div className="voice-title">
                <span className="eyebrow">{selectedModelLoaded ? "Active voice" : "Selected voice"}</span>
                <h1>{selectedModel.name}</h1>
                <div className="metadata-row">
                  <span>{selectedModel.format}</span>
                  <span>{selectedModel.sampleRate ? `${selectedModel.sampleRate / 1000} kHz` : "Unverified rate"}</span>
                  {selectedModel.sourcePath && <span>{selectedModel.indexPaths?.length ? "Retrieval index paired" : "Retrieval index off"}</span>}
                  <span className={`model-state-badge ${selectedModelStatus.tone}`}>{selectedModelNeedsReload ? "Reload required" : selectedModelLoaded ? "Loaded and warmed" : modelLoadBusy ? modelLoadStageLabel : selectedModelStatus.label}</span>
                </div>
                {(selectedModel.pairingNote || selectedModelIsPreviewOnly) && <p className={`voice-note ${selectedModelIsPreviewOnly ? "warning" : ""}`}>{selectedModelIsPreviewOnly ? "Import this ONNX voice from your local model folder to enable live conversion." : selectedModel.pairingNote}</p>}
              </div>
              <div className="voice-actions">
                <button className="compact-voice-button voice-picker-button" aria-expanded={modelDrawerOpen} aria-controls="voice-library" onClick={() => setModelDrawerOpen(true)}>
                  <Icon name="library" size={15} />
                  <span>Change voice</span>
                </button>
                {selectedModelCanLoad && (
                  <button className={`secondary-button ${selectedModelLoaded && !selectedModelNeedsReload ? "danger-outline" : ""}`} onClick={selectedModelLoaded && !selectedModelNeedsReload ? requestUnloadSelectedModel : loadSelectedModel} disabled={running || modelLoadBusy || modelUnloadBusy} title={selectedModelNeedsReload ? "Reload this voice to apply the selected retrieval index" : selectedModelLoaded ? "Release this voice from memory" : "Load this voice into memory"}>
                    {modelLoadBusy ? "Loading model…" : modelUnloadBusy ? "Unloading model…" : selectedModelNeedsReload ? "Reload voice" : selectedModelLoaded ? "Unload voice" : "Load voice"}
                  </button>
                )}
              </div>
            </section>

            {modelLoadBusy && (
              <section className="operation-strip" role="status" aria-live="polite">
                  <span className="operation-spinner" aria-hidden="true" />
                  <span className="operation-copy">
                    <strong>{modelLoadStageLabel}</strong>
                    <small>{modelLoadStageDetail} · {modelLoadElapsedLabel}{modelLoadSlow ? " · CUDA setup is taking longer than usual; the window is still responsive." : ""}</small>
                  <span className="operation-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={modelLoadProgress} aria-label="Voice loading progress"><span style={{ width: `${modelLoadProgress}%` }} /></span>
                </span>
                <span className="operation-state">{modelLoadProgress}%</span>
              </section>
            )}

            <section className={`setup-checklist ${quickSetupOpen ? "open" : "collapsed"}`} aria-labelledby="checklist-title">
              <div className="checklist-heading">
                <div><span className="eyebrow">Quick setup</span><h2 id="checklist-title">Ready when you are</h2></div>
                <div className="checklist-heading-actions"><span className={`progress-badge ${completedSetupSteps === 3 ? "complete" : ""}`}>{running ? "Live" : completedSetupSteps === 3 ? "Ready" : `${completedSetupSteps}/3 ready`}</span><button type="button" className="details-toggle" aria-expanded={quickSetupOpen} onClick={() => setQuickSetupOpen((value) => !value)}>{quickSetupOpen ? "Hide" : "Show"}</button></div>
              </div>
              {quickSetupOpen && <div className="checklist-grid">
                <button type="button" className={`checklist-step ${audioReady ? "complete" : ""}`} onClick={() => handleChecklistStep("audio")}>
                  <span className="step-number">{audioReady ? "✓" : "1"}</span>
                  <span><strong>Audio routing</strong><small>{audioReady ? "Microphone and output selected" : "Choose input and output"}</small></span>
                </button>
                <button type="button" className={`checklist-step ${conversionReady ? "complete" : ""}`} onClick={() => handleChecklistStep("voice")} disabled={modelLibraryLocked || conversionReady}>
                  <span className="step-number">{conversionReady ? "✓" : "2"}</span>
                  <span><strong>Voice conversion</strong><small>{conversionReady ? "Voice resident in the local engine" : selectedModelNeedsReload ? "Reload voice to apply the selected index" : selectedModelIsPreviewOnly ? "Import an ONNX or .pth voice to convert" : selectedModel.sourcePath ? selectedModelHasIndex ? "Load voice + retrieval index" : "Load voice (retrieval index optional)" : "Import a local .pth or .onnx voice"}</small></span>
                </button>
                <button type="button" className={`checklist-step ${running ? "complete" : ""}`} onClick={() => handleChecklistStep("session")}>
                  <span className="step-number">{running ? "✓" : "3"}</span>
                  <span><strong>Start session</strong><small>{running ? engineLabel : startBlockedReason ?? (conversionReady ? "Start local conversion" : selectedModelIsPreviewOnly ? "Start passthrough preview" : "Starts in passthrough")}</small></span>
                </button>
              </div>}
            </section>

            <section className="signal-card">
              <div className="signal-header">
                <div><h2>Signal monitor</h2><p>Microphone input and routed output</p></div>
                <div className={`engine-state ${engineTone}`}><span />{engineLabel}</div>
              </div>
              <div className="waveform-stage">
                <Waveform active={hasInputSignal} />
              </div>
              <div className="signal-metrics">
                <div className={`peak-metric ${inputPeakTone}`}><span>Input peak</span><strong>{peakDb(engineStatus.inputPeak)} dB</strong></div>
                <div className="signal-route"><span>Input</span><i /><span>{monitorDevice ? "Output + Monitor" : "Output"}</span></div>
                <div className={`peak-metric ${outputPeakTone}`}><span>Output peak</span><strong>{peakDb(engineStatus.outputPeak)} dB</strong></div>
              </div>
            </section>

            <section className="controls-section">
              <div className="section-heading"><div><h2>Voice controls</h2><p>Shape the character of the converted voice</p></div><div className="section-heading-actions"><span>{selectedModelNeedsReload ? "Reload voice to apply the selected index" : selectedModelLoaded ? "Changes apply when the next audio session starts" : "Load an imported voice to enable conversion"}</span><button className="details-toggle" onClick={resetSelectedModelSettings} disabled={running}>Reset</button></div></div>
              <div className="parameter-grid">
                <label className="parameter-card">
                  <span className="parameter-title"><span><strong>Pitch</strong><small>Shift vocal range</small></span><output>{pitch > 0 ? "+" : ""}{pitch} st</output></span>
                  <input aria-label="Pitch shift" type="range" min="-50" max="50" value={pitch} disabled={running} style={{ "--range-progress": `${((pitch + 50) / 100) * 100}%` } as CSSProperties} onChange={(event) => updateSelectedSettings({ pitchShift: Number(event.target.value) })} />
                  <span className="range-labels"><small>Lower</small><small>Original</small><small>Higher</small></span>
                </label>
                <label className="parameter-card">
                  <span className="parameter-title"><span><strong title="RVC FAISS retrieval strength">Index retrieval</strong><small>{selectedModelHasIndex ? "Match target identity" : "No sibling .index found"}</small></span><output>{selectedModelHasIndex ? `${indexRate}%` : "Off"}</output></span>
                  <input aria-label="Index retrieval" type="range" min="0" max="100" value={selectedModelHasIndex ? indexRate : 0} disabled={running || !selectedModelHasIndex} style={{ "--range-progress": `${selectedModelHasIndex ? indexRate : 0}%` } as CSSProperties} onChange={(event) => updateSelectedSettings({ indexRatio: Number(event.target.value) / 100 })} />
                  <span className="range-labels"><small>Original</small><small>Target index</small></span>
                </label>
                <label className="parameter-card">
                  <span className="parameter-title"><span><strong title="Preserve unvoiced consonants from retrieval artifacts">Protect ratio</strong><small>Preserve consonants</small></span><output>{selectedModelHasIndex ? `${protection}%` : "Off"}</output></span>
                  <input aria-label="Protect ratio" type="range" min="0" max="50" value={protection} disabled={running || !selectedModelHasIndex} style={{ "--range-progress": `${(protection / 50) * 100}%` } as CSSProperties} onChange={(event) => updateSelectedSettings({ protectRatio: Number(event.target.value) / 100 })} />
                  <span className="range-labels"><small>More protection</small><small>Off</small></span>
                </label>
              </div>
            </section>

            <section className="processing-panel">
              <div><h2>Processing mode</h2><p>{modeDescriptions[mode]}</p></div>
              <div className="mode-switch" role="group" aria-label="Processing mode">
                {(Object.keys(modeLabels) as ConversionMode[]).map((key) => (
                  <button key={key} className={mode === key ? "active" : ""} aria-pressed={mode === key} disabled={running} onClick={() => updateSelectedSettings({ streamingPreset: key, chunkFrames: streamProfiles[key].hop * 48, extraFrames: streamProfiles[key].extra * 48 })}>{modeLabels[key]}</button>
                ))}
              </div>
              <button className="secondary-button" aria-expanded={advancedOpen} onClick={() => setAdvancedOpen((value) => !value)}>{advancedOpen ? "Hide advanced" : "Advanced settings"}</button>
              {advancedOpen && (
                <div className="advanced-panel">
                  <div><span>Retrieval strength</span><strong>{selectedModelHasIndex ? `${indexRate}% ${selectedModelLoaded && !selectedModelNeedsReload ? "loaded" : "on next load"}` : "No index available"}</strong></div>
                  <label className="advanced-select"><span>Index file</span><select aria-label="Index file" value={selectedSettings.indexPath ?? ""} disabled={running || !selectedModel.indexPaths?.length} onChange={(event) => updateSelectedSettings({ indexPath: event.target.value || null })}><option value="">{selectedModel.indexPaths?.length ? "Off" : "No index available"}</option>{selectedModel.indexPaths?.map((path) => <option key={path} value={path}>{windowsFileName(path)}</option>)}</select></label>
                  <div><span>Feature embedder</span><strong>{selectedSettings.contentvecPath ? windowsFileName(selectedSettings.contentvecPath) : "Auto-discover ContentVec"}</strong></div>
                  {selectedModelLoaded && <div><span>Active feature backend</span><strong>{liveRvcStatus.featureBackend === "fairseq-hubert" ? "Fairseq HuBERT (CUDA)" : liveRvcStatus.featureBackend === "contentvec-onnx" ? "ContentVec ONNX (CUDA)" : liveRvcStatus.featureBackend ?? "Pending"}</strong></div>}
                  <label className="advanced-select"><span>Target speaker</span><select aria-label="Target speaker" value={selectedSettings.speakerId} disabled={running || !selectedModelLoaded || (liveRvcStatus.speakerCount ?? 1) <= 1} onChange={(event) => updateSelectedSettings({ speakerId: Number(event.target.value) })}>{Array.from({ length: Math.max(1, liveRvcStatus.speakerCount ?? 1) }, (_, speakerId) => <option key={speakerId} value={speakerId}>Speaker {speakerId}</option>)}</select></label>
                  <label className="advanced-range"><span>RMVPE threshold <output>{selectedSettings.f0Threshold.toFixed(2)}</output></span><input aria-label="RMVPE threshold" type="range" min="1" max="99" value={Math.round(selectedSettings.f0Threshold * 100)} disabled={running} style={{ "--range-progress": `${((selectedSettings.f0Threshold - 0.01) / 0.98) * 100}%` } as CSSProperties} onChange={(event) => updateSelectedSettings({ f0Threshold: Number(event.target.value) / 100 })} /></label>
                  <label className="advanced-select"><span>Chunk / streaming hop</span><select aria-label="Chunk size" value={selectedSettings.chunkFrames} disabled={running} onChange={(event) => updateSelectedSettings({ chunkFrames: Number(event.target.value) })}>{CHUNK_OPTIONS.map((frames) => <option key={frames} value={frames}>{frameDurationLabel(frames)}</option>)}</select></label>
                  <label className="advanced-select"><span>Extra / context</span><select aria-label="Extra context" value={selectedSettings.extraFrames} disabled={running} onChange={(event) => updateSelectedSettings({ extraFrames: Number(event.target.value) })}>{EXTRA_OPTIONS.map((frames) => <option key={frames} value={frames}>{frameDurationLabel(frames)}</option>)}</select></label>
                   <div><span>Streaming hop</span><strong>{Number(streamProfile.hop.toFixed(1))} ms · {modeLabels[mode]}</strong></div>
                   <div><span>Extra context</span><strong>{Number(streamProfile.extra.toFixed(1))} ms selected</strong></div>
                   <div><span>Effective analysis</span><strong>{liveRvcStatus.state === "ready" ? `${liveRvcStatus.analysisMilliseconds.toFixed(1)} ms after RVC rounding` : "Computed when loaded"}</strong></div>
                   <div><span>SOLA overlap</span><strong>{streamProfile.overlap} ms + {streamProfile.search} ms search</strong></div>
                   <div className="calibration-action"><div><span>Hardware calibration</span><small>Measures each profile on the loaded voice before audio starts.</small></div><button type="button" className="details-toggle" onClick={calibrateSelectedModel} disabled={calibrationBusy || running || !selectedModelLoaded || selectedModelNeedsReload}>{calibrationBusy ? "Measuring…" : "Run calibration"}</button></div>
                   {calibrationResult && <div className="calibration-result" role="status"><strong>{calibrationResult.message}</strong>{calibrationResult.profiles.map((measurement) => <div key={measurement.preset}><span>{modeLabels[measurement.preset]} · P95 {measurement.processMs.toFixed(1)} ms / {measurement.deadlineMs.toFixed(1)} ms{measurement.maxProcessMs === undefined ? "" : ` · max ${measurement.maxProcessMs.toFixed(1)}`}</span><button type="button" className={measurement.preset === calibrationResult.recommendedPreset ? "recommended" : ""} onClick={() => applyCalibration(measurement.preset)}>{measurement.preset === calibrationResult.recommendedPreset ? "Recommended · Apply" : "Apply"}</button></div>)}</div>}
                   <p className="advanced-note">Smaller chunks respond faster but can sound less stable. Extra adds context; the engine enforces enough context for safe SOLA stitching.</p>
                </div>
              )}
            </section>
          </div>
        </main>

        <button
          type="button"
          className="sidebar-resize-handle session-resize-handle"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize session sidebar"
          aria-valuemin={sidebarWidthBounds("session").minimum}
          aria-valuemax={sidebarWidthBounds("session").maximum}
          aria-valuenow={Math.round(sessionWidth)}
          title="Drag to resize session sidebar · Double-click to reset"
          onPointerDown={(event) => beginSidebarResize("session", event)}
          onDoubleClick={() => resetSidebarWidth("session")}
          onKeyDown={(event) => handleSidebarKeyDown("session", event)}
        />
        <aside className="setup-panel">
          <div className="panel-heading setup-heading">
            <div><span className="eyebrow">Session</span><h2>Audio setup</h2><p className="panel-subtitle">Route your local audio safely</p></div>
            <div className="setup-heading-actions">
              <button className="details-toggle refresh-button" onClick={runRouteTest} disabled={running || routeTestBusy || !outputDeviceId}>{routeTestBusy ? "Testing…" : "Test routes"}</button>
              <button className="details-toggle refresh-button" onClick={refreshDevices} disabled={running || deviceRefreshBusy}>{deviceRefreshBusy ? "Checking…" : "Refresh"}</button>
            </div>
          </div>

          <section className="setup-section">
            <label className="device-field">
              <span className="field-label"><span className="device-glyph"><Icon name="microphone" /></span><span><strong>Microphone</strong><small>{deviceSummary(inputDevice)}</small></span></span>
              <select value={inputDeviceId} onChange={(event) => setInputDeviceId(event.target.value)} disabled={running}>
                {devices.inputs.map((device) => <option key={device.id} value={device.id}>{device.name}</option>)}
              </select>
            </label>
            <LevelMeter active={hasInputSignal} peak={engineStatus.inputPeak} />

            <label className="device-field">
              <span className="field-label"><span className="device-glyph"><Icon name="speaker" /></span><span><strong>Output</strong><small>{deviceSummary(outputDevice)}</small></span></span>
              <select value={outputDeviceId} onChange={(event) => { const next = event.target.value; setOutputDeviceId(next); setRouteTestResult(null); if (monitorDeviceId === next) setMonitorDeviceId(""); }} disabled={running}>
                {devices.outputs.map((device) => <option key={device.id} value={device.id}>{device.name}</option>)}
              </select>
            </label>
            <LevelMeter active={hasOutputSignal} output peak={engineStatus.outputPeak} />

            <label className="device-field">
              <span className="field-label"><span className="device-glyph"><Icon name="headset" /></span><span><strong>Monitor</strong><small>{monitorDevice ? `${deviceSummary(monitorDevice)} · headphones` : "Optional headphone monitor · off"}</small></span></span>
              <select aria-label="Monitor" value={monitorDeviceId} onChange={(event) => { setMonitorDeviceId(event.target.value); setRouteTestResult(null); }} disabled={running}>
                <option value="">Off</option>
                {devices.outputs.filter((device) => device.id !== outputDeviceId).map((device) => <option key={device.id} value={device.id}>{device.name}</option>)}
              </select>
            </label>
            <LevelMeter active={hasMonitorSignal} output peak={engineStatus.monitorPeak} />

            {routeTestResult && <div className={`info-callout ${routeTestResult.outputError ? "error" : routeTestResult.monitorError ? "warning" : "success"}`} role="status">
              <span>{routeTestResult.outputError ? "!" : routeTestResult.monitorError ? "!" : "✓"}</span>
              <p><strong>{routeTestResult.outputError ? "Output callback needs attention." : routeTestResult.monitorError ? "Output callback passed; monitor needs attention." : "Output callbacks passed."}</strong> {routeTestResult.outputError ?? routeTestResult.monitorError ?? `${routeTestResult.outputDeviceName}${routeTestResult.monitorDeviceName ? ` + ${routeTestResult.monitorDeviceName}` : ""} · ${routeTestResult.durationMs} ms test tone`} <small>Callback test only; a virtual-cable loopback or downstream app route is not measured here.</small></p>
            </div>}

            <div className="audio-processing-toggle">
              <div><strong>Audio processing</strong><small>{audioProcessingSummary}</small></div>
              <div><button className="details-toggle" onClick={resetAudioSettings} disabled={running}>Reset</button><button className="details-toggle" aria-expanded={audioProcessingOpen} onClick={() => setAudioProcessingOpen((value) => !value)}>{audioProcessingOpen ? "Hide" : "Show"}</button></div>
            </div>
            {audioProcessingOpen && <div className="audio-processing-controls">
              <label><span><strong>Input gain</strong><output>{audioSettings.inputGainDb > 0 ? "+" : ""}{audioSettings.inputGainDb} dB</output></span><input aria-label="Input gain" type="range" min="-24" max="24" value={audioSettings.inputGainDb} disabled={running} style={{ "--range-progress": `${((audioSettings.inputGainDb + 24) / 48) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, inputGainDb: Number(event.target.value) }))} /></label>
              <label><span><strong>Output gain</strong><output>{audioSettings.outputGainDb > 0 ? "+" : ""}{audioSettings.outputGainDb} dB</output></span><input aria-label="Output gain" type="range" min="-24" max="12" value={audioSettings.outputGainDb} disabled={running} style={{ "--range-progress": `${((audioSettings.outputGainDb + 24) / 36) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, outputGainDb: Number(event.target.value) }))} /></label>
               <label><span><strong>Monitor gain</strong><output>{audioSettings.monitorGainDb > 0 ? "+" : ""}{audioSettings.monitorGainDb} dB</output></span><input aria-label="Monitor gain" type="range" min="-24" max="12" value={audioSettings.monitorGainDb} disabled={running || !monitorDeviceId} style={{ "--range-progress": `${((audioSettings.monitorGainDb + 24) / 36) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, monitorGainDb: Number(event.target.value) }))} /></label>
               <label><span><strong>Noise suppression</strong><output>{audioSettings.noiseSuppressionStrength <= 0 ? "Off" : `${Math.round(audioSettings.noiseSuppressionStrength * 100)}%`}</output></span><input aria-label="Noise suppression" type="range" min="0" max="100" value={Math.round(audioSettings.noiseSuppressionStrength * 100)} disabled={running} style={{ "--range-progress": `${audioSettings.noiseSuppressionStrength * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, noiseSuppressionStrength: Number(event.target.value) / 100 }))} /></label>
               <label><span><strong>Echo control</strong><output>{audioSettings.echoControlStrength <= 0 ? "Off" : `${Math.round(audioSettings.echoControlStrength * 100)}%`}</output></span><input aria-label="Echo control" type="range" min="0" max="100" value={Math.round(audioSettings.echoControlStrength * 100)} disabled={running} style={{ "--range-progress": `${audioSettings.echoControlStrength * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, echoControlStrength: Number(event.target.value) / 100 }))} /></label>
               <label className="audio-processing-check"><span><strong>High-pass / DC filter</strong><output>{audioSettings.highPassEnabled ? "On" : "Off"}</output></span><input aria-label="High-pass and DC filter" type="checkbox" checked={audioSettings.highPassEnabled} disabled={running} onChange={(event) => setAudioSettings((current) => ({ ...current, highPassEnabled: event.target.checked }))} /></label>
               <label><span><strong>Noise gate</strong><output>{audioSettings.noiseGateDb <= -80 ? "Off" : `${audioSettings.noiseGateDb} dB`}</output></span><input aria-label="Noise gate" type="range" min="-80" max="-20" value={audioSettings.noiseGateDb} disabled={running} style={{ "--range-progress": `${((audioSettings.noiseGateDb + 80) / 60) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, noiseGateDb: Number(event.target.value) }))} /></label>
               <p className="audio-processing-note">Noise suppression is a conservative adaptive expander. Echo control uses the converted output as a reference; headphones provide the cleanest result.</p>
            </div>}

            {!startupBusy && !audioReady && <div className="info-callout error" role="alert"><span>!</span><p><strong>Audio route incomplete.</strong> Refresh devices, then choose both a microphone and an output.</p></div>}
            {sampleRateDifference && <div className="info-callout warning" role="status"><span>↔</span><p><strong>Device rates differ.</strong> VC Next will resample the route to its 48 kHz RVC path. A matched-rate route may use slightly less CPU.</p></div>}
            {(inputPeakTone === "clip" || outputPeakTone === "clip") && <div className="info-callout error" role="alert"><span>!</span><p><strong>Clipping detected.</strong> {clippingMessage}</p></div>}
            {inputRouteSilent && <div className="info-callout warning" role="alert"><span>!</span><p><strong>No input signal detected.</strong> The selected device is open but delivering silence. Check the microphone level or the VoiceMeeter/CABLE bus feeding it, then restart audio.</p><button type="button" className="details-toggle" onClick={recoverAudioSession} disabled={engineBusy}>Restart audio</button></div>}
            {outputRouteStalled && <div className="info-callout error" role="alert"><span>!</span><p><strong>Input is active but output is idle.</strong> Check the selected output or virtual-cable input, then restart audio.</p><button type="button" className="details-toggle" onClick={recoverAudioSession} disabled={engineBusy}>Restart audio</button></div>}
            {selectedModelCanLoad && !inferenceRuntime.readyForRvc && !startupBusy && <div className="info-callout warning"><span>!</span><p><strong>RVC runtime needs attention.</strong> {inferenceRuntime.blockers.slice(0, 3).join(" · ") || "Check Engine details before loading this voice."}</p><div className="callout-actions"><button type="button" className="details-toggle" onClick={() => void launchRuntimeSetup()} disabled={runtimeRefreshBusy || running}>{runtimeRefreshBusy ? "Opening…" : "Run setup"}</button><button type="button" className="details-toggle" onClick={() => void copyRuntimeSetupCommand()}>Copy command</button></div></div>}
            {onnxCpuFallback && <div className="info-callout warning" role="status"><span>!</span><p><strong>ONNX generator is using CPU.</strong> The model is compatible, but this provider is not a low-latency guarantee. Install a working ONNX Runtime CUDA stack or use the PyTorch checkpoint when available.</p></div>}
            {workerRecovering && <div className="info-callout warning" role="status"><span>↻</span><p><strong>Voice worker is recovering.</strong> Audio stays live with silence while the model process restarts and warms again.</p></div>}
             {engineError && <div className="info-callout error" role="alert"><span>!</span><p>{engineError}</p>{running && <button type="button" className="details-toggle" onClick={recoverAudioSession} disabled={engineBusy}>Restart audio</button>}</div>}
            {!engineError && <div className={`setup-state ${engineStatus.state === "rvc" ? "live" : ""}`} role="status"><span />{engineStatus.state === "rvc" ? "Live conversion active" : selectedModelNeedsReload ? "Voice loaded · reload required" : selectedModelLoaded ? "Voice loaded · ready to start" : selectedModelIsPreviewOnly ? "Preview · passthrough ready" : "Passthrough ready"}</div>}
          </section>

          <section className="setup-section engine-summary">
            <div className="section-heading compact">
              <div><h3>Engine</h3><p>{running ? engineLabel : selectedModelNeedsReload ? "Reload required" : selectedModelLoaded ? "Voice ready" : "Waiting for a model"}</p></div>
              <div className="section-heading-meta"><span className="status-badge">{running ? engineLabel : selectedModelNeedsReload ? "Reload" : selectedModelLoaded ? "Ready" : "Preview"}</span><button className="details-toggle" onClick={refreshRuntime} disabled={runtimeRefreshBusy || running || startupBusy}>{runtimeRefreshBusy ? "Checking…" : "Recheck runtime"}</button><button className="details-toggle" aria-expanded={diagnosticsOpen} onClick={() => setDiagnosticsOpen((value) => !value)}>{diagnosticsOpen ? "Hide details" : "Details"}</button></div>
            </div>
            {!diagnosticsOpen && <div className="engine-glance">
              <div><span>Worker</span><strong>{liveRvcStatus.workerState === "recovering" ? "Recovering" : liveRvcStatus.workerState === "failed" ? "Needs attention" : liveRvcStatus.state === "ready" ? "Resident" : "Not loaded"}</strong></div>
              <div><span>Retrieval</span><strong>{liveRvcStatus.indexLoaded ? "Loaded" : selectedModelHasIndex ? "Available" : "Off"}</strong></div>
              <div><span>Runtime</span><strong>{inferenceRuntime.readyForRvc ? "Ready" : "Needs attention"}</strong></div>
            </div>}
            {diagnosticsOpen && <dl className="details-list">
              <div><dt>Audio backend</dt><dd>{devices.backend}</dd></div>
              <div><dt>Device rates</dt><dd>{inputDevice?.sampleRate?.toLocaleString() ?? "—"} / {outputDevice?.sampleRate?.toLocaleString() ?? "—"}{monitorDevice ? ` / ${monitorDevice.sampleRate.toLocaleString()}` : ""} Hz</dd></div>
              <div><dt>RVC path</dt><dd>{engineStatus.inferenceSampleRate.toLocaleString()} Hz internal</dd></div>
              <div><dt>Current stage</dt><dd>{running ? engineStatus.inferenceBackend : selectedModelLoaded ? "RVC warmed" : "RVC pending"}</dd></div>
              <div><dt>Generator backend</dt><dd>{liveRvcStatus.state === "ready" ? `${liveRvcStatus.backend === "onnx" ? "ONNX" : "PyTorch"} · ${liveRvcStatus.device ?? "unknown device"}` : "Not loaded"}</dd></div>
              <div><dt>Generator providers</dt><dd>{liveRvcStatus.generatorProviders?.join(" / ") || (liveRvcStatus.state === "ready" ? "PyTorch CUDA" : "Not loaded")}</dd></div>
              <div><dt>Feature backend</dt><dd>{liveRvcStatus.state === "ready" ? (liveRvcStatus.featureBackend === "fairseq-hubert" ? "Fairseq HuBERT · CUDA" : liveRvcStatus.featureBackend === "contentvec-onnx" ? "ContentVec · ONNX CUDA" : liveRvcStatus.featureBackend ?? "Unknown") : "Not loaded"}</dd></div>
              <div><dt>Model worker</dt><dd>{liveRvcStatus.workerState === "recovering" ? "Recovering" : liveRvcStatus.workerState === "failed" ? "Recovery failed" : liveRvcStatus.state === "ready" ? "Resident" : "Not loaded"}</dd></div>
              <div><dt>Worker restarts</dt><dd>{liveRvcStatus.workerRestarts ?? 0}</dd></div>
              <div><dt>Retrieval index</dt><dd>{liveRvcStatus.indexLoaded ? `${liveRvcStatus.indexVectorCount?.toLocaleString()} vectors` : "Not loaded"}</dd></div>
              <div><dt>Retrieval neighbors</dt><dd>{liveRvcStatus.indexLoaded ? `Nearest (${liveRvcStatus.indexNeighbors ?? 1})` : "Not loaded"}</dd></div>
              <div><dt>Idle silence gate</dt><dd>{liveRvcStatus.state === "ready" ? `${(liveRvcStatus.silenceSuppressedCalls ?? 0).toLocaleString()} hops suppressed` : "Not loaded"}</dd></div>
              <div><dt>Native idle backstop</dt><dd>{engineStatus.inferenceSilenceSuppressedCalls.toLocaleString()} model chunks suppressed</dd></div>
              <div><dt>Input floor / max</dt><dd>{liveRvcStatus.state === "ready" && liveRvcStatus.lastInputRms !== undefined ? `${liveRvcStatus.lastInputRms.toExponential(2)} / ${(liveRvcStatus.maxInputRms ?? 0).toExponential(2)} RMS` : "Waiting"}</dd></div>
              <div><dt>RVC loudness match</dt><dd>{liveRvcStatus.state === "ready" && liveRvcStatus.lastOutputGain !== undefined ? `${liveRvcStatus.lastOutputGain.toFixed(3)}× output gain` : "Waiting"}</dd></div>
              <div><dt>Python sidecar</dt><dd>{inferenceRuntime.source === "python-sidecar" ? "Connected" : "Desktop only"}</dd></div>
              <div><dt>Python runtime</dt><dd>{inferenceRuntime.python.version}{inferenceRuntime.python.rvcEnvironmentCompatible ? "" : " · needs 3.11"}</dd></div>
              <div><dt>RVC packages</dt><dd>{inferenceRuntime.readyForRvc ? "Ready" : `${inferenceRuntime.blockers.length} blockers`}</dd></div>
              <div><dt>ONNX Runtime</dt><dd>{inferenceRuntime.onnxRuntime.cudaProviderAvailable ? "CUDA provider" : inferenceRuntime.onnxRuntime.imported ? "CPU only" : "Unavailable"}</dd></div>
              <div><dt>ONNX providers</dt><dd>{inferenceRuntime.onnxRuntime.availableProviders.join(" / ") || "Not detected"}</dd></div>
              <div><dt>Pitch target</dt><dd>RMVPE</dd></div>
              <div><dt>F0 threshold</dt><dd>{selectedSettings.f0Threshold.toFixed(2)}</dd></div>
              <div><dt>Gain staging</dt><dd>{audioSettings.inputGainDb} / {audioSettings.outputGainDb} / {audioSettings.monitorGainDb} dB</dd></div>
              <div><dt>Monitor route</dt><dd>{engineStatus.monitorDeviceName ?? "Off"}</dd></div>
              <div><dt>Monitor buffer</dt><dd>{engineStatus.monitorDeviceId ? `${engineStatus.monitorBufferedFrames} fr · ${engineStatus.monitorUnderruns + engineStatus.monitorOverruns} XRuns` : "Off"}</dd></div>
              <div><dt>Output safety depth</dt><dd>{engineStatus.primeTargetFrames} fr · {engineStatus.reprimes} reprimes</dd></div>
              <div><dt>Clock corrections</dt><dd>{engineStatus.driftDroppedFrames} drop · {engineStatus.driftRepeatedFrames} repeat</dd></div>
              <div><dt>Monitor corrections</dt><dd>{engineStatus.monitorDeviceId ? `${engineStatus.monitorDriftDroppedFrames} drop · ${engineStatus.monitorDriftRepeatedFrames} repeat` : "Off"}</dd></div>
               <div><dt>Noise gate</dt><dd>{audioSettings.noiseGateDb <= -80 ? "Off" : `${audioSettings.noiseGateDb} dB`}</dd></div>
               <div><dt>Noise suppression</dt><dd>{audioSettings.noiseSuppressionStrength <= 0 ? "Off" : `${Math.round(audioSettings.noiseSuppressionStrength * 100)}%`}</dd></div>
               <div><dt>Echo control</dt><dd>{audioSettings.echoControlStrength <= 0 ? "Off" : `${Math.round(audioSettings.echoControlStrength * 100)}%`}</dd></div>
               <div><dt>High-pass / DC filter</dt><dd>{audioSettings.highPassEnabled ? "On" : "Off"}</dd></div>
              <div><dt>GPU memory</dt><dd>{Math.round(profile.vramMb / 1024)} GB</dd></div>
              <div><dt>Native chunk</dt><dd>{engineStatus.inferenceChunkFrames} frames</dd></div>
              <div><dt>Streaming hop</dt><dd>{liveRvcStatus.chunkFrames.toLocaleString()} frames</dd></div>
              <div><dt>Analysis window</dt><dd>{liveRvcStatus.analysisFrames.toLocaleString()} frames</dd></div>
              <div><dt>SOLA overlap</dt><dd>{liveRvcStatus.crossfadeFrames.toLocaleString()} frames</dd></div>
              <div><dt>Model time</dt><dd>{liveRvcStatus.processCalls ? `${liveRvcStatus.lastProcessMs.toFixed(1)} ms` : "Waiting"}</dd></div>
              <div><dt>Generator time</dt><dd>{liveRvcStatus.processCalls ? `${(liveRvcStatus.lastGeneratorMs ?? 0).toFixed(1)} ms` : "Waiting"}</dd></div>
              <div><dt>Retrieval time</dt><dd>{liveRvcStatus.processCalls ? `${(liveRvcStatus.lastRetrievalMs ?? 0).toFixed(1)} ms` : "Waiting"}</dd></div>
              <div><dt>SOLA offset</dt><dd>{liveRvcStatus.processCalls ? `${liveRvcStatus.lastSolaOffsetFrames ?? 0} frames` : "Waiting"}</dd></div>
            </dl>}
          </section>

          <section className="setup-section sidebar-tool-section telemetry-launcher">
            <div className="sidebar-tool-heading">
              <div><span className="eyebrow">Diagnostics</span><h3>Performance &amp; live log</h3><p>{running ? `${performanceLabel} · ${processLatencyMs === null ? "waiting for samples" : `${processLatencyMs.toFixed(1)} ms process`}` : liveLogs[0]?.message ?? "Open the session monitor"}</p></div>
              <button className="details-toggle" aria-haspopup="dialog" onClick={() => setTelemetryModalOpen(true)}>Open</button>
            </div>
          </section>

        </aside>
      </div>

      <main className={`about-view ${activeTab === "about" ? "active" : ""}`} aria-labelledby="about-title" hidden={activeTab !== "about"}>
        <div className="about-content">
          <section className="about-hero">
            <div className="about-brand-mark"><img src="/vc-next-icon.png" alt="" /></div>
            <div>
              <span className="eyebrow">About VC Next</span>
              <h1 id="about-title">Local voice studio</h1>
              <p>Shape, route, and convert voices locally with a focused workspace built for live audio.</p>
              <span className="about-privacy"><span />All processing stays on this computer</span>
            </div>
          </section>

          <div className="about-grid">
            <section className="about-card">
              <span className="eyebrow">Designed for live work</span>
              <h2>Keep the signal in view.</h2>
              <p>VC Next keeps model selection, audio routing, voice controls, and engine health close together so you can make changes without losing the session context.</p>
            </section>
            <section className="about-card">
              <span className="eyebrow">Current workspace</span>
              <h2>Local and ready.</h2>
              <dl className="about-details">
                <div><dt>Voice library</dt><dd>{availableModels.length} models</dd></div>
                <div><dt>Runtime</dt><dd>{inferenceRuntime.readyForRvc ? "RVC ready" : "Preview mode"}</dd></div>
                <div><dt>Audio backend</dt><dd>{devices.backend}</dd></div>
              </dl>
            </section>
          </div>

          <section className="about-card about-specs-card">
            <div className="about-specs-heading"><div><span className="eyebrow">System specs</span><h2>Local machine details.</h2></div><span className="about-specs-source">{inferenceRuntime.source === "python-sidecar" ? "Local runtime probe" : "Runtime fallback"}</span></div>
            <dl className="about-specs">
              <div><dt>CPU</dt><dd>{cpuSummary}</dd></div>
              <div><dt>RAM</dt><dd>{memorySummary}</dd></div>
              <div><dt>GPU</dt><dd title={profile.gpu}>{profile.gpu}</dd></div>
              <div><dt>VRAM</dt><dd>{Math.round(profile.vramMb / 1024)} GB</dd></div>
              <div><dt>CUDA</dt><dd>{cudaSummary}</dd></div>
              <div><dt>Python</dt><dd>{pythonSummary}</dd></div>
              <div><dt>GPU driver</dt><dd>{profile.driverVersion || "Unavailable"}</dd></div>
              <div><dt>Platform</dt><dd>{inferenceRuntime.platform || profile.os}</dd></div>
            </dl>
          </section>

          <section className="about-card about-footer-card">
            <div><span className="eyebrow">Built around your workflow</span><h2>Make the next session feel lighter.</h2></div>
            <p>Use Live for conversion and monitoring. Use About whenever you need a quick view of what VC Next is doing and where your local setup stands.</p>
          </section>

          <section className="about-card about-links-card">
            <div>
              <span className="eyebrow">Project links</span>
              <h2>Built in the open.</h2>
              <p>Follow development, inspect the source, or get to know the person behind VC Next.</p>
            </div>
            <div className="about-link-list">
              <a href="https://github.com/itzcaesar" target="_blank" rel="noreferrer"><Icon name="github" size={16} /><span><strong>GitHub profile</strong><small>github.com/itzcaesar</small></span><b>↗</b></a>
              <a href="https://github.com/itzcaesar/VC-Next" target="_blank" rel="noreferrer"><Icon name="library" size={16} /><span><strong>VC Next repository</strong><small>github.com/itzcaesar/VC-Next</small></span><b>↗</b></a>
              <div className="about-author"><span>Made by</span><strong>Muhammad Caesar Rifqi</strong></div>
            </div>
          </section>
        </div>
      </main>

      {telemetryModalOpen && <div className="model-modal-backdrop telemetry-modal-backdrop" role="presentation" onMouseDown={() => setTelemetryModalOpen(false)}>
        <section className="model-modal telemetry-modal" role="dialog" aria-modal="true" aria-labelledby="telemetry-modal-title" onMouseDown={(event) => event.stopPropagation()}>
          <div className="model-modal-heading">
            <div><span className="eyebrow">Session diagnostics</span><h2 id="telemetry-modal-title">Performance monitor</h2><p>Inspect live conversion timing, audio buffering, and recent session events without crowding the audio setup sidebar.</p></div>
            <div className="telemetry-modal-actions"><button type="button" className="details-toggle" onClick={exportDiagnostics}>Copy report</button><button type="button" className="icon-button" aria-label="Close performance monitor" onClick={() => setTelemetryModalOpen(false)}><Icon name="close" size={16} /></button></div>
          </div>

          <div className="telemetry-modal-body">
            <section className="telemetry-card" aria-labelledby="telemetry-performance-title">
              <div className="telemetry-card-heading">
                <div><span className="eyebrow">Performance</span><h3 id="telemetry-performance-title">Live metrics</h3><p>{running ? `${performanceLabel} · ${processLatencyMs === null ? "waiting for samples" : `${processLatencyMs.toFixed(1)} ms process`}` : "Start audio to collect live metrics"}</p></div>
                <span className={`telemetry-state ${performanceTone}`}>{performanceLabel}</span>
              </div>
              <div className="performance-panel" aria-live="polite">
                <div className="performance-grid">
                  <div className="performance-stat"><span>Process</span><strong>{processLatencyMs === null ? "—" : `${processLatencyMs.toFixed(1)} ms`}</strong><small>last voice pass</small></div>
                  <div className="performance-stat"><span>Peak</span><strong>{maxInferenceMs === null ? "—" : `${maxInferenceMs.toFixed(1)} ms`}</strong><small>max inference</small></div>
                  <div className="performance-stat"><span>Buffer</span><strong>{bufferedLatencyMs === null ? "—" : `${bufferedLatencyMs.toFixed(1)} ms`}</strong><small>audio queue</small></div>
                  <div className={`performance-stat ${performanceTone}`}><span>XRuns</span><strong>{xrunCount}</strong><small>underruns + overruns</small></div>
                </div>
                <div className="performance-meta"><span>{liveRvcStatus.processCalls.toLocaleString()} voice calls</span><span>{engineStatus.missedInferenceDeadlines} missed deadlines</span><span>{engineStatus.inferenceSilenceSuppressedCalls} native silence blocks</span></div>
              </div>
            </section>

            <section className="telemetry-card telemetry-log-card" aria-labelledby="telemetry-log-title">
              <div className="telemetry-card-heading">
                <div><span className="eyebrow">Live log</span><h3 id="telemetry-log-title">Session events</h3><p>{liveLogs.length > 0 ? `${liveLogs.length} recent event${liveLogs.length === 1 ? "" : "s"}` : "No events recorded in this session"}</p></div>
                <button className="details-toggle" onClick={() => setLiveLogs([])} disabled={liveLogs.length === 0}>Clear</button>
              </div>
              {liveLogs.length > 0 ? <div className="live-log" role="log" aria-live="polite">{liveLogs.map((entry) => <div className={`live-log-entry ${entry.tone}`} key={entry.id}><time>{entry.time}</time><span /><p>{entry.message}</p></div>)}</div> : <div className="live-log-empty">Start or configure a session to see events here.</div>}
            </section>
          </div>
        </section>
      </div>}

      {modelPackageOpen && <div className="model-modal-backdrop" role="presentation" onMouseDown={closeModelPackage}>
        <section className="model-modal model-package-modal" role="dialog" aria-modal="true" aria-labelledby="model-package-title" onMouseDown={(event) => event.stopPropagation()}>
          <div className="model-modal-heading">
            <div><span className="eyebrow">Voice library</span><h2 id="model-package-title">Add model package</h2><p>Assemble the local files once. A <strong>.pth</strong> checkpoint is required; the index and embedder are optional.</p></div>
            <button type="button" className="icon-button" aria-label="Close model package dialog" onClick={closeModelPackage} disabled={Boolean(modelPackageBusy)}><Icon name="close" size={16} /></button>
          </div>

          <div className="model-package-grid">
            <section className={`model-package-card required ${modelPackage.inspection ? "complete" : ""}`} aria-labelledby="checkpoint-list-title">
              <div className="model-package-card-heading"><div><span className="package-step">1</span><div><h3 id="checkpoint-list-title">Checkpoint</h3><p>Required · .pth</p></div></div><span className="package-required">Required</span></div>
              <div className="package-file-list">
                {modelPackage.checkpointPath ? <button type="button" className="package-file-row selected" onClick={selectModelCheckpoint} disabled={Boolean(modelPackageBusy)}><span className="package-file-icon">PTH</span><span><strong>{windowsFileName(modelPackage.checkpointPath)}</strong><small>{modelPackage.inspection?.container ?? "Inspected locally"}</small></span><span className="package-file-state">Selected</span></button> : <div className="package-empty"><strong>No checkpoint selected</strong><small>Choose the voice weights to begin.</small></div>}
              </div>
              {modelFolderCandidates.length > 0 && <div className="package-folder-candidates" role="listbox" aria-label="Voice models found in selected folder">
                <strong>{modelFolderCandidates.length} models found</strong>
                {modelFolderCandidates.map((path) => <button type="button" className="package-folder-candidate" key={path} onClick={() => void chooseModelFolderCandidate(path)} disabled={Boolean(modelPackageBusy)}><span>{windowsFileName(path)}</span><small>{path}</small></button>)}
              </div>}
              <div className="package-card-actions package-checkpoint-actions"><button type="button" className="package-select-button" onClick={selectModelCheckpoint} disabled={Boolean(modelPackageBusy)}>{modelPackageBusy === "checkpoint" ? "Inspecting…" : modelPackage.checkpointPath ? "Replace checkpoint" : "Choose .pth checkpoint"}</button><button type="button" className="package-secondary-button" onClick={selectModelFolder} disabled={Boolean(modelPackageBusy)}>{modelPackageBusy === "folder" ? "Scanning…" : "Choose folder"}</button></div>
            </section>

            <section className={`model-package-card ${modelPackage.indexPath ? "complete" : ""}`} aria-labelledby="index-list-title">
              <div className="model-package-card-heading"><div><span className="package-step">2</span><div><h3 id="index-list-title">Retrieval index</h3><p>Optional · .index</p></div></div><span className="package-optional">Optional</span></div>
              <div className="package-file-list">
                {modelPackage.indexCandidates.length > 0 ? modelPackage.indexCandidates.map((path) => <button type="button" className={`package-file-row ${sameWindowsPath(path, modelPackage.indexPath) ? "selected" : ""}`} key={path} onClick={() => setModelPackage((current) => ({ ...current, indexPath: path }))} disabled={Boolean(modelPackageBusy) || !modelPackage.inspection}><span className="package-file-icon">IDX</span><span><strong>{windowsFileName(path)}</strong><small>{sameWindowsPath(path, modelPackage.inspection?.recommendedIndex) ? "Recommended sibling" : "Detected index"}</small></span><span className="package-file-state">{sameWindowsPath(path, modelPackage.indexPath) ? "Selected" : "Use"}</span></button>) : <div className="package-empty"><strong>No index detected</strong><small>Conversion can run without retrieval.</small></div>}
              </div>
              <div className="package-card-actions"><button type="button" className="package-select-button" onClick={selectModelIndex} disabled={Boolean(modelPackageBusy) || !modelPackage.inspection}>{modelPackageBusy === "index" ? "Choosing…" : "Choose .index file"}</button>{modelPackage.indexPath && <button type="button" className="package-clear-button" onClick={() => setModelPackage((current) => ({ ...current, indexPath: null }))} disabled={Boolean(modelPackageBusy)}>Use none</button>}</div>
            </section>

            <section className={`model-package-card ${modelPackage.embedderPath ? "complete" : ""}`} aria-labelledby="embedder-list-title">
              <div className="model-package-card-heading"><div><span className="package-step">3</span><div><h3 id="embedder-list-title">Feature embedder</h3><p>Optional · ContentVec .onnx or Fairseq HuBERT .pt</p></div></div><span className="package-optional">Optional</span></div>
              <div className="package-file-list">
                {modelPackage.embedderPath ? <button type="button" className="package-file-row selected" onClick={selectModelEmbedder} disabled={Boolean(modelPackageBusy)}><span className="package-file-icon">FEAT</span><span><strong>{windowsFileName(modelPackage.embedderPath)}</strong><small>Explicit feature embedder path</small></span><span className="package-file-state">Selected</span></button> : <div className="package-empty"><strong>Auto-discover enabled</strong><small>Leave empty to use the runtime default.</small></div>}
              </div>
              <div className="package-card-actions"><button type="button" className="package-select-button" onClick={selectModelEmbedder} disabled={Boolean(modelPackageBusy) || !modelPackage.inspection}>{modelPackageBusy === "embedder" ? "Choosing…" : modelPackage.embedderPath ? "Replace embedder" : "Choose embedder"}</button>{modelPackage.embedderPath && <button type="button" className="package-clear-button" onClick={() => setModelPackage((current) => ({ ...current, embedderPath: null }))} disabled={Boolean(modelPackageBusy)}>Auto-discover</button>}</div>
            </section>
          </div>

          <section className={`model-package-cover ${modelPackage.coverImage ? "complete" : ""}`} aria-labelledby="cover-image-title">
            <div className="model-package-cover-heading">
              <div><span className="package-step">4</span><div><h3 id="cover-image-title">Cover image</h3><p>Optional · PNG, JPG, WEBP, BMP, or GIF</p></div></div>
              <span className="package-optional">Optional</span>
            </div>
            <div className="model-package-cover-content">
              <div className="model-package-cover-preview">
                <span>{modelInitials(modelPackageName || "VC")}</span>
                <CoverImage coverImage={modelPackage.coverImage} className="model-package-cover-image" />
              </div>
              <div className="model-package-cover-copy">
                <strong>{modelPackage.coverImage?.name ?? "No cover image selected"}</strong>
                <small>{modelPackage.coverImage ? "Animated GIFs stay animated in the library." : "Add artwork to make this voice easier to recognize."}</small>
                <div className="model-package-cover-actions">
                  <button type="button" className="package-select-button" onClick={() => coverInputRef.current?.click()} disabled={Boolean(modelPackageBusy) || !modelPackage.inspection}>{modelPackageBusy === "cover" ? "Reading image…" : modelPackage.coverImage ? "Replace image" : "Choose cover image"}</button>
                  {modelPackage.coverImage && <button type="button" className="package-clear-button" onClick={() => setModelPackage((current) => ({ ...current, coverImage: null }))} disabled={Boolean(modelPackageBusy)}>Remove</button>}
                </div>
              </div>
            </div>
            <input ref={coverInputRef} className="file-input-hidden" type="file" accept=".png,.jpg,.jpeg,.webp,.bmp,.gif,image/png,image/jpeg,image/webp,image/bmp,image/gif" onChange={handleModelCoverChange} />
          </section>

          <label className="model-package-name"><span>Display name</span><input value={modelPackageName} onChange={(event) => setModelPackageName(event.target.value)} placeholder="Name this voice" maxLength={80} disabled={!modelPackage.inspection || Boolean(modelPackageBusy)} /></label>
          <div className="model-package-review"><span className={`package-review-dot ${modelPackage.inspection ? "ready" : ""}`} /><div><strong>{modelPackage.inspection ? `${modelPackageName.trim() || "Unnamed voice"} is ready to add` : "Choose a checkpoint to review the package"}</strong><small>{modelPackage.inspection ? `${modelPackage.indexPath ? "Retrieval index selected" : "No retrieval index"} · ${modelPackage.embedderPath ? "Explicit embedder" : "Auto-discover embedder"} · ${modelPackage.inspection.modelDefaults && Object.keys(modelPackage.inspection.modelDefaults).length > 0 ? "w-okada settings imported" : "manual settings"}` : "The checkpoint is inspected locally before it enters your library."}</small></div></div>
          <div className="model-modal-actions"><button type="button" className="secondary-button" onClick={closeModelPackage} disabled={Boolean(modelPackageBusy)}>Cancel</button><button type="button" className="primary-button" onClick={addModelFromPackage} disabled={!modelPackage.inspection || !modelPackage.checkpointPath || !modelPackageName.trim() || Boolean(modelPackageBusy)}>{modelPackageBusy === "adding" ? "Adding…" : "Add model"}</button></div>
        </section>
      </div>}

      {modelEditorId && modelBeingEdited && <div className="model-modal-backdrop" role="presentation" onMouseDown={() => { if (!modelEditorCoverBusy) closeModelEditor(); }}>
        <section className="model-modal model-editor-modal" role="dialog" aria-modal="true" aria-labelledby="model-editor-title" onMouseDown={(event) => event.stopPropagation()}>
          <div className="model-modal-heading">
            <div><span className="eyebrow">Voice library</span><h2 id="model-editor-title">Edit voice</h2><p>Update the name and icon image shown in your library. The source files stay untouched.</p></div>
            <button type="button" className="icon-button" aria-label="Close edit voice dialog" onClick={closeModelEditor} disabled={modelEditorCoverBusy}><Icon name="close" size={16} /></button>
          </div>
          <label className="model-modal-field"><span>Display name</span><input autoFocus value={modelDraftName} onChange={(event) => setModelDraftName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") saveModelEdit(); }} maxLength={80} /></label>
          <section className="model-editor-cover" aria-labelledby="model-editor-cover-title">
            <div className="model-editor-cover-preview"><span>{modelInitials(modelDraftName || "VC")}</span><CoverImage coverImage={modelDraftCoverImage} className="model-editor-cover-image" /></div>
            <div className="model-editor-cover-copy">
              <strong id="model-editor-cover-title">{modelDraftCoverImage?.name ?? "No icon image selected"}</strong>
              <small>PNG, JPG, WEBP, BMP, or GIF. Animated GIFs stay animated in the library.</small>
              <div className="model-editor-cover-actions">
                <button type="button" className="package-select-button" onClick={() => editCoverInputRef.current?.click()} disabled={modelEditorCoverBusy}>{modelEditorCoverBusy ? "Reading image…" : modelDraftCoverImage ? "Replace image" : "Choose icon image"}</button>
                {modelDraftCoverImage && <button type="button" className="package-clear-button" onClick={() => setModelDraftCoverImage(null)} disabled={modelEditorCoverBusy}>Remove</button>}
              </div>
            </div>
            <input ref={editCoverInputRef} className="file-input-hidden" type="file" accept=".png,.jpg,.jpeg,.webp,.bmp,.gif,image/png,image/jpeg,image/webp,image/bmp,image/gif" onChange={handleEditCoverChange} />
          </section>
          <div className="model-modal-source"><span>Source file</span><strong title={modelBeingEdited.sourcePath}>{windowsFileName(modelBeingEdited.sourcePath)}</strong></div>
          <div className="model-modal-actions"><button type="button" className="secondary-button" onClick={closeModelEditor} disabled={modelEditorCoverBusy}>Cancel</button><button type="button" className="primary-button" onClick={saveModelEdit} disabled={modelEditorCoverBusy}>{modelEditorCoverBusy ? "Reading image…" : "Save changes"}</button></div>
        </section>
      </div>}

      {modelOrganizationId && modelBeingOrganized && <div className="model-modal-backdrop" role="presentation" onMouseDown={closeModelOrganization}>
        <section className="model-modal model-organization-modal" role="dialog" aria-modal="true" aria-labelledby="model-organization-title" onMouseDown={(event) => event.stopPropagation()}>
          <div className="model-modal-heading">
            <div><span className="eyebrow">Voice library</span><h2 id="model-organization-title">Organize voice</h2><p>Keep <strong>{modelBeingOrganized.name}</strong> easy to find without changing its source files.</p></div>
            <button type="button" className="icon-button" aria-label="Close organize voice dialog" onClick={closeModelOrganization}><Icon name="close" size={16} /></button>
          </div>
          <label className="model-modal-field"><span>Folder</span><input autoFocus value={modelOrganizationFolder} onChange={(event) => setModelOrganizationFolder(event.target.value)} placeholder="e.g. Stream voices" maxLength={40} list="model-folder-suggestions" onKeyDown={(event) => { if (event.key === "Enter") saveModelOrganization(); }} /></label>
          <datalist id="model-folder-suggestions">{modelFolders.map((folder) => <option key={folder} value={folder} />)}</datalist>
          <label className="model-modal-field"><span>Tags <small>(comma-separated)</small></span><input value={modelOrganizationTags} onChange={(event) => setModelOrganizationTags(event.target.value)} placeholder="e.g. warm, female, gaming" maxLength={220} onKeyDown={(event) => { if (event.key === "Enter") saveModelOrganization(); }} /></label>
          {modelTags.length > 0 && <div className="organization-suggestions"><span>Existing tags</span><div>{modelTags.slice(0, 8).map((tag) => <button type="button" key={tag} className={normalizeModelTags(modelOrganizationTags).some((current) => current.toLocaleLowerCase() === tag.toLocaleLowerCase()) ? "active" : ""} onClick={() => toggleOrganizationTag(tag)}>{tag}</button>)}</div></div>}
          <div className="model-organization-preview"><span className="eyebrow">Preview</span><div><span className="model-folder-chip">{normalizeModelFolder(modelOrganizationFolder) || "Unsorted"}</span>{normalizeModelTags(modelOrganizationTags).map((tag) => <span className="model-tag-chip" key={tag}>#{tag}</span>)}</div></div>
          <div className="model-modal-actions"><button type="button" className="secondary-button" onClick={closeModelOrganization}>Cancel</button><button type="button" className="primary-button" onClick={saveModelOrganization}>Save organization</button></div>
        </section>
      </div>}

      {modelRemovalId && modelBeingRemoved && <div className="model-modal-backdrop" role="presentation" onMouseDown={() => setModelRemovalId(null)}>
        <section className="model-modal removal-modal" role="dialog" aria-modal="true" aria-labelledby="model-removal-title" onMouseDown={(event) => event.stopPropagation()}>
          <div className="model-modal-heading">
            <div><span className="eyebrow">Voice library</span><h2 id="model-removal-title">Remove voice?</h2><p>This removes <strong>{modelBeingRemoved.name}</strong> from VC Next’s library only.</p></div>
            <button type="button" className="icon-button" aria-label="Close remove dialog" onClick={() => setModelRemovalId(null)}><Icon name="close" size={16} /></button>
          </div>
          <div className="model-modal-warning"><span>!</span><p>The source file and any matching index files will remain on disk. You can import this voice again later.</p></div>
          <div className="model-modal-actions"><button type="button" className="secondary-button" onClick={() => setModelRemovalId(null)}>Keep voice</button><button type="button" className="danger-button" onClick={removeModelFromLibrary}>Remove from library</button></div>
        </section>
      </div>}

      {modelUnloadConfirmOpen && <div className="model-modal-backdrop" role="presentation" onMouseDown={() => { if (!modelUnloadBusy) setModelUnloadConfirmOpen(false); }}>
        <section className="model-modal removal-modal" role="dialog" aria-modal="true" aria-labelledby="model-unload-title" onMouseDown={(event) => event.stopPropagation()}>
          <div className="model-modal-heading">
            <div><span className="eyebrow">Voice memory</span><h2 id="model-unload-title">Unload voice?</h2><p>Release <strong>{selectedModel.name}</strong> from memory while keeping it in your library.</p></div>
            <button type="button" className="icon-button" aria-label="Close unload dialog" onClick={() => setModelUnloadConfirmOpen(false)} disabled={modelUnloadBusy}><Icon name="close" size={16} /></button>
          </div>
          <div className="model-modal-warning"><span>!</span><p>The source checkpoint, index, embedder, cover image, and saved settings will remain untouched. You can load this voice again whenever you need it.</p></div>
          <div className="model-modal-actions"><button type="button" className="secondary-button" onClick={() => setModelUnloadConfirmOpen(false)} disabled={modelUnloadBusy}>Keep loaded</button><button type="button" className="danger-button" onClick={unloadSelectedModel} disabled={modelUnloadBusy}>{modelUnloadBusy ? "Unloading…" : "Unload voice"}</button></div>
        </section>
      </div>}

      <footer className="status-bar">
        <div className="status-primary"><span className={signalActive && engineStatus.state !== "preview" ? "active" : ""} />{sessionStatusLabel}</div>
        <div className="status-message" role="status" aria-live="polite">{engineError ?? (workerRecovering ? "Restarting the local voice worker and restoring the loaded model…" : modelLoadBusy ? `Loading and warming ${selectedModel.name} on CUDA…` : modelUnloadBusy ? `Releasing ${selectedModel.name} from memory…` : modelPackageBusy ? modelPackageBusy === "adding" ? "Adding the model to your local library…" : modelPackageBusy === "cover" ? "Reading cover image…" : modelPackageBusy === "folder" ? "Scanning the selected model folder…" : "Selecting and inspecting model package files…" : startupBusy ? "Discovering audio devices and local inference runtime…" : running ? `${engineLabel} · ${engineStatus.state === "rvc" ? "Local AI processing" : "Unconverted audio"}` : selectedModelLoaded ? "Voice resident on GPU · Ready for local conversion" : "All processing stays on this computer")}</div>
        <div className="status-items"><span>{engineStatus.sampleRate ? `${engineStatus.sampleRate / 1000} kHz` : "48 kHz target"}</span><span>Queue {engineStatus.bufferedFrames}/{engineStatus.primeTargetFrames} fr</span><span>XRuns {engineStatus.underruns + engineStatus.overruns + engineStatus.monitorUnderruns + engineStatus.monitorOverruns}</span></div>
      </footer>
    </div>
  );
}

export default App;

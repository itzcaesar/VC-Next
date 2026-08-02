import { useEffect, useMemo, useState, type CSSProperties } from "react";
import "./App.css";
import {
  FALLBACK_AUDIO_DEVICES,
  FALLBACK_INFERENCE_RUNTIME,
  FALLBACK_PROFILE,
  EMPTY_LIVE_RVC_STATUS,
  MODEL_PRESETS,
  STOPPED_ENGINE_STATUS,
  chooseAndInspectRvcPackage,
  getAudioDevices,
  getAudioEngineStatus,
  getLiveRvcStatus,
  getSystemProfile,
  probeInferenceRuntime,
  loadLiveRvcModel,
  setLiveRvcSettings,
  startAudioEngine,
  stopAudioEngine,
  type AudioDevice,
  type AudioDeviceSnapshot,
  type AudioEngineStatus,
  type AudioProcessingSettings,
  type ConversionMode,
  type InferenceRuntimeProbe,
  type LiveRvcStatus,
  type ModelPreset,
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

const streamProfiles: Record<ConversionMode, { hop: number; analysis: number; overlap: number; search: number }> = {
  quality: { hop: 250, analysis: 600, overlap: 50, search: 15 },
  balanced: { hop: 200, analysis: 500, overlap: 40, search: 12 },
  latency: { hop: 160, analysis: 400, overlap: 30, search: 10 },
};

const CHUNK_OPTIONS = [
  3_072, 3_840, 4_800, 5_760, 7_200, 7_680, 9_600, 12_000, 12_288,
  14_400, 16_800, 19_200, 21_600, 24_000, 28_800, 33_600, 38_400,
  43_200, 48_000, 49_152, 52_800,
];

const EXTRA_OPTIONS = [
  3_840, 7_680, 16_320, 24_000, 32_640, 65_280, 131_040, 144_000,
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

function LevelMeter({ active, output = false }: { active: boolean; output?: boolean }) {
  return (
    <div className={`level-meter ${active ? "active" : ""}`} aria-label={active ? "Signal active" : "No signal"}>
      {Array.from({ length: 20 }, (_, index) => (
        <i key={index} className={index > 16 ? "peak" : ""} style={{ animationDelay: `${-(index + (output ? 4 : 0)) * 36}ms` }} />
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

function sameWindowsPath(left: string | null | undefined, right: string | null | undefined) {
  return Boolean(left && right && left.toLocaleLowerCase() === right.toLocaleLowerCase());
}

const MODEL_SETTINGS_STORAGE_KEY = "vc-next:model-settings:v1";
const AUDIO_SETTINGS_STORAGE_KEY = "vc-next:audio-settings:v1";
const THEME_STORAGE_KEY = "vc-next:theme:v1";
const DEVICE_SETTINGS_STORAGE_KEY = "vc-next:device-settings:v1";

function defaultModelSettings(model: ModelPreset): RvcModelSettings {
  return {
    pitchShift: 0,
    indexRatio: model.indexPaths?.length ? 0.5 : 0,
    protectRatio: 0.33,
    speakerId: 0,
    indexPath: model.recommendedIndexPath ?? model.indexPaths?.[0] ?? null,
    contentvecPath: model.embedderPath ?? null,
    f0Threshold: 0.03,
    streamingPreset: "balanced",
    chunkFrames: 9_600,
    extraFrames: 24_000,
  };
}

const DEFAULT_AUDIO_SETTINGS: AudioProcessingSettings = {
  inputGainDb: 0,
  outputGainDb: 0,
  monitorGainDb: -6,
  noiseGateDb: -80,
};

function loadStoredAudioSettings(): AudioProcessingSettings {
  try {
    const value = window.localStorage.getItem(AUDIO_SETTINGS_STORAGE_KEY);
    if (!value) return DEFAULT_AUDIO_SETTINGS;
    return { ...DEFAULT_AUDIO_SETTINGS, ...JSON.parse(value) };
  } catch {
    return DEFAULT_AUDIO_SETTINGS;
  }
}

function loadStoredModelSettings(): Record<string, RvcModelSettings> {
  try {
    const value = window.localStorage.getItem(MODEL_SETTINGS_STORAGE_KEY);
    if (!value) return {};
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
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

function App() {
  const [profile, setProfile] = useState<SystemProfile>(FALLBACK_PROFILE);
  const [devices, setDevices] = useState<AudioDeviceSnapshot>(FALLBACK_AUDIO_DEVICES);
  const [engineStatus, setEngineStatus] = useState<AudioEngineStatus>(STOPPED_ENGINE_STATUS);
  const [inferenceRuntime, setInferenceRuntime] = useState<InferenceRuntimeProbe>(FALLBACK_INFERENCE_RUNTIME);
  const [liveRvcStatus, setLiveRvcStatus] = useState<LiveRvcStatus>(EMPTY_LIVE_RVC_STATUS);
  const [running, setRunning] = useState(false);
  const [engineBusy, setEngineBusy] = useState(false);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [inputDeviceId, setInputDeviceId] = useState(() => loadStoredDeviceSelection().inputDeviceId || FALLBACK_AUDIO_DEVICES.defaultInputId || "");
  const [outputDeviceId, setOutputDeviceId] = useState(() => loadStoredDeviceSelection().outputDeviceId || FALLBACK_AUDIO_DEVICES.defaultOutputId || "");
  const [monitorDeviceId, setMonitorDeviceId] = useState(() => loadStoredDeviceSelection().monitorDeviceId);
  const [modelId, setModelId] = useState(MODEL_PRESETS[0].id);
  const [modelSettings, setModelSettings] = useState<Record<string, RvcModelSettings>>(loadStoredModelSettings);
  const [audioSettings, setAudioSettings] = useState<AudioProcessingSettings>(loadStoredAudioSettings);
  const [importedModels, setImportedModels] = useState<ModelPreset[]>([]);
  const [modelImportBusy, setModelImportBusy] = useState(false);
  const [modelLoadBusy, setModelLoadBusy] = useState(false);
  const [deviceRefreshBusy, setDeviceRefreshBusy] = useState(false);
  const [startupBusy, setStartupBusy] = useState(true);
  const [modelQuery, setModelQuery] = useState("");
  const [modelDrawerOpen, setModelDrawerOpen] = useState(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(loadStoredTheme);

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
        })
        .catch((error: unknown) => setEngineError(String(error)));
    }, 500);
    return () => window.clearInterval(interval);
  }, [running, engineBusy, engineStatus.state]);

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
    if (!modelDrawerOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setModelDrawerOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [modelDrawerOpen]);

  const availableModels = useMemo(
    () => [...MODEL_PRESETS, ...importedModels],
    [importedModels],
  );
  const selectedModel = useMemo(
    () => availableModels.find((model) => model.id === modelId) ?? MODEL_PRESETS[0],
    [availableModels, modelId],
  );
  const modelDefaults = defaultModelSettings(selectedModel);
  const selectedSettings = {
    ...modelDefaults,
    ...modelSettings[selectedModel.id],
  };
  if (selectedSettings.indexPath && !selectedModel.indexPaths?.some((path) => sameWindowsPath(path, selectedSettings.indexPath))) {
    selectedSettings.indexPath = modelDefaults.indexPath;
  }
  const mode = selectedSettings.streamingPreset;
  const presetProfile = streamProfiles[mode];
  const streamProfile = {
    ...presetProfile,
    hop: selectedSettings.chunkFrames / 48,
    analysis: selectedSettings.extraFrames / 48,
  };
  const pitch = selectedSettings.pitchShift;
  const indexRate = Math.round(selectedSettings.indexRatio * 100);
  const protection = Math.round(selectedSettings.protectRatio * 100);
  const filteredModels = useMemo(() => {
    const query = modelQuery.trim().toLocaleLowerCase();
    if (!query) return availableModels;
    return availableModels.filter((model) => `${model.name} ${model.format}`.toLocaleLowerCase().includes(query));
  }, [availableModels, modelQuery]);
  const inputDevice = devices.inputs.find((device) => device.id === inputDeviceId);
  const outputDevice = devices.outputs.find((device) => device.id === outputDeviceId);
  const monitorDevice = devices.outputs.find((device) => device.id === monitorDeviceId);
  const selectedModelLoaded = liveRvcStatus.state === "ready"
    && sameWindowsPath(liveRvcStatus.modelPath, selectedModel.sourcePath);
  const selectedModelCanLoad = selectedModel.format === "RVC v2" && Boolean(selectedModel.sourcePath);
  const selectedModelIsPreviewOnly = selectedModel.format === "RVC ONNX";
  const voiceNeedsLoad = selectedModelCanLoad && !selectedModelLoaded;
  const selectedModelHasIndex = Boolean(selectedSettings.indexPath && selectedModel.indexPaths?.some((path) => sameWindowsPath(path, selectedSettings.indexPath)));
  const audioReady = Boolean(inputDevice && outputDevice);
  const sampleRateMismatch = Boolean(
    inputDevice && outputDevice && (
      inputDevice.sampleRate !== outputDevice.sampleRate
      || Boolean(monitorDevice && monitorDevice.sampleRate !== inputDevice.sampleRate)
    ),
  );
  const conversionReady = selectedModelLoaded;
  const signalActive = running;
  const hasInputSignal = signalActive && ["passthrough", "rvc"].includes(engineStatus.state) && engineStatus.inputPeak > 0.001;
  const hasOutputSignal = signalActive && ["passthrough", "rvc"].includes(engineStatus.state) && engineStatus.outputPeak > 0.001;
  const hasMonitorSignal = signalActive && Boolean(engineStatus.monitorDeviceId) && engineStatus.monitorPeak > 0.001;
  const workerRecovering = running && liveRvcStatus.workerState === "recovering";
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
  const hardwareLabel = startupBusy
    ? "Checking engine"
    : profile.source === "native-probe"
      ? inferenceRuntime.readyForRvc
        ? "RVC ready"
        : inferenceRuntime.torchRuntime.cudaAvailable ? "CUDA found" : "Runtime setup"
      : "Prototype baseline";
  const hardwareTone = startupBusy
    ? "checking"
    : profile.source === "native-probe"
      ? inferenceRuntime.readyForRvc ? "ready" : "warning"
      : "preview";
  const completedSetupSteps = [audioReady, conversionReady, running].filter(Boolean).length;
  const waveformMessage = !running
    ? "Start audio to monitor the signal"
    : devices.source === "browser-preview"
      ? "Browser preview · no live capture"
      : "Waiting for input…";
  const sessionStatusLabel = modelLoadBusy
    ? "Loading voice"
    : modelImportBusy
      ? "Inspecting"
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
      : sampleRateMismatch
        ? `Match the selected device sample rates (${inputDevice?.sampleRate.toLocaleString()} Hz input, ${outputDevice?.sampleRate.toLocaleString()} Hz output${monitorDevice ? `, ${monitorDevice.sampleRate.toLocaleString()} Hz monitor` : ""}).`
        : modelLoadBusy
          ? "Wait for the voice model to finish loading."
          : voiceNeedsLoad
            ? "Load the selected voice before starting conversion."
            : null;
  const startDisabled = running
    ? engineBusy
    : engineBusy || startupBusy || modelLoadBusy || !audioReady || sampleRateMismatch || voiceNeedsLoad;
  const startButtonLabel = running
    ? "Stop audio"
    : startupBusy
      ? "Preparing…"
      : modelLoadBusy
        ? "Loading voice…"
        : !audioReady
          ? "Choose audio"
          : sampleRateMismatch
            ? "Match sample rates"
              : voiceNeedsLoad
                ? "Load voice first"
                : "Start audio";
  const modelStatusFor = (model: ModelPreset) => {
    const loaded = liveRvcStatus.state === "ready" && sameWindowsPath(liveRvcStatus.modelPath, model.sourcePath);
    if (loaded) return { label: "Ready", tone: "ready" };
    if (!model.sourcePath) return { label: "Preview", tone: "preview" };
    if (model.format === "RVC ONNX") return { label: "ONNX preview", tone: "neutral" };
    if (!model.indexPaths?.length) return { label: "No index · optional", tone: "neutral" };
    return { label: "Needs load", tone: "pending" };
  };
  const selectedModelStatus = modelStatusFor(selectedModel);

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
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setDeviceRefreshBusy(false);
    }
  }

  function handleChecklistStep(step: "audio" | "voice" | "session") {
    if (step === "audio") {
      document.querySelector<HTMLElement>(".setup-panel")?.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    if (step === "voice") {
      if (selectedModelIsPreviewOnly) {
        setNotice("ONNX voices are preview-only here; import a .pth voice for live conversion");
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

  useEffect(() => {
    try {
      window.localStorage.setItem(MODEL_SETTINGS_STORAGE_KEY, JSON.stringify(modelSettings));
    } catch {
      // Settings persistence is best-effort; inference remains fully local either way.
    }
  }, [modelSettings]);

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

  async function toggleEngine() {
    if (engineBusy) return;
    setEngineBusy(true);
    setEngineError(null);
    try {
      if (!running && selectedModelLoaded) {
        setLiveRvcStatus(await setLiveRvcSettings(selectedSettings));
      }
      const status = running
        ? await stopAudioEngine()
        : await startAudioEngine(inputDeviceId, outputDeviceId, monitorDeviceId || null, audioSettings);
      setEngineStatus(status);
      setRunning(status.state !== "stopped");
      if (running) {
        getLiveRvcStatus().then(setLiveRvcStatus).catch(() => undefined);
      }
    } catch (error) {
      setRunning(false);
      setEngineStatus(STOPPED_ENGINE_STATUS);
      setEngineError(String(error));
    } finally {
      setEngineBusy(false);
    }
  }

  async function loadSelectedModel() {
    if (modelLoadBusy || running || !selectedModel.sourcePath || selectedModel.format !== "RVC v2") return;
    setModelLoadBusy(true);
    setEngineError(null);
    try {
      const status = await loadLiveRvcModel(selectedModel.sourcePath, selectedSettings);
      setLiveRvcStatus(status);
      setNotice(`${selectedModel.name} is loaded and ready`);
      if (status.targetSampleRate) {
        setImportedModels((current) => current.map((model) => (
          sameWindowsPath(model.sourcePath, status.modelPath)
            ? { ...model, sampleRate: status.targetSampleRate ?? model.sampleRate }
            : model
        )));
      }
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelLoadBusy(false);
    }
  }

  async function importModel() {
    if (modelImportBusy) return;
    setModelImportBusy(true);
    setEngineError(null);
    try {
      const packageSelection = await chooseAndInspectRvcPackage();
      if (!packageSelection) return;
      const { inspection } = packageSelection;
      const displayName = inspection.name.replace(/\.(pth|onnx)$/i, "");
      const initials = displayName.replace(/[^a-z0-9]/gi, "").slice(0, 2).toUpperCase() || "VC";
      const indexPaths = [packageSelection.indexPath, ...inspection.siblingIndexes]
        .filter((path): path is string => Boolean(path))
        .filter((path, index, paths) => paths.findIndex((candidate) => sameWindowsPath(candidate, path)) === index);
      const model: ModelPreset = {
        id: inspection.path,
        name: displayName,
        initials,
        format: inspection.extension === ".onnx" ? "RVC ONNX" : "RVC v2",
        sampleRate: null,
        sourcePath: inspection.path,
        indexPaths,
        recommendedIndexPath: packageSelection.indexPath ?? inspection.recommendedIndex ?? null,
        embedderPath: packageSelection.contentvecPath,
        pairingNote: packageSelection.indexPath
          ? "The explicitly selected .index file will be loaded with this checkpoint."
          : inspection.pairingNote,
      };
      setImportedModels((current) => [model, ...current.filter((item) => item.id !== model.id)]);
      setModelId(model.id);
      setModelDrawerOpen(false);
      setNotice(
        indexPaths.length
          ? `${displayName} added with its matching retrieval index`
          : `${displayName} added without a retrieval index`,
      );
    } catch (error) {
      setEngineError(String(error));
    } finally {
      setModelImportBusy(false);
    }
  }

  return (
    <div className="app-frame" data-theme={theme} aria-busy={modelLoadBusy || startupBusy}>
      <header className="app-header" data-tauri-drag-region>
        <div className="brand-block">
          <span className="brand-mark"><img className="brand-icon" src="/vc-next-icon.png" alt="" /></span>
          <span><strong>VC Next</strong><small>Local voice studio</small></span>
        </div>

        <nav className="primary-nav" aria-label="Main navigation">
          <button className="active" aria-current="page"><Icon name="activity" />Live</button>
        </nav>

        <div className="header-actions">
          <div className={`hardware-chip ${hardwareTone}`} title={inferenceRuntime.readyForRvc ? profile.gpu : inferenceRuntime.blockers[0] ?? profile.gpu}>
            <span />
            <span><strong>{hardwareLabel}</strong><small>{profile.gpu.replace("NVIDIA GeForce ", "")}</small></span>
          </div>
          <button className="compact-voice-button header-voice-button" aria-expanded={modelDrawerOpen} aria-controls="voice-library" onClick={() => setModelDrawerOpen(true)}>
            <Icon name="library" size={15} />
            <span>Voices</span>
          </button>
          <button className="icon-button" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} onClick={() => setTheme((value) => value === "dark" ? "light" : "dark")}>
            <Icon name={theme === "dark" ? "sun" : "moon"} />
          </button>
          <button className={`start-button ${running ? "stop" : ""}`} onClick={toggleEngine} aria-pressed={running} aria-keyshortcuts="Control+Enter" title={startBlockedReason ?? "Start or stop audio with Ctrl+Enter"} disabled={startDisabled}>
            <Icon name={running ? "stop" : "play"} />
            {engineBusy ? "Working…" : startButtonLabel}
          </button>
        </div>
      </header>

      {notice && <div className="toast" role="status" aria-live="polite">{notice}</div>}
      {modelDrawerOpen && <button className="model-drawer-shade" aria-label="Close voice library" onClick={() => setModelDrawerOpen(false)} />}

      <div className="workspace">
        <aside className={`model-panel ${modelDrawerOpen ? "open" : ""}`} id="voice-library">
          <div className="panel-heading">
            <div><span className="eyebrow">Voice models</span><h2>Your library</h2></div>
            <button className="icon-button subtle" aria-label="Add model" onClick={importModel} disabled={modelImportBusy}><Icon name="plus" /></button>
          </div>

          <label className="search-field">
            <Icon name="search" size={17} />
            <input placeholder="Search your models" aria-label="Search models" value={modelQuery} onChange={(event) => setModelQuery(event.target.value)} />
            {modelQuery && <button type="button" className="search-clear" aria-label="Clear model search" onMouseDown={(event) => event.preventDefault()} onClick={() => setModelQuery("")}><Icon name="close" size={14} /></button>}
          </label>

          <div className="model-list" aria-label="Local voice models" role="listbox">
            {filteredModels.length > 0 ? filteredModels.map((model, index) => (
              <button key={model.id} role="option" aria-selected={modelId === model.id} className={modelId === model.id ? "selected" : ""} onClick={() => { setModelId(model.id); setModelDrawerOpen(false); }} disabled={running || modelLoadBusy}>
                <span className={`model-art art-${(index % 4) + 1}`}>{model.initials}</span>
                <span className="model-copy"><strong>{model.name}</strong><small>{model.format} · {model.sampleRate ? `${model.sampleRate / 1000} kHz` : "Needs validation"}</small></span>
                <span className={`model-status ${modelStatusFor(model).tone}`}>{modelStatusFor(model).label}</span>
              </button>
            )) : (
              <div className="model-empty">
                <strong>No voices found</strong>
                <small>Try a different name or format.</small>
              </div>
            )}
          </div>

          <div className="library-spacer" />
          <button className="drop-model-button" onClick={importModel} disabled={modelImportBusy}>
            <span className="drop-icon"><Icon name="plus" /></span>
            <span><strong>{modelImportBusy ? "Selecting model package…" : "Import a voice model"}</strong><small>Model · optional .index · optional embedder</small></span>
          </button>
          <p className="library-note">Imported voices stay local to this desktop session.</p>
        </aside>

        <main className="studio-scroll">
          <div className="studio-content">
            <section className="voice-overview">
              <div className="voice-art-large">{selectedModel.initials}<span /></div>
              <div className="voice-title">
                <span className="eyebrow">{selectedModelLoaded ? "Active voice" : "Selected voice"}</span>
                <h1>{selectedModel.name}</h1>
                <div className="metadata-row">
                  <span>{selectedModel.format}</span>
                  <span>{selectedModel.sampleRate ? `${selectedModel.sampleRate / 1000} kHz` : "Unverified rate"}</span>
                  {selectedModel.sourcePath && <span>{selectedModel.indexPaths?.length ? "Retrieval index paired" : "Retrieval index off"}</span>}
                  <span className={`model-state-badge ${selectedModelStatus.tone}`}>{selectedModelLoaded ? "Loaded and warmed" : modelLoadBusy ? "Loading and warming" : selectedModelStatus.label}</span>
                </div>
                {(selectedModel.pairingNote || selectedModelIsPreviewOnly) && <p className={`voice-note ${selectedModelIsPreviewOnly ? "warning" : ""}`}>{selectedModelIsPreviewOnly ? "ONNX is preview-only here; import a .pth voice for live conversion." : selectedModel.pairingNote}</p>}
              </div>
              <div className="voice-actions">
                <button className="compact-voice-button voice-picker-button" aria-expanded={modelDrawerOpen} aria-controls="voice-library" onClick={() => setModelDrawerOpen(true)}>
                  <Icon name="library" size={15} />
                  <span>Change voice</span>
                </button>
                {selectedModelCanLoad && (
                  <button className="secondary-button" onClick={loadSelectedModel} disabled={running || modelLoadBusy || selectedModelLoaded}>
                    {modelLoadBusy ? "Loading model…" : selectedModelLoaded ? "Voice loaded" : "Load voice"}
                  </button>
                )}
              </div>
            </section>

            {modelLoadBusy && (
              <section className="operation-strip" role="status" aria-live="polite">
                <span className="operation-spinner" aria-hidden="true" />
                <span>
                  <strong>Loading {selectedModel.name} on the GPU</strong>
                  <small>Starting the local RVC worker, validating the checkpoint, and warming CUDA. This can take several seconds.</small>
                </span>
                <span className="operation-state">Working</span>
              </section>
            )}

            <section className="setup-checklist" aria-labelledby="checklist-title">
              <div className="checklist-heading">
                <div><span className="eyebrow">Quick setup</span><h2 id="checklist-title">Ready when you are</h2></div>
                <span className={`progress-badge ${running ? "complete" : ""}`}>{running ? "Live" : `${completedSetupSteps}/3 ready`}</span>
              </div>
              <div className="checklist-grid">
                <button type="button" className={`checklist-step ${audioReady ? "complete" : ""}`} onClick={() => handleChecklistStep("audio")}>
                  <span className="step-number">{audioReady ? "✓" : "1"}</span>
                  <span><strong>Audio routing</strong><small>{audioReady ? "Microphone and output selected" : "Choose input and output"}</small></span>
                </button>
                <button type="button" className={`checklist-step ${conversionReady ? "complete" : ""}`} onClick={() => handleChecklistStep("voice")} disabled={modelImportBusy || modelLoadBusy || conversionReady}>
                  <span className="step-number">{conversionReady ? "✓" : "2"}</span>
                  <span><strong>Voice conversion</strong><small>{conversionReady ? "Voice resident on GPU" : selectedModelIsPreviewOnly ? "Preview only · import a .pth voice to convert" : selectedModel.sourcePath ? selectedModelHasIndex ? "Load voice + retrieval index" : "Load voice (retrieval index optional)" : "Import a local .pth voice"}</small></span>
                </button>
                <button type="button" className={`checklist-step ${running ? "complete" : ""}`} onClick={() => handleChecklistStep("session")}>
                  <span className="step-number">{running ? "✓" : "3"}</span>
                  <span><strong>Start session</strong><small>{running ? engineLabel : startBlockedReason ?? (conversionReady ? "Start local conversion" : selectedModelIsPreviewOnly ? "Start passthrough preview" : "Starts in passthrough")}</small></span>
                </button>
              </div>
            </section>

            <section className="signal-card">
              <div className="signal-header">
                <div><h2>Signal monitor</h2><p>Microphone input and routed output</p></div>
                <div className={`engine-state ${engineTone}`}><span />{engineLabel}</div>
              </div>
              <div className="waveform-stage">
                <Waveform active={hasInputSignal} />
                {!hasInputSignal && <span className="waveform-empty">{waveformMessage}</span>}
              </div>
              <div className="signal-metrics">
                <div><span>Input peak</span><strong>{peakDb(engineStatus.inputPeak)} dB</strong></div>
                <div className="signal-route"><span>Input</span><i /><span>{monitorDevice ? "Output + Monitor" : "Output"}</span></div>
                <div><span>Output peak</span><strong>{peakDb(engineStatus.outputPeak)} dB</strong></div>
              </div>
            </section>

            <section className="controls-section">
              <div className="section-heading"><div><h2>Voice controls</h2><p>Shape the character of the converted voice</p></div><div className="section-heading-actions"><span>{selectedModelLoaded ? "Changes apply when the next audio session starts" : "Load an imported voice to enable conversion"}</span><button className="details-toggle" onClick={resetSelectedModelSettings} disabled={running}>Reset</button></div></div>
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
                  <button key={key} className={mode === key ? "active" : ""} aria-pressed={mode === key} disabled={running} onClick={() => updateSelectedSettings({ streamingPreset: key, chunkFrames: streamProfiles[key].hop * 48, extraFrames: streamProfiles[key].analysis * 48 })}>{modeLabels[key]}</button>
                ))}
              </div>
              <button className="secondary-button" aria-expanded={advancedOpen} onClick={() => setAdvancedOpen((value) => !value)}>{advancedOpen ? "Hide advanced" : "Advanced settings"}</button>
              {advancedOpen && (
                <div className="advanced-panel">
                  <div><span>Retrieval strength</span><strong>{selectedModelHasIndex ? `${indexRate}% ${selectedModelLoaded ? "loaded" : "on next load"}` : "No index available"}</strong></div>
                  <label className="advanced-select"><span>Index file</span><select aria-label="Index file" value={selectedSettings.indexPath ?? ""} disabled={running || !selectedModel.indexPaths?.length} onChange={(event) => updateSelectedSettings({ indexPath: event.target.value || null })}><option value="">{selectedModel.indexPaths?.length ? "Off" : "No index available"}</option>{selectedModel.indexPaths?.map((path) => <option key={path} value={path}>{windowsFileName(path)}</option>)}</select></label>
                  <div><span>Content embedder</span><strong>{selectedSettings.contentvecPath ? windowsFileName(selectedSettings.contentvecPath) : "Auto-discover ContentVec"}</strong></div>
                  <label className="advanced-select"><span>Target speaker</span><select aria-label="Target speaker" value={selectedSettings.speakerId} disabled={running || !selectedModelLoaded || (liveRvcStatus.speakerCount ?? 1) <= 1} onChange={(event) => updateSelectedSettings({ speakerId: Number(event.target.value) })}>{Array.from({ length: Math.max(1, liveRvcStatus.speakerCount ?? 1) }, (_, speakerId) => <option key={speakerId} value={speakerId}>Speaker {speakerId}</option>)}</select></label>
                  <label className="advanced-range"><span>RMVPE threshold <output>{selectedSettings.f0Threshold.toFixed(2)}</output></span><input aria-label="RMVPE threshold" type="range" min="1" max="20" value={Math.round(selectedSettings.f0Threshold * 100)} disabled={running} style={{ "--range-progress": `${((selectedSettings.f0Threshold - 0.01) / 0.19) * 100}%` } as CSSProperties} onChange={(event) => updateSelectedSettings({ f0Threshold: Number(event.target.value) / 100 })} /></label>
                  <label className="advanced-select"><span>Chunk / streaming hop</span><select aria-label="Chunk size" value={selectedSettings.chunkFrames} disabled={running} onChange={(event) => updateSelectedSettings({ chunkFrames: Number(event.target.value) })}>{CHUNK_OPTIONS.map((frames) => <option key={frames} value={frames}>{frameDurationLabel(frames)}</option>)}</select></label>
                  <label className="advanced-select"><span>Extra / context</span><select aria-label="Extra context" value={selectedSettings.extraFrames} disabled={running} onChange={(event) => updateSelectedSettings({ extraFrames: Number(event.target.value) })}>{EXTRA_OPTIONS.map((frames) => <option key={frames} value={frames}>{frameDurationLabel(frames)}</option>)}</select></label>
                  <div><span>Streaming hop</span><strong>{Number(streamProfile.hop.toFixed(1))} ms · {modeLabels[mode]}</strong></div>
                  <div><span>Analysis window</span><strong>{Number(streamProfile.analysis.toFixed(1))} ms selected</strong></div>
                  <div><span>SOLA overlap</span><strong>{streamProfile.overlap} ms + {streamProfile.search} ms search</strong></div>
                  <p className="advanced-note">Smaller chunks respond faster but can sound less stable. Extra adds context; the engine enforces enough context for safe SOLA stitching.</p>
                </div>
              )}
            </section>
          </div>
        </main>

        <aside className="setup-panel">
          <div className="panel-heading setup-heading">
            <div><span className="eyebrow">Session</span><h2>Audio setup</h2><p className="panel-subtitle">Route your local audio safely</p></div>
            <button className="details-toggle refresh-button" onClick={refreshDevices} disabled={running || deviceRefreshBusy}>{deviceRefreshBusy ? "Checking…" : "Refresh"}</button>
          </div>

          <section className="setup-section">
            <label className="device-field">
              <span className="field-label"><span className="device-glyph"><Icon name="microphone" /></span><span><strong>Microphone</strong><small>{deviceSummary(inputDevice)}</small></span></span>
              <select value={inputDeviceId} onChange={(event) => setInputDeviceId(event.target.value)} disabled={running}>
                {devices.inputs.map((device) => <option key={device.id} value={device.id}>{device.name}</option>)}
              </select>
            </label>
            <LevelMeter active={hasInputSignal} />

            <label className="device-field">
              <span className="field-label"><span className="device-glyph"><Icon name="speaker" /></span><span><strong>Output</strong><small>{deviceSummary(outputDevice)}</small></span></span>
              <select value={outputDeviceId} onChange={(event) => { const next = event.target.value; setOutputDeviceId(next); if (monitorDeviceId === next) setMonitorDeviceId(""); }} disabled={running}>
                {devices.outputs.map((device) => <option key={device.id} value={device.id}>{device.name}</option>)}
              </select>
            </label>
            <LevelMeter active={hasOutputSignal} output />

            <label className="device-field">
              <span className="field-label"><span className="device-glyph"><Icon name="speaker" /></span><span><strong>Monitor</strong><small>{monitorDevice ? `${deviceSummary(monitorDevice)} · headphones` : "Optional headphone monitor · off"}</small></span></span>
              <select aria-label="Monitor" value={monitorDeviceId} onChange={(event) => setMonitorDeviceId(event.target.value)} disabled={running}>
                <option value="">Off</option>
                {devices.outputs.filter((device) => device.id !== outputDeviceId).map((device) => <option key={device.id} value={device.id}>{device.name}</option>)}
              </select>
            </label>
            <LevelMeter active={hasMonitorSignal} output />

            <div className="audio-processing-controls">
              <div className="audio-processing-heading"><strong>Audio processing</strong><button className="details-toggle" onClick={resetAudioSettings} disabled={running}>Reset</button></div>
              <label><span><strong>Input gain</strong><output>{audioSettings.inputGainDb > 0 ? "+" : ""}{audioSettings.inputGainDb} dB</output></span><input aria-label="Input gain" type="range" min="-24" max="24" value={audioSettings.inputGainDb} disabled={running} style={{ "--range-progress": `${((audioSettings.inputGainDb + 24) / 48) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, inputGainDb: Number(event.target.value) }))} /></label>
              <label><span><strong>Output gain</strong><output>{audioSettings.outputGainDb > 0 ? "+" : ""}{audioSettings.outputGainDb} dB</output></span><input aria-label="Output gain" type="range" min="-24" max="12" value={audioSettings.outputGainDb} disabled={running} style={{ "--range-progress": `${((audioSettings.outputGainDb + 24) / 36) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, outputGainDb: Number(event.target.value) }))} /></label>
              <label><span><strong>Monitor gain</strong><output>{audioSettings.monitorGainDb > 0 ? "+" : ""}{audioSettings.monitorGainDb} dB</output></span><input aria-label="Monitor gain" type="range" min="-24" max="12" value={audioSettings.monitorGainDb} disabled={running || !monitorDeviceId} style={{ "--range-progress": `${((audioSettings.monitorGainDb + 24) / 36) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, monitorGainDb: Number(event.target.value) }))} /></label>
              <label><span><strong>Noise gate</strong><output>{audioSettings.noiseGateDb <= -80 ? "Off" : `${audioSettings.noiseGateDb} dB`}</output></span><input aria-label="Noise gate" type="range" min="-80" max="-20" value={audioSettings.noiseGateDb} disabled={running} style={{ "--range-progress": `${((audioSettings.noiseGateDb + 80) / 60) * 100}%` } as CSSProperties} onChange={(event) => setAudioSettings((current) => ({ ...current, noiseGateDb: Number(event.target.value) }))} /></label>
            </div>

            {!startupBusy && !audioReady && <div className="info-callout error" role="alert"><span>!</span><p><strong>Audio route incomplete.</strong> Refresh devices, then choose both a microphone and an output.</p></div>}
            {sampleRateMismatch && <div className="info-callout error" role="alert"><span>!</span><p><strong>Sample rates do not match.</strong> Choose devices using the same sample rate before starting audio.</p></div>}
            {selectedModelCanLoad && !inferenceRuntime.readyForRvc && !startupBusy && <div className="info-callout warning"><span>!</span><p><strong>RVC runtime needs attention.</strong> {inferenceRuntime.blockers[0] ?? "Check Engine details before loading this voice."}</p></div>}
            {workerRecovering && <div className="info-callout warning" role="status"><span>↻</span><p><strong>Voice worker is recovering.</strong> Audio stays live with silence while the model process restarts and warms again.</p></div>}
            <div className="info-callout warning"><span>!</span><p><strong>{engineStatus.state === "rvc" ? "Live RVC conversion is active." : selectedModelLoaded ? "The live RVC voice is loaded and warmed." : selectedModelIsPreviewOnly ? "ONNX preview selected; passthrough is available." : "Passthrough is active until a voice is loaded."}</strong> Use headphones to prevent microphone feedback.</p></div>
            {engineError && <div className="info-callout error" role="alert"><span>!</span><p>{engineError}</p></div>}
          </section>

          <section className="setup-section engine-summary">
            <div className="section-heading compact">
              <div><h3>Engine</h3><p>Local processing status</p></div>
              <div className="section-heading-meta"><span className="status-badge">{running ? engineLabel : selectedModelLoaded ? "Ready" : "Preview"}</span><button className="details-toggle" aria-expanded={diagnosticsOpen} onClick={() => setDiagnosticsOpen((value) => !value)}>{diagnosticsOpen ? "Hide" : "Details"}</button></div>
            </div>
            {diagnosticsOpen && <dl className="details-list">
              <div><dt>Audio backend</dt><dd>{devices.backend}</dd></div>
              <div><dt>Current stage</dt><dd>{running ? engineStatus.inferenceBackend : selectedModelLoaded ? "RVC warmed" : "RVC pending"}</dd></div>
              <div><dt>Model worker</dt><dd>{liveRvcStatus.workerState === "recovering" ? "Recovering" : liveRvcStatus.workerState === "failed" ? "Recovery failed" : liveRvcStatus.state === "ready" ? "Resident" : "Not loaded"}</dd></div>
              <div><dt>Worker restarts</dt><dd>{liveRvcStatus.workerRestarts ?? 0}</dd></div>
              <div><dt>Retrieval index</dt><dd>{liveRvcStatus.indexLoaded ? `${liveRvcStatus.indexVectorCount?.toLocaleString()} vectors` : "Not loaded"}</dd></div>
              <div><dt>Python sidecar</dt><dd>{inferenceRuntime.source === "python-sidecar" ? "Connected" : "Desktop only"}</dd></div>
              <div><dt>Python runtime</dt><dd>{inferenceRuntime.python.version}{inferenceRuntime.python.rvcEnvironmentCompatible ? "" : " · needs 3.11"}</dd></div>
              <div><dt>RVC packages</dt><dd>{inferenceRuntime.readyForRvc ? "Ready" : `${inferenceRuntime.blockers.length} blockers`}</dd></div>
              <div><dt>Pitch target</dt><dd>RMVPE</dd></div>
              <div><dt>F0 threshold</dt><dd>{selectedSettings.f0Threshold.toFixed(2)}</dd></div>
              <div><dt>Gain staging</dt><dd>{audioSettings.inputGainDb} / {audioSettings.outputGainDb} / {audioSettings.monitorGainDb} dB</dd></div>
              <div><dt>Monitor route</dt><dd>{engineStatus.monitorDeviceName ?? "Off"}</dd></div>
              <div><dt>Monitor buffer</dt><dd>{engineStatus.monitorDeviceId ? `${engineStatus.monitorBufferedFrames} fr · ${engineStatus.monitorUnderruns + engineStatus.monitorOverruns} XRuns` : "Off"}</dd></div>
              <div><dt>Output safety depth</dt><dd>{engineStatus.primeTargetFrames} fr · {engineStatus.reprimes} reprimes</dd></div>
              <div><dt>Clock corrections</dt><dd>{engineStatus.driftDroppedFrames} drop · {engineStatus.driftRepeatedFrames} repeat</dd></div>
              <div><dt>Monitor corrections</dt><dd>{engineStatus.monitorDeviceId ? `${engineStatus.monitorDriftDroppedFrames} drop · ${engineStatus.monitorDriftRepeatedFrames} repeat` : "Off"}</dd></div>
              <div><dt>Noise gate</dt><dd>{audioSettings.noiseGateDb <= -80 ? "Off" : `${audioSettings.noiseGateDb} dB`}</dd></div>
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

          <section className="setup-section latency-card">
            <div className="latency-heading"><span><strong>Streaming pipeline</strong><small>{modeLabels[mode]} preset · applies next session</small></span><strong>{streamProfile.hop}<small> ms hop</small></strong></div>
            <div className="latency-track"><i style={{ width: "11%" }} /><i style={{ width: "27%" }} /><i style={{ width: "40%" }} /><i style={{ width: "22%" }} /></div>
            <div className="latency-legend"><span>Capture</span><span>Features</span><span>Generator</span><span>SOLA</span></div>
            <p>The hop is not end-to-end latency. Physical loopback measurement is still pending.</p>
          </section>
        </aside>
      </div>

      <footer className="status-bar">
        <div className="status-primary"><span className={signalActive && engineStatus.state !== "preview" ? "active" : ""} />{sessionStatusLabel}</div>
        <div className="status-message" role="status" aria-live="polite">{engineError ?? (workerRecovering ? "Restarting the local voice worker and restoring the loaded model…" : modelLoadBusy ? `Loading and warming ${selectedModel.name} on CUDA…` : modelImportBusy ? "Inspecting the selected model locally…" : startupBusy ? "Discovering audio devices and local inference runtime…" : running ? `${engineLabel} · ${engineStatus.state === "rvc" ? "Local AI processing" : "Unconverted audio"}` : selectedModelLoaded ? "Voice resident on GPU · Ready for local conversion" : "All processing stays on this computer")}</div>
        <div className="status-items"><span>{engineStatus.sampleRate ? `${engineStatus.sampleRate / 1000} kHz` : "48 kHz target"}</span><span>Queue {engineStatus.bufferedFrames}/{engineStatus.primeTargetFrames} fr</span><span>XRuns {engineStatus.underruns + engineStatus.overruns + engineStatus.monitorUnderruns + engineStatus.monitorOverruns}</span></div>
      </footer>
    </div>
  );
}

export default App;

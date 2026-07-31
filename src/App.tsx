import { useEffect, useMemo, useState } from "react";
import "./App.css";
import {
  FALLBACK_PROFILE,
  MODEL_PRESETS,
  getSystemProfile,
  type ConversionMode,
  type SystemProfile,
} from "./lib/engine";

const modeLabels: Record<ConversionMode, string> = {
  quality: "Quality",
  balanced: "Balanced",
  latency: "Low latency",
};

function LevelMeter({ active, output = false }: { active: boolean; output?: boolean }) {
  return (
    <div className={`level-meter ${active ? "active" : ""}`} aria-label={active ? "Signal active" : "No signal"}>
      {Array.from({ length: 18 }, (_, index) => (
        <i key={index} className={index > 14 ? "peak" : ""} style={{ animationDelay: `${-(index + (output ? 4 : 0)) * 36}ms` }} />
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

function App() {
  const [profile, setProfile] = useState<SystemProfile>(FALLBACK_PROFILE);
  const [running, setRunning] = useState(false);
  const [bypassed, setBypassed] = useState(false);
  const [mode, setMode] = useState<ConversionMode>("balanced");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(68);
  const [protection, setProtection] = useState(33);
  const [monitor, setMonitor] = useState(true);
  const [modelId, setModelId] = useState(MODEL_PRESETS[0].id);

  useEffect(() => {
    getSystemProfile().then(setProfile);
  }, []);

  const selectedModel = useMemo(
    () => MODEL_PRESETS.find((model) => model.id === modelId) ?? MODEL_PRESETS[0],
    [modelId],
  );

  const converting = running && !bypassed;

  return (
    <div className="desktop-app">
      <div className="menu-bar" data-tauri-drag-region>
        <div className="app-identity">
          <span className="app-icon"><i /></span>
          <strong>VC Next</strong>
          <span className="prototype-label">Prototype</span>
        </div>
        <div className="desktop-menu">
          <button>File</button><button>Edit</button><button>View</button><button>Help</button>
        </div>
        <div className="session-name">Untitled voice session</div>
        <div className="hardware-state"><span />{profile.gpu.replace("NVIDIA GeForce ", "")}</div>
      </div>

      <div className="command-bar">
        <div className="workspace-tabs" role="tablist" aria-label="Workspace">
          <button className="active" role="tab" aria-selected="true">Live</button>
          <button role="tab" aria-selected="false" disabled>Models</button>
          <button role="tab" aria-selected="false" disabled>Recorder</button>
        </div>
        <div className="command-actions">
          <button className="toolbar-button">Import model</button>
          <button className="toolbar-button">Calibrate audio</button>
          <span className="toolbar-separator" />
          <button className={`bypass ${bypassed ? "active" : ""}`} onClick={() => setBypassed((value) => !value)}>
            {bypassed ? "Bypassed" : "Bypass"}
          </button>
          <button className={`transport-button ${running ? "stop" : ""}`} onClick={() => setRunning((value) => !value)} aria-pressed={running}>
            <span>{running ? "■" : "▶"}</span>{running ? "Stop" : "Start"}
          </button>
        </div>
      </div>

      <div className="workbench">
        <aside className="library-pane">
          <div className="pane-title"><strong>Model library</strong><button aria-label="Add model">＋</button></div>
          <div className="search-box"><span>⌕</span><input placeholder="Search models" aria-label="Search models" /></div>
          <div className="library-section-label">LOCAL MODELS</div>
          <div className="model-list">
            {MODEL_PRESETS.map((model) => (
              <button key={model.id} className={modelId === model.id ? "selected" : ""} onClick={() => setModelId(model.id)}>
                <span className="list-avatar">{model.initials}</span>
                <span><strong>{model.name}</strong><small>{model.format} · {model.sampleRate / 1000} kHz</small></span>
                {modelId === model.id && <i />}
              </button>
            ))}
          </div>
          <div className="empty-library-space">
            <span>Drop voice models here</span>
            <small>.pth, .onnx and .index</small>
          </div>
          <div className="library-footer"><span>2 models</span><button>Open model folder</button></div>
        </aside>

        <main className="editor-pane">
          <div className="editor-tabbar">
            <div className="editor-tab active"><span className="tab-dot" />{selectedModel.name}<button>×</button></div>
            <button className="new-tab">＋</button>
          </div>

          <div className="voice-editor">
            <div className="voice-header">
              <div className="voice-avatar">{selectedModel.initials}</div>
              <div><h1>{selectedModel.name}</h1><p>{selectedModel.format} model · Ready on CUDA</p></div>
              <button className="more-button" aria-label="Model actions">•••</button>
            </div>

            <div className="signal-workspace">
              <div className="signal-ruler"><span>INPUT</span><i /><span>LIVE SIGNAL</span><i /><span>OUTPUT</span></div>
              <Waveform active={converting} />
              <div className="signal-readout">
                <div><small>Input peak</small><strong>{running ? "−12.4" : "−∞"} dB</strong></div>
                <div className={`conversion-state ${converting ? "active" : ""}`}><span />{converting ? "Converting" : bypassed ? "Bypassed" : "Engine ready"}</div>
                <div><small>Output peak</small><strong>{converting ? "−8.7" : "−∞"} dB</strong></div>
              </div>
            </div>

            <div className="parameter-strip">
              <label className="parameter">
                <span><strong>Pitch</strong><output>{pitch > 0 ? "+" : ""}{pitch} st</output></span>
                <input type="range" min="-12" max="12" value={pitch} onChange={(event) => setPitch(Number(event.target.value))} />
                <small><span>−12</span><span>0</span><span>+12</span></small>
              </label>
              <label className="parameter">
                <span><strong>Similarity</strong><output>{indexRate}%</output></span>
                <input type="range" min="0" max="100" value={indexRate} onChange={(event) => setIndexRate(Number(event.target.value))} />
                <small><span>Natural</span><span>Target</span></small>
              </label>
              <label className="parameter">
                <span><strong>Protection</strong><output>{protection}%</output></span>
                <input type="range" min="0" max="50" value={protection} onChange={(event) => setProtection(Number(event.target.value))} />
                <small><span>Expressive</span><span>Stable</span></small>
              </label>
            </div>

            <div className="mode-bar">
              <span>Processing mode</span>
              <div>
                {(Object.keys(modeLabels) as ConversionMode[]).map((key) => (
                  <button key={key} className={mode === key ? "active" : ""} onClick={() => setMode(key)}>{modeLabels[key]}</button>
                ))}
              </div>
              <button className="advanced-link">Advanced parameters…</button>
            </div>
          </div>
        </main>

        <aside className="inspector-pane">
          <div className="pane-title"><strong>Session inspector</strong><button aria-label="Inspector options">•••</button></div>

          <section className="inspector-section">
            <h2>Audio devices <button>⌃</button></h2>
            <label className="device-control"><span>Input</span><button><strong>Default Windows input</strong><small>48,000 Hz · Mono</small><i>⌄</i></button></label>
            <LevelMeter active={running} />
            <label className="device-control"><span>Output</span><button><strong>Virtual output not configured</strong><small>Run audio calibration</small><i>⌄</i></button></label>
            <LevelMeter active={converting} output />
            <label className="check-row"><input type="checkbox" checked={monitor} onChange={(event) => setMonitor(event.target.checked)} /><span>Monitor converted voice</span></label>
          </section>

          <section className="inspector-section">
            <h2>Inference <button>⌃</button></h2>
            <dl className="property-list">
              <div><dt>Backend</dt><dd>ONNX Runtime CUDA</dd></div>
              <div><dt>Precision</dt><dd>FP16 target</dd></div>
              <div><dt>Pitch extractor</dt><dd>RMVPE</dd></div>
              <div><dt>GPU memory</dt><dd>{Math.round(profile.vramMb / 1024)} GB available</dd></div>
            </dl>
          </section>

          <section className="inspector-section latency-section">
            <h2>Latency budget <button>⌃</button></h2>
            <div className="latency-total"><span>Target total</span><strong>{mode === "latency" ? "< 90" : mode === "quality" ? "< 150" : "< 110"}<small> ms</small></strong></div>
            <div className="latency-track"><i style={{ width: "11%" }} /><i style={{ width: "27%" }} /><i style={{ width: "40%" }} /><i style={{ width: "22%" }} /></div>
            <dl className="latency-list">
              <div><dt><i className="capture" />Capture</dt><dd>10 ms</dd></div>
              <div><dt><i className="features" />Pitch + content</dt><dd>24 ms</dd></div>
              <div><dt><i className="inference" />Inference</dt><dd>36 ms</dd></div>
              <div><dt><i className="output" />Output</dt><dd>20 ms</dd></div>
            </dl>
            <p className="budget-warning">Target values—not measured yet.</p>
          </section>
        </aside>
      </div>

      <footer className="status-bar">
        <div className="status-primary"><span className={converting ? "active" : ""} />{converting ? "Converting" : "Ready"}</div>
        <div className="status-message">{bypassed ? "Audio is passing through without conversion" : "Local engine · No audio leaves this computer"}</div>
        <div className="status-items"><span>48 kHz</span><span>Buffer: pending</span><span>CUDA</span><span>Driver {profile.driverVersion}</span></div>
      </footer>
    </div>
  );
}

export default App;

import { useEffect, useMemo, useState } from "react";
import "./App.css";
import {
  FALLBACK_PROFILE,
  MODEL_PRESETS,
  getSystemProfile,
  type ConversionMode,
  type SystemProfile,
} from "./lib/engine";

const modeDetails: Record<ConversionMode, { label: string; hint: string; target: string }> = {
  quality: { label: "Quality", hint: "More context", target: "< 150 ms" },
  balanced: { label: "Balanced", hint: "Voice chat", target: "< 110 ms" },
  latency: { label: "Low latency", hint: "Tuned systems", target: "< 90 ms" },
};

function Meter({ active, offset = 0 }: { active: boolean; offset?: number }) {
  return (
    <div className={`meter ${active ? "is-active" : ""}`} aria-label={active ? "Signal detected" : "No signal"}>
      {Array.from({ length: 14 }, (_, index) => (
        <span
          key={index}
          style={{
            animationDelay: `${(index + offset) * -55}ms`,
            opacity: active ? undefined : Math.max(0.12, 0.45 - index * 0.025),
          }}
        />
      ))}
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
  const [modelId, setModelId] = useState(MODEL_PRESETS[0].id);

  useEffect(() => {
    getSystemProfile().then(setProfile);
  }, []);

  const selectedModel = useMemo(
    () => MODEL_PRESETS.find((model) => model.id === modelId) ?? MODEL_PRESETS[0],
    [modelId],
  );

  const engineState = running ? (bypassed ? "Bypassed" : "Converting") : "Ready";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><span /></div>
          <div>
            <strong>VC Next</strong>
            <small>Feasibility prototype</small>
          </div>
        </div>

        <nav aria-label="Primary navigation">
          <button className="nav-item active"><span>◉</span>Live</button>
          <button className="nav-item" disabled><span>◇</span>Models<em>Next</em></button>
          <button className="nav-item" disabled><span>⌁</span>Recordings</button>
          <button className="nav-item" disabled><span>⌘</span>Presets</button>
        </nav>

        <div className="sidebar-spacer" />
        <div className="gpu-card">
          <div className="gpu-card-head">
            <span className="status-dot" />
            <small>Reference hardware</small>
          </div>
          <strong>{profile.gpu.replace("NVIDIA GeForce ", "")}</strong>
          <span>{Math.round(profile.vramMb / 1024)} GB VRAM · CUDA target</span>
        </div>
        <button className="nav-item settings" disabled><span>⚙</span>Settings</button>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">LOCAL VOICE CONVERSION</p>
            <h1>Live studio</h1>
          </div>
          <div className="topbar-actions">
            <div className="engine-pill"><span className={running ? "online" : ""} />{engineState}</div>
            <button className={`bypass-button ${bypassed ? "active" : ""}`} onClick={() => setBypassed((value) => !value)}>
              {bypassed ? "Resume conversion" : "Bypass"}
            </button>
          </div>
        </header>

        <section className="device-row" aria-label="Audio devices">
          <div className="device-block">
            <span className="device-icon">IN</span>
            <div><small>Microphone</small><strong>Default Windows input</strong></div>
            <Meter active={running} />
            <button aria-label="Change microphone">⌄</button>
          </div>
          <div className="route-line"><i /><span>48 kHz</span><i /></div>
          <div className="device-block output">
            <span className="device-icon">OUT</span>
            <div><small>Virtual output</small><strong>Configure during calibration</strong></div>
            <Meter active={running && !bypassed} offset={4} />
            <button aria-label="Change output">⌄</button>
          </div>
        </section>

        <div className="workspace-grid">
          <section className="panel conversion-panel">
            <div className="panel-heading">
              <div><p className="eyebrow">ENGINE</p><h2>Conversion</h2></div>
              <span className="prototype-badge">UI PROTOTYPE</span>
            </div>

            <div className="power-area">
              <div className={`power-halo ${running ? "active" : ""}`}>
                <button className="power-button" onClick={() => setRunning((value) => !value)} aria-pressed={running}>
                  <span className="power-symbol">↯</span>
                  <strong>{running ? "Stop" : "Start"}</strong>
                  <small>{running ? "Conversion active" : "Ready to convert"}</small>
                </button>
              </div>
              <p>{running ? "Audio is flowing through the prototype pipeline." : "The native engine will connect here after calibration."}</p>
            </div>

            <div className="mode-switch" aria-label="Conversion mode">
              {(Object.keys(modeDetails) as ConversionMode[]).map((key) => (
                <button key={key} className={mode === key ? "active" : ""} onClick={() => setMode(key)}>
                  <strong>{modeDetails[key].label}</strong>
                  <small>{modeDetails[key].hint}</small>
                </button>
              ))}
            </div>
          </section>

          <section className="panel model-panel">
            <div className="panel-heading">
              <div><p className="eyebrow">VOICE</p><h2>Model</h2></div>
              <button className="text-button">Import model</button>
            </div>

            <label className="model-card">
              <span className="model-avatar">{selectedModel.initials}</span>
              <span className="model-meta"><small>Selected model</small><strong>{selectedModel.name}</strong><em>{selectedModel.format} · {selectedModel.sampleRate / 1000} kHz</em></span>
              <select value={modelId} onChange={(event) => setModelId(event.target.value)} aria-label="Select voice model">
                {MODEL_PRESETS.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </label>

            <div className="control-group">
              <div className="control-label"><span><strong>Pitch shift</strong><small>Preserves the model's target range</small></span><output>{pitch > 0 ? "+" : ""}{pitch} st</output></div>
              <input type="range" min="-12" max="12" value={pitch} onChange={(event) => setPitch(Number(event.target.value))} />
              <div className="scale"><span>-12</span><span>Natural</span><span>+12</span></div>
            </div>

            <div className="control-group">
              <div className="control-label"><span><strong>Voice similarity</strong><small>RVC retrieval-index blend</small></span><output>{indexRate}%</output></div>
              <input type="range" min="0" max="100" value={indexRate} onChange={(event) => setIndexRate(Number(event.target.value))} />
              <div className="scale"><span>Natural</span><span>Balanced</span><span>Target</span></div>
            </div>

            <button className="advanced-button">Advanced model controls <span>→</span></button>
          </section>
        </div>

        <section className="performance-panel">
          <div className="performance-title">
            <div><p className="eyebrow">TARGET BUDGET</p><h2>Latency pipeline</h2></div>
            <div className="latency-target"><small>{modeDetails[mode].label} mode</small><strong>{modeDetails[mode].target}</strong></div>
          </div>
          <div className="stage-grid">
            <div><span className="stage-icon">01</span><small>Capture</small><strong>10 ms</strong></div>
            <i />
            <div><span className="stage-icon">02</span><small>Pitch + content</small><strong>24 ms</strong></div>
            <i />
            <div><span className="stage-icon">03</span><small>RVC inference</small><strong>36 ms</strong></div>
            <i />
            <div><span className="stage-icon">04</span><small>Stitch + output</small><strong>20 ms</strong></div>
          </div>
          <div className="benchmark-note"><span>!</span>These are engineering budgets, not measured results. Physical loopback benchmarking is the next milestone.</div>
        </section>
      </main>
    </div>
  );
}

export default App;

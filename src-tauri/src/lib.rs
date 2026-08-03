mod audio;
mod inference;
mod live_sidecar;
mod sidecar;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
};

use audio::{
    AudioDeviceSnapshot, AudioEngine, AudioEngineStatus, AudioLoopbackTestResult,
    AudioProcessingSettings, AudioRouteTestResult,
};
use live_sidecar::LiveRvcService;
use serde::Serialize;

type SharedAudioEngine = Arc<Mutex<AudioEngine>>;
type SharedLiveRvcService = Arc<Mutex<LiveRvcService>>;

async fn run_blocking<T, F>(operation: &'static str, task: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(task)
        .await
        .map_err(|error| format!("{operation} worker stopped unexpectedly: {error}"))?
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SystemProfile {
    os: String,
    gpu: String,
    vram_mb: u32,
    driver_version: String,
    source: String,
}

#[tauri::command]
async fn get_system_profile() -> Result<SystemProfile, String> {
    run_blocking("System profile probe", || Ok(detect_system_profile())).await
}

fn detect_system_profile() -> SystemProfile {
    let os = if cfg!(windows) {
        "Windows".to_owned()
    } else {
        std::env::consts::OS.to_owned()
    };

    if let Some((gpu, vram_mb, driver_version)) = probe_nvidia_smi() {
        return SystemProfile {
            os,
            gpu,
            vram_mb,
            driver_version,
            source: "native-probe".to_owned(),
        };
    }

    if let Some((gpu, vram_mb, driver_version)) = probe_windows_video_controller() {
        return SystemProfile {
            os,
            gpu,
            vram_mb,
            driver_version,
            source: "native-probe".to_owned(),
        };
    }

    SystemProfile {
        os,
        gpu: "Unknown GPU".to_owned(),
        vram_mb: 0,
        driver_version: "Unavailable".to_owned(),
        source: "native-probe".to_owned(),
    }
}

fn probe_nvidia_smi() -> Option<(String, u32, String)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    parse_nvidia_smi_line(line)
}

fn parse_nvidia_smi_line(line: &str) -> Option<(String, u32, String)> {
    let mut fields = line.splitn(3, ',').map(str::trim);
    let gpu = fields.next()?.to_owned();
    let vram_mb = fields.next()?.parse::<u32>().ok()?;
    let driver = fields.next()?.to_owned();
    if gpu.is_empty() || driver.is_empty() {
        return None;
    }
    Some((gpu, vram_mb, driver))
}

fn probe_windows_video_controller() -> Option<(String, u32, String)> {
    if !cfg!(windows) {
        return None;
    }
    let script = r#"Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"#;
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let gpu = value.get("Name")?.as_str()?.trim().to_owned();
    let driver = value
        .get("DriverVersion")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("Unavailable")
        .trim()
        .to_owned();
    let vram_bytes = value
        .get("AdapterRAM")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    if gpu.is_empty() {
        return None;
    }
    Some((gpu, (vram_bytes / 1_048_576) as u32, driver))
}

#[tauri::command]
async fn get_audio_devices() -> Result<AudioDeviceSnapshot, String> {
    run_blocking("Audio device discovery", audio::enumerate_devices).await
}

#[tauri::command]
async fn start_audio_engine(
    input_device_id: String,
    output_device_id: String,
    monitor_device_id: Option<String>,
    input_gain_db: f32,
    output_gain_db: f32,
    monitor_gain_db: f32,
    noise_gate_db: f32,
    noise_suppression_strength: f32,
    echo_control_strength: f32,
    high_pass_enabled: bool,
    engine: tauri::State<'_, SharedAudioEngine>,
    live_rvc: tauri::State<'_, SharedLiveRvcService>,
) -> Result<AudioEngineStatus, String> {
    let engine = Arc::clone(engine.inner());
    let live_rvc = Arc::clone(live_rvc.inner());
    run_blocking("Audio engine startup", move || {
        let processing = AudioProcessingSettings::new(
            input_gain_db,
            output_gain_db,
            monitor_gain_db,
            noise_gate_db,
            noise_suppression_strength,
            echo_control_strength,
        )?
        .with_high_pass(high_pass_enabled);
        let live_client = live_rvc
            .lock()
            .map_err(|_| "The live RVC service lock is unavailable.".to_owned())?
            .ready_client();
        engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .start(
                &input_device_id,
                &output_device_id,
                monitor_device_id.as_deref(),
                live_client,
                processing,
            )
    })
    .await
}

#[tauri::command]
async fn restart_audio_engine(
    engine: tauri::State<'_, SharedAudioEngine>,
    live_rvc: tauri::State<'_, SharedLiveRvcService>,
) -> Result<AudioEngineStatus, String> {
    let engine = Arc::clone(engine.inner());
    let live_rvc = Arc::clone(live_rvc.inner());
    run_blocking("Audio engine recovery", move || {
        let live_client = live_rvc
            .lock()
            .map_err(|_| "The live RVC service lock is unavailable.".to_owned())?
            .ready_client();
        engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .restart(live_client)
    })
    .await
}

#[tauri::command]
async fn stop_audio_engine(
    engine: tauri::State<'_, SharedAudioEngine>,
) -> Result<AudioEngineStatus, String> {
    let engine = Arc::clone(engine.inner());
    run_blocking("Audio engine shutdown", move || {
        Ok(engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .stop())
    })
    .await
}

#[tauri::command]
async fn get_audio_engine_status(
    engine: tauri::State<'_, SharedAudioEngine>,
) -> Result<AudioEngineStatus, String> {
    let engine = Arc::clone(engine.inner());
    run_blocking("Audio engine status refresh", move || {
        Ok(engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .status())
    })
    .await
}

#[tauri::command]
async fn test_audio_routes(
    output_device_id: String,
    monitor_device_id: Option<String>,
    duration_ms: u32,
    engine: tauri::State<'_, SharedAudioEngine>,
) -> Result<AudioRouteTestResult, String> {
    let engine = Arc::clone(engine.inner());
    run_blocking("Audio route test", move || {
        if engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .is_running()
        {
            return Err("Stop audio before testing an output route.".to_owned());
        }
        audio::test_output_routes(&output_device_id, monitor_device_id.as_deref(), duration_ms)
    })
    .await
}

#[tauri::command]
async fn test_audio_loopback(
    input_device_id: String,
    output_device_id: String,
    duration_ms: u32,
    engine: tauri::State<'_, SharedAudioEngine>,
) -> Result<AudioLoopbackTestResult, String> {
    let engine = Arc::clone(engine.inner());
    run_blocking("Audio input/output loopback test", move || {
        if engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .is_running()
        {
            return Err("Stop audio before testing an input/output loopback.".to_owned());
        }
        audio::test_input_output_loopback(&input_device_id, &output_device_id, duration_ms)
    })
    .await
}

#[tauri::command]
async fn probe_inference_runtime() -> Result<serde_json::Value, String> {
    run_blocking("Inference runtime probe", sidecar::probe_runtime).await
}

#[tauri::command]
async fn open_runtime_setup() -> Result<String, String> {
    run_blocking("Runtime setup", || sidecar::open_runtime_setup()).await
}

#[tauri::command]
async fn get_runtime_setup_command() -> Result<String, String> {
    run_blocking("Runtime setup command", || sidecar::runtime_setup_command()).await
}

#[tauri::command]
async fn inspect_rvc_model(path: String) -> Result<serde_json::Value, String> {
    run_blocking("RVC model inspection", move || {
        sidecar::inspect_model(&path)
    })
    .await
}

const MODEL_SCAN_MAX_DEPTH: usize = 4;

fn is_supported_model_file(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase());
    if extension.as_deref() == Some("pth") {
        return true;
    }
    if extension.as_deref() != Some("onnx") {
        return false;
    }
    // A w-okada install also contains ContentVec, HuBERT, and RMVPE ONNX
    // assets. They are feature extractors, not importable voice generators.
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    !["contentvec", "hubert", "rmvpe", "embedder", "pitch"]
        .iter()
        .any(|marker| name.contains(marker))
}

fn scan_model_directory(
    directory: &Path,
    depth: usize,
    models: &mut Vec<String>,
) -> Result<(), String> {
    if depth > MODEL_SCAN_MAX_DEPTH {
        return Ok(());
    }
    let entries = fs::read_dir(directory).map_err(|error| {
        format!(
            "Could not read model folder {}: {error}",
            directory.display()
        )
    })?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("Could not inspect a model-folder entry: {error}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("Could not inspect {}: {error}", path.display()))?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            scan_model_directory(&path, depth + 1, models)?;
        } else if file_type.is_file() && is_supported_model_file(&path) {
            models.push(path.to_string_lossy().into_owned());
        }
    }
    Ok(())
}

#[tauri::command]
async fn discover_rvc_models(path: String) -> Result<Vec<String>, String> {
    run_blocking("RVC model folder scan", move || {
        let root = PathBuf::from(path)
            .canonicalize()
            .map_err(|error| format!("The selected model folder could not be opened: {error}"))?;
        if !root.is_dir() {
            return Err("Choose a folder containing RVC .pth or .onnx files.".to_owned());
        }
        let mut models = Vec::new();
        scan_model_directory(&root, 0, &mut models)?;
        models.sort_by_key(|model| model.to_ascii_lowercase());
        models.dedup_by(|left, right| left.eq_ignore_ascii_case(right));
        if models.is_empty() {
            return Err(format!(
                "No .pth or .onnx voice models were found within {} levels of {}.",
                MODEL_SCAN_MAX_DEPTH,
                root.display()
            ));
        }
        Ok(models)
    })
    .await
}

#[tauri::command]
async fn inspect_trusted_rvc_checkpoint(path: String) -> Result<serde_json::Value, String> {
    run_blocking("Trusted RVC checkpoint inspection", move || {
        sidecar::inspect_trusted_checkpoint(&path)
    })
    .await
}

#[tauri::command]
async fn load_live_rvc_model(
    model_path: String,
    index_path: Option<String>,
    contentvec_path: Option<String>,
    pitch_shift: f64,
    index_ratio: f64,
    protect_ratio: f64,
    speaker_id: i64,
    f0_threshold: f64,
    streaming_preset: String,
    chunk_frames: usize,
    extra_frames: usize,
    engine: tauri::State<'_, SharedAudioEngine>,
    live_rvc: tauri::State<'_, SharedLiveRvcService>,
) -> Result<serde_json::Value, String> {
    let engine = Arc::clone(engine.inner());
    let live_rvc = Arc::clone(live_rvc.inner());
    run_blocking("RVC model loading", move || {
        if engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .is_running()
        {
            return Err("Stop audio before loading a different RVC model.".to_owned());
        }
        live_rvc
            .lock()
            .map_err(|_| "The live RVC service lock is unavailable.".to_owned())?
            .load_model(
                &model_path,
                index_path.as_deref(),
                contentvec_path.as_deref(),
                pitch_shift,
                index_ratio,
                protect_ratio,
                speaker_id,
                f0_threshold,
                &streaming_preset,
                chunk_frames,
                extra_frames,
            )
    })
    .await
}

#[tauri::command]
async fn set_live_rvc_settings(
    pitch_shift: f64,
    index_ratio: f64,
    protect_ratio: f64,
    speaker_id: i64,
    f0_threshold: f64,
    streaming_preset: String,
    chunk_frames: usize,
    extra_frames: usize,
    live_rvc: tauri::State<'_, SharedLiveRvcService>,
) -> Result<serde_json::Value, String> {
    let live_rvc = Arc::clone(live_rvc.inner());
    run_blocking("RVC settings update", move || {
        live_rvc
            .lock()
            .map_err(|_| "The live RVC service lock is unavailable.".to_owned())?
            .set_settings(
                pitch_shift,
                index_ratio,
                protect_ratio,
                speaker_id,
                f0_threshold,
                &streaming_preset,
                chunk_frames,
                extra_frames,
            )
    })
    .await
}

#[tauri::command]
async fn get_live_rvc_status(
    live_rvc: tauri::State<'_, SharedLiveRvcService>,
) -> Result<serde_json::Value, String> {
    let live_rvc = Arc::clone(live_rvc.inner());
    run_blocking("RVC status refresh", move || {
        live_rvc
            .lock()
            .map_err(|_| "The live RVC service lock is unavailable.".to_owned())?
            .refresh_status()
    })
    .await
}

#[tauri::command]
async fn calibrate_live_rvc(
    live_rvc: tauri::State<'_, SharedLiveRvcService>,
) -> Result<serde_json::Value, String> {
    let live_rvc = Arc::clone(live_rvc.inner());
    run_blocking("RVC stream calibration", move || {
        live_rvc
            .lock()
            .map_err(|_| "The live RVC service lock is unavailable.".to_owned())?
            .calibrate()
    })
    .await
}

#[tauri::command]
async fn unload_live_rvc_model(
    engine: tauri::State<'_, SharedAudioEngine>,
    live_rvc: tauri::State<'_, SharedLiveRvcService>,
) -> Result<serde_json::Value, String> {
    let engine = Arc::clone(engine.inner());
    let live_rvc = Arc::clone(live_rvc.inner());
    run_blocking("RVC model unload", move || {
        if engine
            .lock()
            .map_err(|_| "The audio engine lock is unavailable.".to_owned())?
            .is_running()
        {
            return Err("Stop audio before unloading the RVC model.".to_owned());
        }
        live_rvc
            .lock()
            .map_err(|_| "The live RVC service lock is unavailable.".to_owned())?
            .unload()
    })
    .await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(Mutex::new(AudioEngine::default())))
        .manage(Arc::new(Mutex::new(LiveRvcService::default())))
        .invoke_handler(tauri::generate_handler![
            get_system_profile,
            get_audio_devices,
            start_audio_engine,
            restart_audio_engine,
            stop_audio_engine,
            get_audio_engine_status,
            test_audio_routes,
            test_audio_loopback,
            probe_inference_runtime,
            open_runtime_setup,
            get_runtime_setup_command,
            inspect_rvc_model,
            discover_rvc_models,
            inspect_trusted_rvc_checkpoint,
            load_live_rvc_model,
            set_live_rvc_settings,
            get_live_rvc_status,
            calibrate_live_rvc,
            unload_live_rvc_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running VC Next");
}

#[cfg(test)]
mod tests {
    use super::{is_supported_model_file, parse_nvidia_smi_line};
    use std::path::Path;

    #[test]
    fn parses_nvidia_smi_profile_line() {
        let profile = parse_nvidia_smi_line("NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 610.62")
            .expect("profile line should parse");
        assert_eq!(profile.0, "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert_eq!(profile.1, 6141);
        assert_eq!(profile.2, "610.62");
    }

    #[test]
    fn rejects_malformed_nvidia_smi_profile_line() {
        assert!(parse_nvidia_smi_line("NVIDIA,not-a-number,610.62").is_none());
        assert!(parse_nvidia_smi_line("NVIDIA,6141").is_none());
    }

    #[test]
    fn model_folder_scan_accepts_only_rvc_checkpoint_formats() {
        assert!(is_supported_model_file(Path::new("voice.PTH")));
        assert!(is_supported_model_file(Path::new("voice.onnx")));
        assert!(!is_supported_model_file(Path::new("contentvec-f.onnx")));
        assert!(!is_supported_model_file(Path::new("rmvpe_20231006.onnx")));
        assert!(!is_supported_model_file(Path::new("voice.index")));
        assert!(!is_supported_model_file(Path::new("voice.wav")));
    }
}

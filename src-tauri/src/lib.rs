mod audio;
mod inference;
mod live_sidecar;
mod sidecar;

use std::sync::{Arc, Mutex};

use audio::{AudioDeviceSnapshot, AudioEngine, AudioEngineStatus, AudioProcessingSettings};
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
    os: &'static str,
    gpu: &'static str,
    vram_mb: u32,
    driver_version: &'static str,
    source: &'static str,
}

#[tauri::command]
fn get_system_profile() -> SystemProfile {
    // Phase 0 records the reference machine explicitly. A native DXGI/NVML probe
    // replaces these baseline values when the audio-engine spike begins.
    SystemProfile {
        os: "Windows 11",
        gpu: "NVIDIA GeForce RTX 4050 Laptop GPU",
        vram_mb: 6141,
        driver_version: "610.62",
        source: "prototype-baseline",
    }
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
        )?;
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
async fn probe_inference_runtime() -> Result<serde_json::Value, String> {
    run_blocking("Inference runtime probe", sidecar::probe_runtime).await
}

#[tauri::command]
async fn inspect_rvc_model(path: String) -> Result<serde_json::Value, String> {
    run_blocking("RVC model inspection", move || {
        sidecar::inspect_model(&path)
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
            stop_audio_engine,
            get_audio_engine_status,
            probe_inference_runtime,
            inspect_rvc_model,
            inspect_trusted_rvc_checkpoint,
            load_live_rvc_model,
            set_live_rvc_settings,
            get_live_rvc_status,
            unload_live_rvc_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running VC Next");
}

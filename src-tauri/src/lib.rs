use serde::Serialize;

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![get_system_profile])
        .run(tauri::generate_context!())
        .expect("error while running VC Next");
}

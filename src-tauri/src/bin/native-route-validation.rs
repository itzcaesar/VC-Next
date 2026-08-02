//! Exercise the same Rust/CPAL + persistent Python path used by the Tauri host.
//!
//! This is intentionally a small diagnostic binary rather than a second audio
//! engine. It is useful on a reference Windows machine when we need evidence
//! from the native callback path without automating the full window.

#[path = "../audio.rs"]
mod audio;
#[path = "../inference.rs"]
mod inference;
#[path = "../live_sidecar.rs"]
mod live_sidecar;
#[path = "../sidecar.rs"]
mod sidecar;

use std::{env, fs, thread, time::Duration};

use audio::{
    enumerate_devices, test_output_routes, AudioDeviceSnapshot, AudioEngine, AudioEngineStatus,
    AudioProcessingSettings,
};
use live_sidecar::LiveRvcService;
use serde_json::{json, Value};

fn usage() {
    eprintln!(
        "Usage:\n  native-route-validation --list\n  native-route-validation --test-tone --output <device id or name> [--monitor <id/name>] [--milliseconds N]\n  native-route-validation --model <pth|onnx> --input <device id or name> --output <device id or name> [--monitor <id/name>] [--index <index>] [--contentvec <onnx>] [--seconds N] [--pitch N] [--index-ratio N] [--protect N] [--chunk N] [--extra N] [--preset quality|balanced|latency] [--high-pass] [--report <json path>]"
    );
}

fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0].eq_ignore_ascii_case(name))
        .map(|pair| pair[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|value| value.eq_ignore_ascii_case(name))
}

fn parse<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    option(args, name)
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|_| format!("Invalid value for {name}: {value}"))
        })
        .unwrap_or(Ok(default))
}

fn device_rows(snapshot: &AudioDeviceSnapshot, direction: &str) -> Vec<(String, String)> {
    let value = serde_json::to_value(snapshot).expect("audio device snapshot is serializable");
    value
        .get(direction)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|device| {
            Some((
                device.get("id")?.as_str()?.to_owned(),
                device.get("name")?.as_str()?.to_owned(),
            ))
        })
        .collect()
}

fn resolve_device(
    snapshot: &AudioDeviceSnapshot,
    direction: &str,
    requested: &str,
) -> Result<String, String> {
    let requested = requested.trim();
    if requested.is_empty() {
        return Err(format!("The {direction} device selector is empty."));
    }
    let rows = device_rows(snapshot, direction);
    if let Some((id, _)) = rows
        .iter()
        .find(|(id, _)| id.eq_ignore_ascii_case(requested))
    {
        return Ok(id.clone());
    }
    let needle = requested.to_ascii_lowercase();
    let matches = rows
        .iter()
        .filter(|(_, name)| name.to_ascii_lowercase().contains(&needle))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [(id, _)] => Ok((*id).clone()),
        [] => Err(format!(
            "No {direction} device matched {requested:?}. Run --list to see CPAL IDs and names."
        )),
        _ => Err(format!(
            "More than one {direction} device matched {requested:?}; use its exact CPAL id."
        )),
    }
}

fn print_devices(snapshot: &AudioDeviceSnapshot) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(snapshot)
            .map_err(|error| format!("Could not encode device list: {error}"))?
    );
    Ok(())
}

fn status_value(status: &AudioEngineStatus) -> Value {
    serde_json::to_value(status).expect("audio engine status is serializable")
}

fn status_metric(status: &AudioEngineStatus, key: &str) -> f32 {
    status_value(status)
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32
}

fn run(args: &[String]) -> Result<(), String> {
    let snapshot = enumerate_devices()?;
    if has_flag(args, "--list") {
        return print_devices(&snapshot);
    }
    if has_flag(args, "--test-tone") {
        let output_selector =
            option(args, "--output").ok_or_else(|| "--output is required.".to_owned())?;
        let output_id = resolve_device(&snapshot, "outputs", &output_selector)?;
        let monitor_id = option(args, "--monitor")
            .as_deref()
            .map(|selector| resolve_device(&snapshot, "outputs", selector))
            .transpose()?;
        let milliseconds: u32 = parse(args, "--milliseconds", 800_u32)?;
        let result = test_output_routes(&output_id, monitor_id.as_deref(), milliseconds)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&result)
                .map_err(|error| format!("Could not encode route test report: {error}"))?
        );
        return Ok(());
    }
    let model = option(args, "--model").ok_or_else(|| "--model is required.".to_owned())?;
    let input_selector =
        option(args, "--input").ok_or_else(|| "--input is required.".to_owned())?;
    let output_selector =
        option(args, "--output").ok_or_else(|| "--output is required.".to_owned())?;
    let monitor_selector = option(args, "--monitor");
    let input_id = resolve_device(&snapshot, "inputs", &input_selector)?;
    let output_id = resolve_device(&snapshot, "outputs", &output_selector)?;
    let monitor_id = monitor_selector
        .as_deref()
        .map(|selector| resolve_device(&snapshot, "outputs", selector))
        .transpose()?;
    let index = option(args, "--index");
    let contentvec = option(args, "--contentvec");
    let seconds: f64 = parse(args, "--seconds", 5.0_f64)?;
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 86_400.0 {
        return Err("--seconds must be between 0 and 86400.".to_owned());
    }
    let pitch: f64 = parse(args, "--pitch", 14.0)?;
    let index_ratio: f64 = parse(args, "--index-ratio", 0.30)?;
    let protect_ratio: f64 = parse(args, "--protect", 0.50)?;
    let chunk_frames: usize = parse(args, "--chunk", 24_000)?;
    let extra_frames: usize = parse(args, "--extra", 32_768)?;
    let preset = option(args, "--preset").unwrap_or_else(|| "quality".to_owned());
    let high_pass = has_flag(args, "--high-pass");
    let report_path = option(args, "--report");

    let mut service = LiveRvcService::default();
    let load_status = service.load_model(
        &model,
        index.as_deref(),
        contentvec.as_deref(),
        pitch,
        index_ratio,
        protect_ratio,
        0,
        0.30,
        &preset,
        chunk_frames,
        extra_frames,
    )?;
    let client = service
        .ready_client()
        .ok_or_else(|| "The live worker did not report a ready model.".to_owned())?;
    // Match the app's fidelity-first default; use the UI controls to opt in
    // to suppression when validating a noisy environment.
    let processing =
        AudioProcessingSettings::new(0.0, 0.0, -6.0, -80.0, 0.0, 0.0)?.with_high_pass(high_pass);
    let mut engine = AudioEngine::default();
    engine.start(
        &input_id,
        &output_id,
        monitor_id.as_deref(),
        Some(client),
        processing,
    )?;

    let started = std::time::Instant::now();
    let mut max_input_peak = 0.0_f32;
    let mut max_output_peak = 0.0_f32;
    let mut max_monitor_peak = 0.0_f32;
    while started.elapsed().as_secs_f64() < seconds {
        thread::sleep(Duration::from_millis(100));
        let status = engine.status();
        max_input_peak = max_input_peak.max(status_metric(&status, "inputPeak"));
        max_output_peak = max_output_peak.max(status_metric(&status, "outputPeak"));
        max_monitor_peak = max_monitor_peak.max(status_metric(&status, "monitorPeak"));
    }
    let final_status = engine.status();
    let live_status = service
        .refresh_status()
        .unwrap_or_else(|_| load_status.clone());
    let final_status_value = status_value(&final_status);
    let report = json!({
        "mode": "native-route",
        "requestedSeconds": seconds,
        "inputDevice": {"id": input_id, "name": final_status_value.get("inputDeviceName")},
        "outputDevice": {"id": output_id, "name": final_status_value.get("outputDeviceName")},
        "monitorDevice": {"id": monitor_id, "name": final_status_value.get("monitorDeviceName")},
        "loadStatus": load_status,
        "liveStatus": live_status,
        "maxInputPeak": max_input_peak,
        "maxOutputPeak": max_output_peak,
        "maxMonitorPeak": max_monitor_peak,
        "highPassRequested": high_pass,
        "audioStatus": final_status_value,
    });
    engine.stop();
    let _ = service.unload();
    let report_json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("Could not encode native route report: {error}"))?;
    if let Some(path) = report_path {
        fs::write(&path, format!("{report_json}\n"))
            .map_err(|error| format!("Could not write native route report {path:?}: {error}"))?;
        eprintln!("Native route report: {path}");
    }
    println!("{report_json}");
    Ok(())
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage();
        std::process::exit(2);
    }
    if let Err(error) = run(&args) {
        eprintln!("native-route-validation: {error}");
        std::process::exit(1);
    }
}

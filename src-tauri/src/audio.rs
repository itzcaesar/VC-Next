use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::{
    fs,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SampleFormat, Stream, StreamConfig,
};
use crossbeam_queue::ArrayQueue;
use serde::Serialize;

use crate::inference::{
    InferenceBackend, InferenceTelemetry, InferenceWorker, NoopInferenceBackend,
    INFERENCE_SAMPLE_RATE, WORKER_CHUNK_FRAMES,
};
use crate::live_sidecar::{LiveRvcClient, LiveRvcInferenceBackend};

const RING_BUFFER_FRAMES: usize = 96_000;
// Keep the far-end reference bounded so device-clock drift cannot accumulate stale data.
const ECHO_REFERENCE_RING_FRAMES: usize = 8_192;
const ECHO_HISTORY_FRAMES: usize = 2_048;
const ECHO_TAPS: usize = 64;
const ECHO_DELAY_FRAMES: usize = 480;
// Keep a little more output queued on consumer Windows routes. The previous
// 40 ms baseline was enough for short tests but could starve during an
// occasional scheduler spike on split-rate 44.1/48 kHz virtual cables. The
// cushion is still bounded and is reduced again after a sustained clean run.
const INITIAL_PRIME_FRAMES: usize = WORKER_CHUNK_FRAMES * 6;
const STARTUP_PRIME_FRAMES: usize = WORKER_CHUNK_FRAMES * 12;
const MAX_PRIME_FRAMES: usize = WORKER_CHUNK_FRAMES * 16;
const STARTUP_PRIME_TIMEOUT: Duration = Duration::from_millis(1_000);
const DRIFT_HIGH_MARGIN_FRAMES: usize = WORKER_CHUNK_FRAMES * 2;
const DRIFT_LOW_WATERMARK_FRAMES: usize = WORKER_CHUNK_FRAMES / 2;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDeviceInfo {
    id: String,
    name: String,
    is_default: bool,
    channels: u16,
    sample_rate: u32,
    sample_format: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDeviceSnapshot {
    inputs: Vec<AudioDeviceInfo>,
    outputs: Vec<AudioDeviceInfo>,
    default_input_id: Option<String>,
    default_output_id: Option<String>,
    backend: &'static str,
    source: &'static str,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEngineStatus {
    state: &'static str,
    input_device_id: Option<String>,
    output_device_id: Option<String>,
    monitor_device_id: Option<String>,
    input_device_name: Option<String>,
    output_device_name: Option<String>,
    monitor_device_name: Option<String>,
    sample_rate: Option<u32>,
    input_channels: Option<u16>,
    output_channels: Option<u16>,
    monitor_channels: Option<u16>,
    inference_sample_rate: u32,
    buffer_capacity_frames: usize,
    buffered_frames: usize,
    capture_buffered_frames: usize,
    captured_frames: u64,
    processed_frames: u64,
    played_frames: u64,
    monitor_buffered_frames: usize,
    monitor_played_frames: u64,
    underruns: u64,
    overruns: u64,
    monitor_underruns: u64,
    monitor_overruns: u64,
    prime_target_frames: usize,
    monitor_prime_target_frames: usize,
    reprimes: u64,
    monitor_reprimes: u64,
    drift_dropped_frames: u64,
    drift_repeated_frames: u64,
    monitor_drift_dropped_frames: u64,
    monitor_drift_repeated_frames: u64,
    inference_backend: &'static str,
    inference_stateful: bool,
    inference_chunk_frames: usize,
    inference_calls: u64,
    last_inference_micros: u64,
    max_inference_micros: u64,
    missed_inference_deadlines: u64,
    dropped_inference_frames: u64,
    inference_silence_suppressed_calls: u64,
    input_peak: f32,
    output_peak: f32,
    monitor_peak: f32,
    input_gain_db: f32,
    output_gain_db: f32,
    monitor_gain_db: f32,
    noise_gate_db: f32,
    noise_suppression_strength: f32,
    echo_control_strength: f32,
    high_pass_enabled: bool,
    last_error: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioRouteTestResult {
    output_device_name: String,
    monitor_device_name: Option<String>,
    duration_ms: u32,
    output_frames: u64,
    monitor_frames: u64,
    output_peak: f32,
    monitor_peak: f32,
    output_error: Option<String>,
    monitor_error: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioLoopbackTestResult {
    input_device_name: String,
    output_device_name: String,
    duration_ms: u32,
    input_frames: u64,
    output_frames: u64,
    input_peak: f32,
    output_peak: f32,
    signal_detected: bool,
    input_error: Option<String>,
    output_error: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FixturePlaybackResult {
    input_path: String,
    output_device_name: String,
    requested_seconds: f64,
    source_sample_rate: u32,
    output_sample_rate: u32,
    written_frames: u64,
    peak: f32,
    output_error: Option<String>,
}

#[derive(Clone, Copy)]
pub struct AudioProcessingSettings {
    input_gain_db: f32,
    output_gain_db: f32,
    monitor_gain_db: f32,
    noise_gate_db: f32,
    noise_suppression_strength: f32,
    echo_control_strength: f32,
    high_pass_enabled: bool,
}

impl AudioProcessingSettings {
    pub fn new(
        input_gain_db: f32,
        output_gain_db: f32,
        monitor_gain_db: f32,
        noise_gate_db: f32,
        noise_suppression_strength: f32,
        echo_control_strength: f32,
    ) -> Result<Self, String> {
        validate_db("Input gain", input_gain_db, -24.0, 24.0)?;
        validate_db("Output gain", output_gain_db, -24.0, 12.0)?;
        validate_db("Monitor gain", monitor_gain_db, -24.0, 12.0)?;
        validate_db("Noise gate", noise_gate_db, -80.0, -20.0)?;
        validate_strength("Noise suppression", noise_suppression_strength)?;
        validate_strength("Echo control", echo_control_strength)?;
        Ok(Self {
            input_gain_db,
            output_gain_db,
            monitor_gain_db,
            noise_gate_db,
            noise_suppression_strength,
            echo_control_strength,
            high_pass_enabled: false,
        })
    }

    pub fn with_high_pass(mut self, enabled: bool) -> Self {
        self.high_pass_enabled = enabled;
        self
    }
}

impl Default for AudioProcessingSettings {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            monitor_gain_db: -6.0,
            noise_gate_db: -80.0,
            // Fidelity-first default; suppression remains opt-in for noisy rooms.
            noise_suppression_strength: 0.0,
            echo_control_strength: 0.0,
            high_pass_enabled: false,
        }
    }
}

#[derive(Default)]
pub struct AudioEngine {
    active: Option<ActiveAudioEngine>,
}

struct ActiveAudioEngine {
    _input_stream: Stream,
    _output_stream: Stream,
    _monitor_stream: Option<Stream>,
    input_device_id: String,
    output_device_id: String,
    monitor_device_id: Option<String>,
    input_device_name: String,
    output_device_name: String,
    monitor_device_name: Option<String>,
    sample_rate: u32,
    input_channels: u16,
    output_channels: u16,
    monitor_channels: Option<u16>,
    inference_sample_rate: u32,
    state: &'static str,
    processing: AudioProcessingSettings,
    telemetry: Arc<AudioTelemetry>,
    inference_worker: InferenceWorker,
}

struct AudioTelemetry {
    capture_ring: Arc<ArrayQueue<f32>>,
    playback_ring: Arc<ArrayQueue<f32>>,
    monitor_ring: Option<Arc<ArrayQueue<f32>>>,
    echo_reference_ring: Arc<ArrayQueue<f32>>,
    inference: Arc<InferenceTelemetry>,
    captured_frames: AtomicU64,
    played_frames: AtomicU64,
    monitor_played_frames: AtomicU64,
    underruns: AtomicU64,
    overruns: AtomicU64,
    monitor_underruns: AtomicU64,
    monitor_overruns: AtomicU64,
    prime_target_frames: AtomicUsize,
    monitor_prime_target_frames: AtomicUsize,
    reprimes: AtomicU64,
    monitor_reprimes: AtomicU64,
    drift_dropped_frames: AtomicU64,
    drift_repeated_frames: AtomicU64,
    monitor_drift_dropped_frames: AtomicU64,
    monitor_drift_repeated_frames: AtomicU64,
    monitor_enabled: AtomicBool,
    input_peak_bits: AtomicU32,
    output_peak_bits: AtomicU32,
    monitor_peak_bits: AtomicU32,
    primed: AtomicBool,
    monitor_primed: AtomicBool,
    last_error: Mutex<Option<String>>,
}

impl AudioTelemetry {
    fn new(
        capture_ring: Arc<ArrayQueue<f32>>,
        playback_ring: Arc<ArrayQueue<f32>>,
        monitor_ring: Option<Arc<ArrayQueue<f32>>>,
        echo_reference_ring: Arc<ArrayQueue<f32>>,
        inference: Arc<InferenceTelemetry>,
    ) -> Self {
        let monitor_enabled = monitor_ring.is_some();
        Self {
            capture_ring,
            playback_ring,
            monitor_ring,
            echo_reference_ring,
            inference,
            captured_frames: AtomicU64::new(0),
            played_frames: AtomicU64::new(0),
            monitor_played_frames: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            overruns: AtomicU64::new(0),
            monitor_underruns: AtomicU64::new(0),
            monitor_overruns: AtomicU64::new(0),
            prime_target_frames: AtomicUsize::new(INITIAL_PRIME_FRAMES),
            monitor_prime_target_frames: AtomicUsize::new(INITIAL_PRIME_FRAMES),
            reprimes: AtomicU64::new(0),
            monitor_reprimes: AtomicU64::new(0),
            drift_dropped_frames: AtomicU64::new(0),
            drift_repeated_frames: AtomicU64::new(0),
            monitor_drift_dropped_frames: AtomicU64::new(0),
            monitor_drift_repeated_frames: AtomicU64::new(0),
            monitor_enabled: AtomicBool::new(monitor_enabled),
            input_peak_bits: AtomicU32::new(0),
            output_peak_bits: AtomicU32::new(0),
            monitor_peak_bits: AtomicU32::new(0),
            primed: AtomicBool::new(false),
            monitor_primed: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }
    }

    fn record_error(&self, message: String) {
        if let Ok(mut last_error) = self.last_error.lock() {
            *last_error = Some(message);
        }
    }

    fn disable_monitor(&self, message: String) {
        self.monitor_enabled.store(false, Ordering::Release);
        self.record_error(message);
    }
}

impl AudioEngine {
    pub fn is_running(&self) -> bool {
        self.active.is_some()
    }

    pub fn start(
        &mut self,
        input_id: &str,
        output_id: &str,
        monitor_id: Option<&str>,
        live_client: Option<LiveRvcClient>,
        processing: AudioProcessingSettings,
    ) -> Result<AudioEngineStatus, String> {
        self.stop();

        let monitor_id = monitor_id.filter(|id| !id.trim().is_empty());
        if monitor_id == Some(output_id) {
            return Err(
                "Monitor must use a different device from the main output. Choose headphones or turn monitoring off."
                    .to_owned(),
            );
        }

        let host = cpal::default_host();
        let (input_device, input_name) = find_device(&host, DeviceDirection::Input, input_id)?;
        let (output_device, output_name) = find_device(&host, DeviceDirection::Output, output_id)?;
        let mut monitor_route_error = None;
        let monitor_device =
            monitor_id.and_then(|id| match find_device(&host, DeviceDirection::Output, id) {
                Ok(device) => Some(device),
                Err(error) => {
                    monitor_route_error = Some(format!(
                        "Monitor disabled; the selected device is unavailable: {error}"
                    ));
                    None
                }
            });

        let input_supported = input_device
            .default_input_config()
            .map_err(|error| format!("Could not read the input format: {error}"))?;
        let output_supported = output_device
            .default_output_config()
            .map_err(|error| format!("Could not read the output format: {error}"))?;
        let monitor_supported =
            monitor_device
                .as_ref()
                .and_then(|(device, _)| match device.default_output_config() {
                    Ok(config) => Some(config),
                    Err(error) => {
                        monitor_route_error = Some(format!(
                            "Monitor disabled; the selected device format is unavailable: {error}"
                        ));
                        None
                    }
                });

        let input_rate = input_supported.sample_rate();

        let input_channels = input_supported.channels();
        let output_channels = output_supported.channels();
        let monitor_channels = monitor_supported.as_ref().map(|config| config.channels());
        let input_config: StreamConfig = input_supported.into();
        let output_config: StreamConfig = output_supported.into();
        let monitor_config = monitor_supported
            .as_ref()
            .map(|config| config.clone().into());
        let capture_ring = Arc::new(ArrayQueue::new(RING_BUFFER_FRAMES));
        let playback_ring = Arc::new(ArrayQueue::new(RING_BUFFER_FRAMES));
        let monitor_ring = monitor_id.map(|_| Arc::new(ArrayQueue::new(RING_BUFFER_FRAMES)));
        let echo_reference_ring = Arc::new(ArrayQueue::new(ECHO_REFERENCE_RING_FRAMES));
        let (backend, state): (Box<dyn InferenceBackend>, &'static str) = match live_client {
            Some(client) => (Box::new(LiveRvcInferenceBackend::new(client)), "rvc"),
            None => (Box::new(NoopInferenceBackend), "passthrough"),
        };
        let inference_worker = InferenceWorker::start(
            backend,
            Arc::clone(&capture_ring),
            Arc::clone(&playback_ring),
            input_rate,
        )?;
        let telemetry = Arc::new(AudioTelemetry::new(
            capture_ring,
            playback_ring,
            monitor_ring,
            echo_reference_ring,
            inference_worker.telemetry(),
        ));
        if let Some(error) = monitor_route_error {
            telemetry.disable_monitor(error);
        }

        let input_stream = build_input_stream(
            &input_device,
            input_supported.sample_format(),
            &input_config,
            Arc::clone(&telemetry),
            processing,
        )?;
        let output_stream = build_output_stream(
            &output_device,
            output_supported.sample_format(),
            &output_config,
            Arc::clone(&telemetry),
            processing,
        )?;
        let mut monitor_stream = match (
            telemetry.monitor_enabled.load(Ordering::Acquire),
            monitor_device.as_ref(),
            monitor_supported.as_ref(),
            monitor_config.as_ref(),
        ) {
            (false, _, _, _) => None,
            (true, Some((device, _)), Some(supported), Some(config)) => {
                match build_monitor_stream(
                    device,
                    supported.sample_format(),
                    config,
                    Arc::clone(&telemetry),
                    processing,
                ) {
                    Ok(stream) => Some(stream),
                    Err(error) => {
                        // Monitoring is an optional side route. A busy or temporarily
                        // incompatible monitor endpoint must not take down the main
                        // converted output; keep the session alive and expose the reason
                        // through the normal audio status error field.
                        telemetry.disable_monitor(format!(
                            "Monitor disabled; the selected device could not be opened: {error}"
                        ));
                        None
                    }
                }
            }
            _ => None,
        };

        // Let capture and inference establish the first playback cushion before opening the
        // output device.  Starting all three streams at once made the output callback reach its
        // prime threshold while the first CUDA/RVC call was still warming up, producing a
        // one-time click/underrun even though the steady-state route was healthy.  The bounded
        // wait keeps passthrough and silent devices responsive while protecting the real-time
        // callback from startup starvation.
        input_stream
            .play()
            .map_err(|error| format!("Could not start the input stream: {error}"))?;
        wait_for_startup_prime(&telemetry.playback_ring);
        output_stream
            .play()
            .map_err(|error| format!("Could not start the output stream: {error}"))?;
        if let Some(stream) = monitor_stream.as_ref() {
            if let Err(error) = stream.play() {
                telemetry.disable_monitor(format!(
                    "Monitor disabled; the selected device could not start: {error}"
                ));
                monitor_stream = None;
            }
        }

        let monitor_active = monitor_stream.is_some();

        self.active = Some(ActiveAudioEngine {
            _input_stream: input_stream,
            _output_stream: output_stream,
            _monitor_stream: monitor_stream,
            input_device_id: input_id.to_owned(),
            output_device_id: output_id.to_owned(),
            monitor_device_id: monitor_active
                .then(|| monitor_id)
                .flatten()
                .map(str::to_owned),
            input_device_name: input_name,
            output_device_name: output_name,
            monitor_device_name: monitor_active
                .then(|| monitor_device)
                .flatten()
                .map(|(_, name)| name),
            sample_rate: input_rate,
            input_channels,
            output_channels,
            monitor_channels: monitor_active.then_some(monitor_channels).flatten(),
            inference_sample_rate: INFERENCE_SAMPLE_RATE,
            state,
            processing,
            telemetry,
            inference_worker,
        });

        Ok(self.status())
    }

    pub fn stop(&mut self) -> AudioEngineStatus {
        if let Some(mut active) = self.active.take() {
            active.inference_worker.stop();
        }
        self.status()
    }

    /// Recreate the native streams using the last selected route and processing settings.
    /// Device callbacks cannot safely rebuild CPAL streams themselves, so recovery is exposed
    /// as an explicit host-thread operation for the UI and future reconnect watcher.
    pub fn restart(
        &mut self,
        live_client: Option<LiveRvcClient>,
    ) -> Result<AudioEngineStatus, String> {
        let Some(active) = self.active.as_ref() else {
            return Err("There is no active audio session to restart.".to_owned());
        };
        let input_id = active.input_device_id.clone();
        let output_id = active.output_device_id.clone();
        let monitor_id = active.monitor_device_id.clone();
        let processing = active.processing;
        self.stop();
        self.start(
            &input_id,
            &output_id,
            monitor_id.as_deref(),
            live_client,
            processing,
        )
    }

    pub fn status(&self) -> AudioEngineStatus {
        let Some(active) = &self.active else {
            return AudioEngineStatus {
                state: "stopped",
                input_device_id: None,
                output_device_id: None,
                monitor_device_id: None,
                input_device_name: None,
                output_device_name: None,
                monitor_device_name: None,
                sample_rate: None,
                input_channels: None,
                output_channels: None,
                monitor_channels: None,
                inference_sample_rate: INFERENCE_SAMPLE_RATE,
                buffer_capacity_frames: RING_BUFFER_FRAMES,
                buffered_frames: 0,
                capture_buffered_frames: 0,
                captured_frames: 0,
                processed_frames: 0,
                played_frames: 0,
                monitor_buffered_frames: 0,
                monitor_played_frames: 0,
                underruns: 0,
                overruns: 0,
                monitor_underruns: 0,
                monitor_overruns: 0,
                prime_target_frames: INITIAL_PRIME_FRAMES,
                monitor_prime_target_frames: INITIAL_PRIME_FRAMES,
                reprimes: 0,
                monitor_reprimes: 0,
                drift_dropped_frames: 0,
                drift_repeated_frames: 0,
                monitor_drift_dropped_frames: 0,
                monitor_drift_repeated_frames: 0,
                inference_backend: "Not running",
                inference_stateful: false,
                inference_chunk_frames: WORKER_CHUNK_FRAMES,
                inference_calls: 0,
                last_inference_micros: 0,
                max_inference_micros: 0,
                missed_inference_deadlines: 0,
                dropped_inference_frames: 0,
                inference_silence_suppressed_calls: 0,
                input_peak: 0.0,
                output_peak: 0.0,
                monitor_peak: 0.0,
                input_gain_db: 0.0,
                output_gain_db: 0.0,
                monitor_gain_db: -6.0,
                noise_gate_db: -80.0,
                noise_suppression_strength: 0.0,
                echo_control_strength: 0.0,
                high_pass_enabled: false,
                last_error: None,
            };
        };

        let telemetry = &active.telemetry;
        let inference = telemetry.inference.snapshot();
        let audio_error = telemetry
            .last_error
            .lock()
            .ok()
            .and_then(|error| error.clone());
        AudioEngineStatus {
            state: active.state,
            input_device_id: Some(active.input_device_id.clone()),
            output_device_id: Some(active.output_device_id.clone()),
            monitor_device_id: active.monitor_device_id.clone(),
            input_device_name: Some(active.input_device_name.clone()),
            output_device_name: Some(active.output_device_name.clone()),
            monitor_device_name: active.monitor_device_name.clone(),
            sample_rate: Some(active.sample_rate),
            input_channels: Some(active.input_channels),
            output_channels: Some(active.output_channels),
            monitor_channels: active.monitor_channels,
            inference_sample_rate: active.inference_sample_rate,
            buffer_capacity_frames: RING_BUFFER_FRAMES,
            buffered_frames: telemetry.playback_ring.len(),
            capture_buffered_frames: telemetry.capture_ring.len(),
            captured_frames: telemetry.captured_frames.load(Ordering::Relaxed),
            processed_frames: inference.processed_frames,
            played_frames: telemetry.played_frames.load(Ordering::Relaxed),
            monitor_buffered_frames: telemetry.monitor_ring.as_ref().map_or(0, |ring| ring.len()),
            monitor_played_frames: telemetry.monitor_played_frames.load(Ordering::Relaxed),
            underruns: telemetry.underruns.load(Ordering::Relaxed),
            overruns: telemetry.overruns.load(Ordering::Relaxed),
            monitor_underruns: telemetry.monitor_underruns.load(Ordering::Relaxed),
            monitor_overruns: telemetry.monitor_overruns.load(Ordering::Relaxed),
            prime_target_frames: telemetry.prime_target_frames.load(Ordering::Relaxed),
            monitor_prime_target_frames: telemetry
                .monitor_prime_target_frames
                .load(Ordering::Relaxed),
            reprimes: telemetry.reprimes.load(Ordering::Relaxed),
            monitor_reprimes: telemetry.monitor_reprimes.load(Ordering::Relaxed),
            drift_dropped_frames: telemetry.drift_dropped_frames.load(Ordering::Relaxed),
            drift_repeated_frames: telemetry.drift_repeated_frames.load(Ordering::Relaxed),
            monitor_drift_dropped_frames: telemetry
                .monitor_drift_dropped_frames
                .load(Ordering::Relaxed),
            monitor_drift_repeated_frames: telemetry
                .monitor_drift_repeated_frames
                .load(Ordering::Relaxed),
            inference_backend: inference.backend_name,
            inference_stateful: inference.stateful,
            inference_chunk_frames: WORKER_CHUNK_FRAMES,
            inference_calls: inference.process_calls,
            last_inference_micros: inference.last_process_micros,
            max_inference_micros: inference.max_process_micros,
            missed_inference_deadlines: inference.missed_deadlines,
            dropped_inference_frames: inference.dropped_output_frames,
            inference_silence_suppressed_calls: inference.silence_suppressed_calls,
            input_peak: f32::from_bits(telemetry.input_peak_bits.load(Ordering::Relaxed)),
            output_peak: f32::from_bits(telemetry.output_peak_bits.load(Ordering::Relaxed)),
            monitor_peak: f32::from_bits(telemetry.monitor_peak_bits.load(Ordering::Relaxed)),
            input_gain_db: active.processing.input_gain_db,
            output_gain_db: active.processing.output_gain_db,
            monitor_gain_db: active.processing.monitor_gain_db,
            noise_gate_db: active.processing.noise_gate_db,
            noise_suppression_strength: active.processing.noise_suppression_strength,
            echo_control_strength: active.processing.echo_control_strength,
            high_pass_enabled: active.processing.high_pass_enabled,
            last_error: audio_error.or(inference.last_error),
        }
    }
}

/// Play a short, bounded tone on the selected output and optional monitor
/// routes. This deliberately bypasses the inference engine: it is a setup
/// diagnostic for Windows endpoints, not a production audio path.
pub fn test_output_routes(
    output_id: &str,
    monitor_id: Option<&str>,
    duration_ms: u32,
) -> Result<AudioRouteTestResult, String> {
    if !(100..=5_000).contains(&duration_ms) {
        return Err("Route test duration must be between 100 and 5,000 ms.".to_owned());
    }
    let monitor_id = monitor_id.filter(|id| !id.trim().is_empty());
    if monitor_id == Some(output_id) {
        return Err(
            "Monitor must use a different device from the main output. Choose headphones or turn monitoring off."
                .to_owned(),
        );
    }

    let host = cpal::default_host();
    let (output_device, output_name) = find_device(&host, DeviceDirection::Output, output_id)?;
    let output_supported = output_device
        .default_output_config()
        .map_err(|error| format!("Could not read the output format: {error}"))?;
    let output_state = Arc::new(ToneTelemetry::new(
        duration_ms,
        output_supported.sample_rate(),
    ));
    let output_stream = build_tone_stream(
        &output_device,
        output_supported.sample_format(),
        &output_supported.clone().into(),
        Arc::clone(&output_state),
    )?;

    let mut monitor_name = None;
    let mut monitor_state = None;
    let mut monitor_stream = None;
    let mut monitor_error = None;
    if let Some(id) = monitor_id {
        match find_device(&host, DeviceDirection::Output, id) {
            Ok((device, name)) => match device.default_output_config() {
                Ok(supported) => {
                    monitor_name = Some(name);
                    let state = Arc::new(ToneTelemetry::new(duration_ms, supported.sample_rate()));
                    match build_tone_stream(
                        &device,
                        supported.sample_format(),
                        &supported.clone().into(),
                        Arc::clone(&state),
                    ) {
                        Ok(stream) => {
                            monitor_state = Some(state);
                            monitor_stream = Some(stream);
                        }
                        Err(error) => monitor_error = Some(error),
                    }
                }
                Err(error) => {
                    monitor_error = Some(format!("Could not read the monitor format: {error}"));
                }
            },
            Err(error) => monitor_error = Some(error),
        }
    }

    output_stream
        .play()
        .map_err(|error| format!("Could not start the output route test: {error}"))?;
    if let Some(stream) = monitor_stream.as_ref() {
        if let Err(error) = stream.play() {
            monitor_error = Some(format!("Could not start the monitor route test: {error}"));
            monitor_stream = None;
        }
    }

    thread::sleep(Duration::from_millis(duration_ms as u64 + 50));
    drop(monitor_stream);
    drop(output_stream);

    let output_frames = output_state.frames.load(Ordering::Relaxed);
    let output_error = output_state.last_error().or_else(|| {
        (output_frames == 0).then(|| "The output callback did not deliver any frames.".to_owned())
    });
    let monitor_frames = monitor_state
        .as_ref()
        .map(|state| state.frames.load(Ordering::Relaxed))
        .unwrap_or(0);
    let monitor_error = monitor_error.or_else(|| {
        monitor_state.as_ref().and_then(|state| {
            state.last_error().or_else(|| {
                (monitor_frames == 0)
                    .then(|| "The monitor callback did not deliver any frames.".to_owned())
            })
        })
    });

    Ok(AudioRouteTestResult {
        output_device_name: output_name,
        monitor_device_name: monitor_name,
        duration_ms,
        output_frames,
        monitor_frames,
        output_peak: f32::from_bits(output_state.peak_bits.load(Ordering::Relaxed)),
        monitor_peak: monitor_state
            .as_ref()
            .map(|state| f32::from_bits(state.peak_bits.load(Ordering::Relaxed)))
            .unwrap_or(0.0),
        output_error,
        monitor_error,
    })
}

/// Emit a bounded setup tone while listening on a selected input endpoint.
///
/// This is intentionally separate from the production engine. It answers the
/// practical setup question that an output callback test cannot: does the
/// selected return input actually receive audio from the selected output? A
/// quiet endpoint is reported as a route result rather than treated as a
/// model or inference failure.
pub fn test_input_output_loopback(
    input_id: &str,
    output_id: &str,
    duration_ms: u32,
) -> Result<AudioLoopbackTestResult, String> {
    if !(250..=5_000).contains(&duration_ms) {
        return Err("Loopback test duration must be between 250 and 5,000 ms.".to_owned());
    }

    let host = cpal::default_host();
    let (input_device, input_name) = find_device(&host, DeviceDirection::Input, input_id)?;
    let (output_device, output_name) = find_device(&host, DeviceDirection::Output, output_id)?;
    let input_supported = input_device
        .default_input_config()
        .map_err(|error| format!("Could not read the loopback input format: {error}"))?;
    let output_supported = output_device
        .default_output_config()
        .map_err(|error| format!("Could not read the loopback output format: {error}"))?;

    let input_state = Arc::new(InputProbeTelemetry::default());
    let output_state = Arc::new(ToneTelemetry::new(
        duration_ms,
        output_supported.sample_rate(),
    ));
    let input_stream = build_input_probe_stream(
        &input_device,
        input_supported.sample_format(),
        &input_supported.clone().into(),
        Arc::clone(&input_state),
    )?;
    let output_stream = build_tone_stream(
        &output_device,
        output_supported.sample_format(),
        &output_supported.clone().into(),
        Arc::clone(&output_state),
    )?;

    input_stream
        .play()
        .map_err(|error| format!("Could not start the loopback input route: {error}"))?;
    output_stream
        .play()
        .map_err(|error| format!("Could not start the loopback output route: {error}"))?;
    thread::sleep(Duration::from_millis(duration_ms as u64 + 100));
    drop(output_stream);
    drop(input_stream);

    let input_peak = f32::from_bits(input_state.peak_bits.load(Ordering::Relaxed));
    let output_peak = f32::from_bits(output_state.peak_bits.load(Ordering::Relaxed));
    Ok(AudioLoopbackTestResult {
        input_device_name: input_name,
        output_device_name: output_name,
        duration_ms,
        input_frames: input_state.frames.load(Ordering::Relaxed),
        output_frames: output_state.frames.load(Ordering::Relaxed),
        input_peak,
        output_peak,
        // The setup tone is emitted at 0.08 peak. A 0.003 threshold leaves
        // headroom for normal device gain while rejecting the near-zero
        // idle floor observed on an open but unconnected endpoint.
        signal_detected: input_peak >= 0.003,
        input_error: input_state.last_error(),
        output_error: output_state.last_error(),
    })
}

/// Play a mono PCM WAV through a selected native output endpoint. This helper
/// is used by the speech-loopback validator so fixture playback and the RVC
/// engine share the same CPAL/WASAPI path. It intentionally stays outside the
/// production engine and only supports the common PCM/IEEE-float WAV formats.
pub fn play_wav_fixture(
    input_path: &str,
    output_id: &str,
    seconds: f64,
    ready_file: Option<&str>,
) -> Result<FixturePlaybackResult, String> {
    if !seconds.is_finite() || !(0.1..=86_400.0).contains(&seconds) {
        return Err("Fixture playback duration must be between 0.1 and 86,400 seconds.".to_owned());
    }
    let (samples, source_sample_rate) = read_wav_mono(Path::new(input_path))?;
    let host = cpal::default_host();
    let (output_device, output_device_name) = find_device(&host, DeviceDirection::Output, output_id)?;
    let output_supported = output_device
        .default_output_config()
        .map_err(|error| format!("Could not read the fixture output format: {error}"))?;
    let output_sample_rate = output_supported.sample_rate();
    let samples = Arc::new(resample_fixture(&samples, source_sample_rate, output_sample_rate));
    if samples.is_empty() {
        return Err("The WAV fixture contains no audio samples.".to_owned());
    }

    let state = Arc::new(FixturePlaybackState {
        samples: Arc::clone(&samples),
        duration_frames: (seconds * f64::from(output_sample_rate)).round() as u64,
        frames: AtomicU64::new(0),
        last_error: Mutex::new(None),
    });
    let stream = build_fixture_stream(
        &output_device,
        output_supported.sample_format(),
        &output_supported.clone().into(),
        Arc::clone(&state),
    )?;
    stream
        .play()
        .map_err(|error| format!("Could not start the fixture output route: {error}"))?;
    if let Some(path) = ready_file {
        let marker = Path::new(path);
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Could not create fixture ready-marker folder: {error}"))?;
        }
        let marker_json = serde_json::json!({
            "input": input_path,
            "outputDevice": output_device_name.clone(),
            "sourceSampleRate": source_sample_rate,
            "outputSampleRate": output_sample_rate,
            "openedAt": format!("{:?}", Instant::now()),
        });
        fs::write(
            marker,
            serde_json::to_vec_pretty(&marker_json)
                .map_err(|error| format!("Could not encode fixture ready marker: {error}"))?,
        )
        .map_err(|error| format!("Could not write fixture ready marker: {error}"))?;
    }
    thread::sleep(Duration::from_secs_f64(seconds + 0.05));
    drop(stream);

    Ok(FixturePlaybackResult {
        input_path: input_path.to_owned(),
        output_device_name,
        requested_seconds: seconds,
        source_sample_rate,
        output_sample_rate,
        written_frames: state.frames.load(Ordering::Relaxed),
        peak: samples
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs())),
        output_error: state.last_error(),
    })
}

struct FixturePlaybackState {
    samples: Arc<Vec<f32>>,
    duration_frames: u64,
    frames: AtomicU64,
    last_error: Mutex<Option<String>>,
}

impl FixturePlaybackState {
    fn record_error(&self, error: String) {
        if let Ok(mut last_error) = self.last_error.lock() {
            *last_error = Some(error);
        }
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|error| error.clone())
    }
}

fn build_fixture_stream(
    device: &Device,
    format: SampleFormat,
    config: &StreamConfig,
    state: Arc<FixturePlaybackState>,
) -> Result<Stream, String> {
    match format {
        SampleFormat::F32 => build_fixture_stream_typed::<f32>(device, config, state, |sample| sample),
        SampleFormat::I16 => build_fixture_stream_typed::<i16>(
            device,
            config,
            state,
            |sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16,
        ),
        SampleFormat::U16 => build_fixture_stream_typed::<u16>(
            device,
            config,
            state,
            |sample| ((sample.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16,
        ),
        unsupported => Err(format!(
            "Output sample format {unsupported} is not supported by native fixture playback."
        )),
    }
}

fn build_fixture_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    state: Arc<FixturePlaybackState>,
    convert: fn(f32) -> T,
) -> Result<Stream, String>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    let channels = config.channels as usize;
    let error_state = Arc::clone(&state);
    device
        .build_output_stream(
            *config,
            move |data: &mut [T], _| {
                for frame in data.chunks_exact_mut(channels) {
                    let frame_index = state.frames.fetch_add(1, Ordering::Relaxed);
                    let sample = if frame_index < state.duration_frames {
                        state.samples[frame_index as usize % state.samples.len()]
                    } else {
                        0.0
                    };
                    frame.fill(convert(sample));
                }
            },
            move |error| error_state.record_error(format!("Fixture output error: {error}")),
            None,
        )
        .map_err(|error| format!("Could not create the fixture output stream: {error}"))
}

fn resample_fixture(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }
    let target_len = ((samples.len() as u64 * u64::from(target_rate))
        / u64::from(source_rate)) as usize;
    let target_len = target_len.max(1);
    let ratio = source_rate as f64 / target_rate as f64;
    (0..target_len)
        .map(|index| {
            let source_position = index as f64 * ratio;
            let left = source_position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = (source_position - left as f64) as f32;
            samples[left.min(samples.len() - 1)]
                + (samples[right] - samples[left.min(samples.len() - 1)]) * fraction
        })
        .collect()
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = fs::read(path).map_err(|error| format!("Could not read WAV fixture {}: {error}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("Fixture {} is not a RIFF/WAVE file.", path.display()));
    }
    let mut cursor = 12usize;
    let mut format_tag = None;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data = None;
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let start = cursor + 8;
        let end = start.saturating_add(size).min(bytes.len());
        if id == b"fmt " && end.saturating_sub(start) >= 16 {
            format_tag = Some(u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()));
            channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
        } else if id == b"data" {
            data = Some((start, end));
        }
        cursor = end + (size & 1);
    }
    let format_tag = format_tag.ok_or_else(|| "WAV fixture has no fmt chunk.".to_owned())?;
    let (data_start, data_end) = data.ok_or_else(|| "WAV fixture has no data chunk.".to_owned())?;
    if channels == 0 || sample_rate == 0 {
        return Err("WAV fixture has invalid channel or sample-rate metadata.".to_owned());
    }
    if !matches!(format_tag, 1 | 3) {
        return Err(format!(
            "WAV fixture format tag {format_tag} is unsupported; use PCM or IEEE float WAV."
        ));
    }
    let bytes_per_sample = usize::from(bits_per_sample / 8);
    if !matches!(bits_per_sample, 16 | 24 | 32) || bytes_per_sample == 0 {
        return Err(format!(
            "WAV fixture bit depth {bits_per_sample} is unsupported; use 16-, 24-, or 32-bit audio."
        ));
    }
    let frame_bytes = bytes_per_sample * usize::from(channels);
    if frame_bytes == 0 || data_end <= data_start || (data_end - data_start) < frame_bytes {
        return Err("WAV fixture contains no complete audio frames.".to_owned());
    }
    let frame_count = (data_end - data_start) / frame_bytes;
    let mut samples = Vec::with_capacity(frame_count);
    for frame in 0..frame_count {
        let frame_start = data_start + frame * frame_bytes;
        let mut sum = 0.0_f32;
        for channel in 0..usize::from(channels) {
            let start = frame_start + channel * bytes_per_sample;
            let value = match (format_tag, bits_per_sample) {
                (1, 16) => i16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()) as f32
                    / i16::MAX as f32,
                (1, 24) => {
                    let raw = i32::from(bytes[start])
                        | (i32::from(bytes[start + 1]) << 8)
                        | (i32::from(bytes[start + 2]) << 16);
                    let signed = if raw & 0x0080_0000 != 0 { raw | !0x00ff_ffff } else { raw };
                    signed as f32 / 8_388_607.0
                }
                (1, 32) => i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()) as f32
                    / i32::MAX as f32,
                (3, 32) => f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()),
                _ => unreachable!(),
            };
            sum += value;
        }
        samples.push((sum / f32::from(channels)).clamp(-1.0, 1.0));
    }
    Ok((samples, sample_rate))
}

fn wait_for_startup_prime(playback_ring: &ArrayQueue<f32>) {
    let deadline = Instant::now() + STARTUP_PRIME_TIMEOUT;
    while playback_ring.len() < STARTUP_PRIME_FRAMES && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(2));
    }
}

pub fn enumerate_devices() -> Result<AudioDeviceSnapshot, String> {
    let host = cpal::default_host();
    let default_input_id = host
        .default_input_device()
        .and_then(|device| device.id().ok())
        .map(|id| id.to_string());
    let default_output_id = host
        .default_output_device()
        .and_then(|device| device.id().ok())
        .map(|id| id.to_string());

    let inputs = collect_devices(&host, DeviceDirection::Input, default_input_id.as_deref())?;
    let outputs = collect_devices(&host, DeviceDirection::Output, default_output_id.as_deref())?;

    Ok(AudioDeviceSnapshot {
        inputs,
        outputs,
        default_input_id,
        default_output_id,
        backend: "WASAPI via CPAL",
        source: "native-probe",
    })
}

#[derive(Clone, Copy)]
enum DeviceDirection {
    Input,
    Output,
}

impl DeviceDirection {
    fn label(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }
}

fn collect_devices(
    host: &cpal::Host,
    direction: DeviceDirection,
    default_id: Option<&str>,
) -> Result<Vec<AudioDeviceInfo>, String> {
    let devices = match direction {
        DeviceDirection::Input => host.input_devices(),
        DeviceDirection::Output => host.output_devices(),
    }
    .map_err(|error| format!("Could not enumerate {} devices: {error}", direction.label()))?;

    let mut result = Vec::new();
    for (index, device) in devices.enumerate() {
        let name = device.to_string();
        let id = device_identifier(&device, direction, index);
        let config = match direction {
            DeviceDirection::Input => device.default_input_config(),
            DeviceDirection::Output => device.default_output_config(),
        };
        let Ok(config) = config else {
            continue;
        };

        result.push(AudioDeviceInfo {
            id: id.clone(),
            is_default: default_id == Some(id.as_str()),
            name,
            channels: config.channels(),
            sample_rate: config.sample_rate(),
            sample_format: config.sample_format().to_string(),
        });
    }

    Ok(result)
}

fn find_device(
    host: &cpal::Host,
    direction: DeviceDirection,
    requested_id: &str,
) -> Result<(Device, String), String> {
    let devices = match direction {
        DeviceDirection::Input => host.input_devices(),
        DeviceDirection::Output => host.output_devices(),
    }
    .map_err(|error| format!("Could not enumerate {} devices: {error}", direction.label()))?;

    for (index, device) in devices.enumerate() {
        let name = device.to_string();
        if device_identifier(&device, direction, index) == requested_id
            || device.to_string() == requested_id
        {
            return Ok((device, name));
        }
    }

    Err(format!(
        "The selected {} device is no longer available.",
        direction.label()
    ))
}

fn device_identifier(device: &Device, direction: DeviceDirection, index: usize) -> String {
    device
        .id()
        .map(|id| id.to_string())
        .unwrap_or_else(|_| format!("{}:{index}", direction.label()))
}

fn build_input_stream(
    device: &Device,
    format: SampleFormat,
    config: &StreamConfig,
    telemetry: Arc<AudioTelemetry>,
    processing: AudioProcessingSettings,
) -> Result<Stream, String> {
    match format {
        SampleFormat::F32 => {
            build_input_stream_typed::<f32>(device, config, telemetry, processing, |sample| sample)
        }
        SampleFormat::I16 => {
            build_input_stream_typed::<i16>(device, config, telemetry, processing, |sample| {
                sample as f32 / i16::MAX as f32
            })
        }
        SampleFormat::U16 => {
            build_input_stream_typed::<u16>(device, config, telemetry, processing, |sample| {
                sample as f32 / u16::MAX as f32 * 2.0 - 1.0
            })
        }
        unsupported => Err(format!(
            "Input sample format {unsupported} is not supported by the native audio engine."
        )),
    }
}

fn build_input_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    telemetry: Arc<AudioTelemetry>,
    processing: AudioProcessingSettings,
    convert: fn(T) -> f32,
) -> Result<Stream, String>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    let channels = config.channels as usize;
    let mut input_processor = InputProcessor::new(processing, config.sample_rate);
    let error_telemetry = Arc::clone(&telemetry);
    device
        .build_input_stream(
            *config,
            move |data: &[T], _| {
                let mut peak = 0.0_f32;
                for frame in data.chunks_exact(channels) {
                    let mono = frame.iter().copied().map(convert).sum::<f32>() / channels as f32;
                    // Consume one far-end sample for every captured sample. When the output
                    // device is unavailable or its callback is late, a missing reference is
                    // treated as silence instead of reusing stale audio.
                    let echo_reference = telemetry.echo_reference_ring.pop().unwrap_or(0.0);
                    let processed = input_processor.process(mono, echo_reference);
                    peak = peak.max(processed.abs());
                    if telemetry.capture_ring.push(processed).is_err() {
                        telemetry.overruns.fetch_add(1, Ordering::Relaxed);
                    }
                    telemetry.captured_frames.fetch_add(1, Ordering::Relaxed);
                }
                telemetry
                    .input_peak_bits
                    .store(peak.to_bits(), Ordering::Relaxed);
            },
            move |error| error_telemetry.record_error(format!("Input stream error: {error}")),
            None,
        )
        .map_err(|error| format!("Could not create the input stream: {error}"))
}

#[derive(Clone, Copy, Default)]
struct StableRead {
    sample: f32,
    primed: bool,
    underrun: bool,
    reprime: bool,
    dropped: bool,
    repeated: bool,
    target_changed: bool,
}

struct AdaptivePlayback {
    sample_rate: u32,
    target_frames: usize,
    primed: bool,
    frames_until_correction: usize,
    stable_frames: u64,
    last_sample: f32,
    recovery_remaining_frames: usize,
    underrun_active: bool,
}

/// Converts the fixed 48 kHz inference stream to an output/monitor device rate. It uses
/// one-sample linear interpolation with persistent phase so 44.1/48/96 kHz endpoints do
/// not require a hard rejection or a discontinuous per-callback ratio reset.
struct OutputResampler {
    source_rate: u32,
    target_rate: u32,
    step: f64,
    phase: f64,
    current: f32,
    next: f32,
    initialized: bool,
}

impl OutputResampler {
    fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            source_rate,
            target_rate,
            step: source_rate as f64 / target_rate as f64,
            phase: 0.0,
            current: 0.0,
            next: 0.0,
            initialized: false,
        }
    }

    fn read(&mut self, stability: &mut AdaptivePlayback, ring: &ArrayQueue<f32>) -> StableRead {
        if self.source_rate == self.target_rate {
            return stability.read(ring);
        }

        if !self.initialized {
            let first = stability.read(ring);
            if !first.primed {
                return first;
            }
            let second = stability.read(ring);
            let mut startup = first;
            merge_stable_read(&mut startup, &second);
            if !second.primed {
                return startup;
            }
            self.current = first.sample;
            self.next = second.sample;
            self.phase = 0.0;
            self.initialized = true;
        }

        let fraction = self.phase as f32;
        let mut result = StableRead {
            sample: self.current + (self.next - self.current) * fraction,
            primed: true,
            ..StableRead::default()
        };
        self.phase += self.step;
        while self.phase >= 1.0 {
            self.phase -= 1.0;
            self.current = self.next;
            let next = stability.read(ring);
            merge_stable_read(&mut result, &next);
            self.next = next.sample;
            if !next.primed {
                self.phase = 0.0;
                self.initialized = false;
                break;
            }
        }
        result
    }
}

fn merge_stable_read(target: &mut StableRead, next: &StableRead) {
    target.primed &= next.primed;
    target.underrun |= next.underrun;
    target.reprime |= next.reprime;
    target.dropped |= next.dropped;
    target.repeated |= next.repeated;
    target.target_changed |= next.target_changed;
}

impl AdaptivePlayback {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            target_frames: INITIAL_PRIME_FRAMES,
            primed: false,
            frames_until_correction: (sample_rate as usize / 20).max(1),
            stable_frames: 0,
            last_sample: 0.0,
            recovery_remaining_frames: 0,
            underrun_active: false,
        }
    }

    fn read(&mut self, ring: &ArrayQueue<f32>) -> StableRead {
        if !self.primed {
            if ring.len() < self.target_frames {
                return StableRead::default();
            }
            self.primed = true;
        }

        self.frames_until_correction = self.frames_until_correction.saturating_sub(1);
        let can_correct = self.frames_until_correction == 0;
        if can_correct {
            self.frames_until_correction = (self.sample_rate as usize / 20).max(1);
        }

        let mut result = StableRead {
            primed: true,
            ..StableRead::default()
        };
        let queue_depth = ring.len();
        if can_correct && queue_depth > self.target_frames.saturating_add(DRIFT_HIGH_MARGIN_FRAMES)
        {
            let first = ring.pop().unwrap_or(self.last_sample);
            let second = ring.pop().unwrap_or(first);
            result.sample = (first + second) * 0.5;
            result.dropped = true;
        } else if can_correct && queue_depth > 0 && queue_depth < DRIFT_LOW_WATERMARK_FRAMES {
            result.sample = self.last_sample;
            result.repeated = true;
        } else if let Some(sample) = ring.pop() {
            result.sample = sample;
            self.recovery_remaining_frames = 0;
            self.underrun_active = false;
        } else {
            self.stable_frames = 0;
            let first_empty_frame = !self.underrun_active;
            self.underrun_active = true;
            if first_empty_frame {
                let previous_target = self.target_frames;
                self.target_frames = self
                    .target_frames
                    .saturating_add(WORKER_CHUNK_FRAMES)
                    .min(MAX_PRIME_FRAMES);
                result.underrun = true;
                result.reprime = true;
                result.target_changed = self.target_frames != previous_target;
                self.recovery_remaining_frames = (self.sample_rate as usize / 100).max(1);
            }

            // A short queue starvation should not turn into a long hard gap.
            // Fade the last sample toward silence for roughly 10 ms, then hold
            // silence until the worker refills the queue. The first empty frame
            // remains visible in telemetry, while repeated empty callbacks are
            // counted as one recovery episode rather than thousands of events.
            if self.recovery_remaining_frames > 0 {
                let holdover_frames = (self.sample_rate as usize / 100).max(1);
                let recovery_gain =
                    self.recovery_remaining_frames as f32 / holdover_frames as f32;
                result.sample = self.last_sample * recovery_gain;
                self.recovery_remaining_frames =
                    self.recovery_remaining_frames.saturating_sub(1);
            } else {
                result.sample = 0.0;
            }
            result.primed = true;
        }

        self.last_sample = result.sample;
        self.stable_frames = self.stable_frames.saturating_add(1);
        let stable_reduction_frames = u64::from(self.sample_rate) * 30;
        if self.stable_frames >= stable_reduction_frames
            && self.target_frames > INITIAL_PRIME_FRAMES
        {
            self.stable_frames = 0;
            self.target_frames = self
                .target_frames
                .saturating_sub(WORKER_CHUNK_FRAMES)
                .max(INITIAL_PRIME_FRAMES);
            result.target_changed = true;
        }
        result
    }
}

fn build_output_stream(
    device: &Device,
    format: SampleFormat,
    config: &StreamConfig,
    telemetry: Arc<AudioTelemetry>,
    processing: AudioProcessingSettings,
) -> Result<Stream, String> {
    match format {
        SampleFormat::F32 => {
            build_output_stream_typed::<f32>(device, config, telemetry, processing, |sample| sample)
        }
        SampleFormat::I16 => {
            build_output_stream_typed::<i16>(device, config, telemetry, processing, |sample| {
                (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
            })
        }
        SampleFormat::U16 => {
            build_output_stream_typed::<u16>(device, config, telemetry, processing, |sample| {
                ((sample.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16
            })
        }
        unsupported => Err(format!(
            "Output sample format {unsupported} is not supported by the native audio engine."
        )),
    }
}

struct ToneTelemetry {
    duration_frames: u64,
    frames: AtomicU64,
    peak_bits: AtomicU32,
    last_error: Mutex<Option<String>>,
}

impl ToneTelemetry {
    fn new(duration_ms: u32, sample_rate: u32) -> Self {
        Self {
            duration_frames: (u64::from(duration_ms) * u64::from(sample_rate)) / 1_000,
            frames: AtomicU64::new(0),
            peak_bits: AtomicU32::new(0),
            last_error: Mutex::new(None),
        }
    }

    fn record_error(&self, error: String) {
        if let Ok(mut last_error) = self.last_error.lock() {
            *last_error = Some(error);
        }
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|error| error.clone())
    }

    fn record_peak(&self, peak: f32) {
        let mut current = self.peak_bits.load(Ordering::Relaxed);
        loop {
            let current_peak = f32::from_bits(current);
            if peak <= current_peak {
                break;
            }
            match self.peak_bits.compare_exchange_weak(
                current,
                peak.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }
}

#[derive(Default)]
struct InputProbeTelemetry {
    frames: AtomicU64,
    peak_bits: AtomicU32,
    last_error: Mutex<Option<String>>,
}

impl InputProbeTelemetry {
    fn record_error(&self, error: String) {
        if let Ok(mut last_error) = self.last_error.lock() {
            *last_error = Some(error);
        }
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|error| error.clone())
    }

    fn record_peak(&self, peak: f32) {
        let mut current = self.peak_bits.load(Ordering::Relaxed);
        loop {
            let current_peak = f32::from_bits(current);
            if peak <= current_peak {
                break;
            }
            match self.peak_bits.compare_exchange_weak(
                current,
                peak.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }
}

fn build_input_probe_stream(
    device: &Device,
    format: SampleFormat,
    config: &StreamConfig,
    telemetry: Arc<InputProbeTelemetry>,
) -> Result<Stream, String> {
    match format {
        SampleFormat::F32 => {
            build_input_probe_stream_typed::<f32>(device, config, telemetry, |sample| sample)
        }
        SampleFormat::I16 => build_input_probe_stream_typed::<i16>(
            device,
            config,
            telemetry,
            |sample| sample as f32 / i16::MAX as f32,
        ),
        SampleFormat::U16 => build_input_probe_stream_typed::<u16>(
            device,
            config,
            telemetry,
            |sample| sample as f32 / u16::MAX as f32 * 2.0 - 1.0,
        ),
        unsupported => Err(format!(
            "Input sample format {unsupported} is not supported by the loopback test."
        )),
    }
}

fn build_input_probe_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    telemetry: Arc<InputProbeTelemetry>,
    convert: fn(T) -> f32,
) -> Result<Stream, String>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    let channels = config.channels as usize;
    let error_telemetry = Arc::clone(&telemetry);
    device
        .build_input_stream(
            *config,
            move |data: &[T], _| {
                let mut block_peak = 0.0_f32;
                for frame in data.chunks_exact(channels) {
                    let mono = frame.iter().copied().map(convert).sum::<f32>() / channels as f32;
                    block_peak = block_peak.max(mono.abs());
                    telemetry.frames.fetch_add(1, Ordering::Relaxed);
                }
                telemetry.record_peak(block_peak);
            },
            move |error| error_telemetry.record_error(format!("Input loopback error: {error}")),
            None,
        )
        .map_err(|error| format!("Could not create the loopback input stream: {error}"))
}

fn build_tone_stream(
    device: &Device,
    format: SampleFormat,
    config: &StreamConfig,
    telemetry: Arc<ToneTelemetry>,
) -> Result<Stream, String> {
    match format {
        SampleFormat::F32 => {
            build_tone_stream_typed::<f32>(device, config, telemetry, |sample| sample)
        }
        SampleFormat::I16 => build_tone_stream_typed::<i16>(device, config, telemetry, |sample| {
            (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
        }),
        SampleFormat::U16 => build_tone_stream_typed::<u16>(device, config, telemetry, |sample| {
            ((sample.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16
        }),
        unsupported => Err(format!(
            "Output sample format {unsupported} is not supported by the route test."
        )),
    }
}

fn build_tone_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    telemetry: Arc<ToneTelemetry>,
    convert: fn(f32) -> T,
) -> Result<Stream, String>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.max(1) as f32;
    let fade_frames = (config.sample_rate / 100).max(1) as u64;
    let error_telemetry = Arc::clone(&telemetry);
    device
        .build_output_stream(
            *config,
            move |data: &mut [T], _| {
                let mut block_peak = 0.0_f32;
                for frame in data.chunks_exact_mut(channels) {
                    let frame_index = telemetry.frames.fetch_add(1, Ordering::Relaxed);
                    let position = frame_index as f32 / sample_rate;
                    let mut envelope = 1.0_f32;
                    if frame_index < fade_frames {
                        envelope = frame_index as f32 / fade_frames as f32;
                    } else if frame_index + fade_frames > telemetry.duration_frames {
                        envelope = telemetry.duration_frames.saturating_sub(frame_index) as f32
                            / fade_frames as f32;
                    }
                    let sample = (0.08
                        * envelope.clamp(0.0, 1.0)
                        * (std::f32::consts::TAU * 440.0 * position).sin())
                    .clamp(-1.0, 1.0);
                    block_peak = block_peak.max(sample.abs());
                    let converted = convert(sample);
                    frame.fill(converted);
                }
                telemetry.record_peak(block_peak);
            },
            move |error| error_telemetry.record_error(format!("Output route test error: {error}")),
            None,
        )
        .map_err(|error| format!("Could not create the route test stream: {error}"))
}

fn build_output_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    telemetry: Arc<AudioTelemetry>,
    processing: AudioProcessingSettings,
    convert: fn(f32) -> T,
) -> Result<Stream, String>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    let channels = config.channels as usize;
    let output_gain = db_to_linear(processing.output_gain_db);
    let error_telemetry = Arc::clone(&telemetry);
    let mut stability = AdaptivePlayback::new(config.sample_rate);
    let mut resampler = OutputResampler::new(INFERENCE_SAMPLE_RATE, config.sample_rate);
    device
        .build_output_stream(
            *config,
            move |data: &mut [T], _| {
                let mut peak = 0.0_f32;
                for frame in data.chunks_exact_mut(channels) {
                    let stable = resampler.read(&mut stability, &telemetry.playback_ring);
                    telemetry.primed.store(stable.primed, Ordering::Release);
                    if stable.underrun {
                        telemetry.underruns.fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.reprime {
                        telemetry.reprimes.fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.dropped {
                        telemetry
                            .drift_dropped_frames
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.repeated {
                        telemetry
                            .drift_repeated_frames
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.target_changed {
                        telemetry
                            .prime_target_frames
                            .store(stability.target_frames, Ordering::Relaxed);
                    }
                    let sample = stable.sample;
                    if telemetry.monitor_enabled.load(Ordering::Acquire) {
                        if let Some(monitor_ring) = &telemetry.monitor_ring {
                            if monitor_ring.push(sample).is_err() {
                                let _ = monitor_ring.pop();
                                let _ = monitor_ring.push(sample);
                                telemetry.monitor_overruns.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    let sample = (sample * output_gain).clamp(-1.0, 1.0);
                    // Feed the actual post-gain signal to the input preprocessor. This is the
                    // far-end reference used by echo control, not the optional monitor route.
                    if telemetry.echo_reference_ring.push(sample).is_err() {
                        let _ = telemetry.echo_reference_ring.pop();
                        let _ = telemetry.echo_reference_ring.push(sample);
                    }
                    peak = peak.max(sample.abs());
                    let converted = convert(sample);
                    frame.fill(converted);
                    telemetry.played_frames.fetch_add(1, Ordering::Relaxed);
                }
                telemetry
                    .output_peak_bits
                    .store(peak.to_bits(), Ordering::Relaxed);
            },
            move |error| error_telemetry.record_error(format!("Output stream error: {error}")),
            None,
        )
        .map_err(|error| format!("Could not create the output stream: {error}"))
}

fn build_monitor_stream(
    device: &Device,
    format: SampleFormat,
    config: &StreamConfig,
    telemetry: Arc<AudioTelemetry>,
    processing: AudioProcessingSettings,
) -> Result<Stream, String> {
    match format {
        SampleFormat::F32 => {
            build_monitor_stream_typed::<f32>(device, config, telemetry, processing, |sample| {
                sample
            })
        }
        SampleFormat::I16 => {
            build_monitor_stream_typed::<i16>(device, config, telemetry, processing, |sample| {
                (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
            })
        }
        SampleFormat::U16 => {
            build_monitor_stream_typed::<u16>(device, config, telemetry, processing, |sample| {
                ((sample.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16
            })
        }
        unsupported => Err(format!(
            "Monitor sample format {unsupported} is not supported by the native audio engine."
        )),
    }
}

fn build_monitor_stream_typed<T>(
    device: &Device,
    config: &StreamConfig,
    telemetry: Arc<AudioTelemetry>,
    processing: AudioProcessingSettings,
    convert: fn(f32) -> T,
) -> Result<Stream, String>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    let monitor_ring = telemetry
        .monitor_ring
        .as_ref()
        .cloned()
        .ok_or_else(|| "The monitor stream has no audio buffer.".to_owned())?;
    let channels = config.channels as usize;
    let monitor_gain = db_to_linear(processing.monitor_gain_db);
    let error_telemetry = Arc::clone(&telemetry);
    let mut stability = AdaptivePlayback::new(config.sample_rate);
    let mut resampler = OutputResampler::new(INFERENCE_SAMPLE_RATE, config.sample_rate);
    device
        .build_output_stream(
            *config,
            move |data: &mut [T], _| {
                let mut peak = 0.0_f32;
                for frame in data.chunks_exact_mut(channels) {
                    let stable = resampler.read(&mut stability, &monitor_ring);
                    telemetry
                        .monitor_primed
                        .store(stable.primed, Ordering::Release);
                    if stable.underrun {
                        telemetry.monitor_underruns.fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.reprime {
                        telemetry.monitor_reprimes.fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.dropped {
                        telemetry
                            .monitor_drift_dropped_frames
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.repeated {
                        telemetry
                            .monitor_drift_repeated_frames
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    if stable.target_changed {
                        telemetry
                            .monitor_prime_target_frames
                            .store(stability.target_frames, Ordering::Relaxed);
                    }
                    let sample = stable.sample;
                    let sample = (sample * monitor_gain).clamp(-1.0, 1.0);
                    peak = peak.max(sample.abs());
                    frame.fill(convert(sample));
                    telemetry
                        .monitor_played_frames
                        .fetch_add(1, Ordering::Relaxed);
                }
                telemetry
                    .monitor_peak_bits
                    .store(peak.to_bits(), Ordering::Relaxed);
            },
            move |error| error_telemetry.record_error(format!("Monitor stream error: {error}")),
            None,
        )
        .map_err(|error| format!("Could not create the monitor stream: {error}"))
}

fn validate_db(label: &str, value: f32, minimum: f32, maximum: f32) -> Result<(), String> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "{label} must be between {minimum:.0} dB and {maximum:.0} dB."
        ));
    }
    Ok(())
}

fn validate_strength(label: &str, value: f32) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("{label} must be between 0% and 100%."));
    }
    Ok(())
}

fn db_to_linear(value: f32) -> f32 {
    10.0_f32.powf(value / 20.0)
}

struct InputProcessor {
    input_gain: f32,
    noise_suppressor: NoiseSuppressor,
    echo_canceller: EchoCanceller,
    high_pass_enabled: bool,
    highpass_previous_input: f32,
    highpass_previous_output: f32,
    gate_threshold: f32,
    envelope_release: f32,
    gate_attack: f32,
    gate_release: f32,
    envelope: f32,
    gate_gain: f32,
}

impl InputProcessor {
    fn new(settings: AudioProcessingSettings, sample_rate: u32) -> Self {
        let gate_enabled = settings.noise_gate_db > -80.0;
        Self {
            input_gain: db_to_linear(settings.input_gain_db),
            noise_suppressor: NoiseSuppressor::new(
                settings.noise_suppression_strength,
                sample_rate,
            ),
            echo_canceller: EchoCanceller::new(settings.echo_control_strength),
            high_pass_enabled: settings.high_pass_enabled,
            highpass_previous_input: 0.0,
            highpass_previous_output: 0.0,
            gate_threshold: if gate_enabled {
                db_to_linear(settings.noise_gate_db)
            } else {
                0.0
            },
            envelope_release: smoothing_coefficient(sample_rate, 0.030),
            gate_attack: smoothing_coefficient(sample_rate, 0.005),
            gate_release: smoothing_coefficient(sample_rate, 0.080),
            envelope: 0.0,
            gate_gain: if gate_enabled { 0.0 } else { 1.0 },
        }
    }

    fn process(&mut self, sample: f32, echo_reference: f32) -> f32 {
        let amplified = sample * self.input_gain;
        // w-okada's default RVC route leaves its high-pass filter disabled. Keep the
        // compatibility path neutral unless the user explicitly enables rumble cleanup.
        let highpassed = if self.high_pass_enabled {
            // The fixed coefficient is a gentle ~30 Hz one-pole high-pass at the
            // 48 kHz native contract.
            let filtered =
                amplified - self.highpass_previous_input + 0.995 * self.highpass_previous_output;
            self.highpass_previous_input = amplified;
            self.highpass_previous_output = filtered;
            filtered
        } else {
            amplified
        };
        let echo_reduced = self.echo_canceller.process(highpassed, echo_reference);
        let denoised = self.noise_suppressor.process(echo_reduced);
        self.envelope = denoised
            .abs()
            .max(self.envelope + (0.0 - self.envelope) * self.envelope_release);
        let target = if self.gate_threshold == 0.0 || self.envelope >= self.gate_threshold {
            1.0
        } else {
            0.0
        };
        let coefficient = if target > self.gate_gain {
            self.gate_attack
        } else {
            self.gate_release
        };
        self.gate_gain += (target - self.gate_gain) * coefficient;
        (denoised * self.gate_gain).clamp(-1.0, 1.0)
    }
}

/// A conservative, callback-safe downward expander. It tracks the quietest recent input
/// instead of making a hard cut, which keeps breaths and consonants available to the pitch
/// extractor while reducing stationary fan/room noise before RVC sees the signal.
struct NoiseSuppressor {
    strength: f32,
    noise_floor: f32,
    gain: f32,
    floor_rise: f32,
    floor_fall: f32,
    gain_attack: f32,
    gain_release: f32,
}

impl NoiseSuppressor {
    fn new(strength: f32, sample_rate: u32) -> Self {
        Self {
            strength,
            // A small but non-zero bootstrap floor lets the estimator learn common mic-room
            // noise (roughly -42 dBFS) without treating the first speech sample as noise.
            noise_floor: 0.008,
            gain: 1.0,
            floor_rise: smoothing_coefficient(sample_rate, 0.250),
            floor_fall: smoothing_coefficient(sample_rate, 0.050),
            gain_attack: smoothing_coefficient(sample_rate, 0.008),
            gain_release: smoothing_coefficient(sample_rate, 0.045),
        }
    }

    fn process(&mut self, sample: f32) -> f32 {
        if self.strength <= 0.0 {
            return sample;
        }

        let magnitude = sample.abs();
        let tracking_ceiling = self.noise_floor * (2.25 + self.strength * 2.0) + 0.0005;
        if magnitude <= tracking_ceiling {
            let coefficient = if magnitude < self.noise_floor {
                self.floor_fall
            } else {
                self.floor_rise
            };
            self.noise_floor += (magnitude - self.noise_floor) * coefficient;
        }

        let threshold = self.noise_floor * (2.0 + self.strength * 4.0) + 0.0004;
        let target_gain = if magnitude < threshold {
            let ratio = (magnitude / threshold.max(1e-5)).clamp(0.0, 1.0);
            // At 100% strength, a sample at the adaptive floor is reduced to about 10%.
            (1.0 - self.strength * (1.0 - ratio)).clamp(0.08, 1.0)
        } else {
            1.0
        };
        let coefficient = if target_gain > self.gain {
            self.gain_attack
        } else {
            self.gain_release
        };
        self.gain += (target_gain - self.gain) * coefficient;
        sample * self.gain
    }
}

/// Lightweight NLMS echo control using the post-gain output callback as the far-end
/// reference. It is intentionally conservative and bounded; a full acoustic echo canceller
/// still needs device-specific delay calibration and room modelling.
struct EchoCanceller {
    strength: f32,
    history: [f32; ECHO_HISTORY_FRAMES],
    coefficients: [f32; ECHO_TAPS],
    write_index: usize,
}

impl EchoCanceller {
    fn new(strength: f32) -> Self {
        Self {
            strength,
            history: [0.0; ECHO_HISTORY_FRAMES],
            coefficients: [0.0; ECHO_TAPS],
            write_index: 0,
        }
    }

    fn process(&mut self, microphone: f32, reference: f32) -> f32 {
        self.history[self.write_index] = reference;
        let delayed_index =
            (self.write_index + ECHO_HISTORY_FRAMES - ECHO_DELAY_FRAMES) % ECHO_HISTORY_FRAMES;
        self.write_index = (self.write_index + 1) % ECHO_HISTORY_FRAMES;

        if self.strength <= 0.0 {
            return microphone;
        }

        let mut predicted = 0.0;
        let mut energy = 1e-5;
        for tap in 0..ECHO_TAPS {
            let index = (delayed_index + ECHO_HISTORY_FRAMES - tap) % ECHO_HISTORY_FRAMES;
            let reference_sample = self.history[index];
            predicted += self.coefficients[tap] * reference_sample;
            energy += reference_sample * reference_sample;
        }

        let error = microphone - predicted;
        let adaptation = 0.08 / energy;
        // Do not let an unexpectedly loud near-end voice train the filter aggressively.
        let adaptation_scale = if microphone.abs() > predicted.abs() * 2.0 + 0.04 {
            0.2
        } else {
            1.0
        };
        for tap in 0..ECHO_TAPS {
            let index = (delayed_index + ECHO_HISTORY_FRAMES - tap) % ECHO_HISTORY_FRAMES;
            self.coefficients[tap] = (self.coefficients[tap]
                + adaptation * adaptation_scale * error * self.history[index])
                .clamp(-2.0, 2.0);
        }

        let correction =
            (predicted * self.strength).clamp(-(microphone.abs() + 0.25), microphone.abs() + 0.25);
        microphone - correction
    }
}

fn smoothing_coefficient(sample_rate: u32, seconds: f32) -> f32 {
    1.0 - (-1.0 / (sample_rate as f32 * seconds)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_device_ids_are_directional_and_repeatable() {
        let input = format!("{}:{}", DeviceDirection::Input.label(), 2);
        let output = format!("{}:{}", DeviceDirection::Output.label(), 2);
        assert_eq!(input, format!("{}:{}", DeviceDirection::Input.label(), 2));
        assert_ne!(input, output);
    }

    #[test]
    fn stopped_status_is_explicit() {
        let status = AudioEngine::default().status();
        assert_eq!(status.state, "stopped");
        assert_eq!(status.buffered_frames, 0);
        assert_eq!(status.noise_gate_db, -80.0);
    }

    #[test]
    fn audio_processing_settings_reject_invalid_ranges() {
        assert!(AudioProcessingSettings::new(25.0, 0.0, -6.0, -60.0, 0.3, 0.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 13.0, -6.0, -60.0, 0.3, 0.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 0.0, 13.0, -60.0, 0.3, 0.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 0.0, -6.0, -19.0, 0.3, 0.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 0.0, -6.0, -80.0, 1.1, 0.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 0.0, -6.0, -80.0, 0.0, -0.1).is_err());
        assert!(AudioProcessingSettings::new(6.0, -3.0, -8.0, -55.0, 0.35, 0.5).is_ok());
    }

    #[test]
    fn noise_gate_at_minimum_is_transparent() {
        let mut processor = InputProcessor::new(AudioProcessingSettings::default(), 48_000);
        assert!((processor.process(0.25, 0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn input_gain_is_applied_before_capture() {
        let settings = AudioProcessingSettings::new(6.0, 0.0, -6.0, -80.0, 0.0, 0.0).unwrap();
        let mut processor = InputProcessor::new(settings, 48_000);
        assert!((processor.process(0.25, 0.0) - 0.4988).abs() < 0.002);
    }

    #[test]
    fn input_preprocessor_limits_gain_staging_to_float_audio_bounds() {
        let settings = AudioProcessingSettings::new(24.0, 0.0, -6.0, -80.0, 0.0, 0.0).unwrap();
        let mut processor = InputProcessor::new(settings, 48_000);
        assert!(processor.process(1.0, 0.0).abs() <= 1.0);
    }

    #[test]
    fn high_pass_is_opt_in_for_w_okada_compatible_defaults() {
        let neutral = AudioProcessingSettings::default();
        let mut neutral_processor = InputProcessor::new(neutral, 48_000);
        let neutral_first = neutral_processor.process(0.25, 0.0);
        let neutral_second = neutral_processor.process(0.25, 0.0);
        assert!((neutral_first - 0.25).abs() < 1e-6);
        assert!((neutral_second - 0.25).abs() < 1e-6);

        let filtered = AudioProcessingSettings::default().with_high_pass(true);
        let mut filtered_processor = InputProcessor::new(filtered, 48_000);
        let filtered_first = filtered_processor.process(0.25, 0.0);
        let filtered_second = filtered_processor.process(0.25, 0.0);
        assert!((filtered_first - 0.25).abs() < 1e-6);
        assert!(filtered_second < filtered_first);
    }

    #[test]
    fn noise_suppression_reduces_stationary_noise_and_keeps_speech() {
        let settings = AudioProcessingSettings::new(0.0, 0.0, -6.0, -80.0, 1.0, 0.0).unwrap();
        let mut processor = InputProcessor::new(settings, 48_000);
        let mut quiet = 0.0;
        for _ in 0..48_000 {
            quiet = processor.process(0.01, 0.0);
        }
        assert!(
            quiet.abs() < 0.005,
            "stationary noise was not reduced: {quiet}"
        );

        let mut speech = 0.0;
        for frame in 0..2_000 {
            speech = processor.process(if frame % 2 == 0 { 0.35 } else { -0.35 }, 0.0);
        }
        assert!(
            speech.abs() > 0.2,
            "speech transient was over-suppressed: {speech}"
        );
    }

    #[test]
    fn echo_control_learns_a_delayed_output_reference() {
        let mut canceller = EchoCanceller::new(1.0);
        let mut residual_sum = 0.0;
        let mut source_sum = 0.0;
        for frame in 0..96_000 {
            let reference = (frame as f32 * 0.021).sin() * 0.12;
            let microphone = if frame >= ECHO_DELAY_FRAMES {
                ((frame - ECHO_DELAY_FRAMES) as f32 * 0.021).sin() * 0.12
            } else {
                0.0
            };
            let output = canceller.process(microphone, reference);
            if frame > 48_000 {
                residual_sum += output * output;
                source_sum += microphone * microphone;
            }
        }
        assert!(
            residual_sum < source_sum * 0.35,
            "echo residual remained too high"
        );
    }

    #[test]
    fn adaptive_playback_primes_only_at_its_target_depth() {
        let ring = ArrayQueue::new(INITIAL_PRIME_FRAMES * 2);
        for _ in 0..INITIAL_PRIME_FRAMES - 1 {
            ring.push(0.25).unwrap();
        }
        let mut playback = AdaptivePlayback::new(48_000);
        let waiting = playback.read(&ring);
        assert!(!waiting.primed);
        assert_eq!(waiting.sample, 0.0);

        ring.push(0.25).unwrap();
        let started = playback.read(&ring);
        assert!(started.primed);
        assert_eq!(started.sample, 0.25);
    }

    #[test]
    fn adaptive_playback_reprime_increases_the_safety_depth() {
        let ring = ArrayQueue::new(INITIAL_PRIME_FRAMES * 2);
        for _ in 0..INITIAL_PRIME_FRAMES {
            ring.push(0.1).unwrap();
        }
        let mut playback = AdaptivePlayback::new(48_000);
        assert!(playback.read(&ring).primed);
        while ring.pop().is_some() {}

        let underrun = playback.read(&ring);
        assert!(underrun.underrun);
        assert!(underrun.reprime);
        assert!(underrun.primed);
        let continuation = playback.read(&ring);
        assert!(!continuation.underrun);
        assert!(continuation.primed);
        assert_eq!(
            playback.target_frames,
            INITIAL_PRIME_FRAMES + WORKER_CHUNK_FRAMES
        );
    }

    #[test]
    fn adaptive_playback_bounds_clock_drift_with_sample_slips() {
        let ring = ArrayQueue::new(INITIAL_PRIME_FRAMES + DRIFT_HIGH_MARGIN_FRAMES + 8);
        for _ in 0..INITIAL_PRIME_FRAMES + DRIFT_HIGH_MARGIN_FRAMES + 1 {
            ring.push(0.2).unwrap();
        }
        let mut playback = AdaptivePlayback::new(48_000);
        playback.frames_until_correction = 0;
        let high = playback.read(&ring);
        assert!(high.dropped);

        while ring.len() > 1 {
            ring.pop();
        }
        playback.primed = true;
        playback.last_sample = 0.3;
        playback.frames_until_correction = 0;
        let low = playback.read(&ring);
        assert!(low.repeated);
        assert_eq!(low.sample, 0.3);
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn output_resampler_interpolates_between_fixed_rate_samples() {
        let ring = ArrayQueue::new(RING_BUFFER_FRAMES);
        for index in 0..(INITIAL_PRIME_FRAMES + 8) {
            ring.push(index as f32).unwrap();
        }
        let mut playback = AdaptivePlayback::new(48_000);
        let mut resampler = OutputResampler::new(48_000, 96_000);
        let first = resampler.read(&mut playback, &ring);
        let second = resampler.read(&mut playback, &ring);
        assert!(first.primed && second.primed);
        assert!((first.sample - 0.0).abs() < 1e-6);
        assert!((second.sample - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reads_pcm_wav_fixtures_and_resamples_them() {
        let path = std::env::temp_dir().join(format!(
            "vc-next-audio-fixture-{}.wav",
            std::process::id()
        ));
        let samples = [-32_768_i16, 0, 32_767, 0];
        let mut data = Vec::new();
        for sample in samples {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36_u32 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&48_000_u32.to_le_bytes());
        wav.extend_from_slice(&96_000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        std::fs::write(&path, wav).expect("test WAV should be writable");

        let (decoded, rate) = read_wav_mono(&path).expect("test WAV should decode");
        assert_eq!(rate, 48_000);
        assert_eq!(decoded.len(), 4);
        assert!((decoded[0] + 1.0).abs() < 1e-5);
        assert!(decoded[2] > 0.99);
        let resampled = resample_fixture(&decoded, rate, 24_000);
        assert_eq!(resampled.len(), 2);
        assert!(resampled.iter().all(|sample| sample.is_finite()));
        let _ = std::fs::remove_file(path);
    }
}

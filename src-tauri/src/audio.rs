use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SampleFormat, Stream, StreamConfig,
};
use crossbeam_queue::ArrayQueue;
use serde::Serialize;

use crate::inference::{
    InferenceBackend, InferenceTelemetry, InferenceWorker, NoopInferenceBackend,
    WORKER_CHUNK_FRAMES,
};
use crate::live_sidecar::{LiveRvcClient, LiveRvcInferenceBackend};

const RING_BUFFER_FRAMES: usize = 96_000;
const INITIAL_PRIME_FRAMES: usize = WORKER_CHUNK_FRAMES * 2;
const MAX_PRIME_FRAMES: usize = WORKER_CHUNK_FRAMES * 10;
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
    input_peak: f32,
    output_peak: f32,
    monitor_peak: f32,
    input_gain_db: f32,
    output_gain_db: f32,
    monitor_gain_db: f32,
    noise_gate_db: f32,
    last_error: Option<String>,
}

#[derive(Clone, Copy)]
pub struct AudioProcessingSettings {
    input_gain_db: f32,
    output_gain_db: f32,
    monitor_gain_db: f32,
    noise_gate_db: f32,
}

impl AudioProcessingSettings {
    pub fn new(
        input_gain_db: f32,
        output_gain_db: f32,
        monitor_gain_db: f32,
        noise_gate_db: f32,
    ) -> Result<Self, String> {
        validate_db("Input gain", input_gain_db, -24.0, 24.0)?;
        validate_db("Output gain", output_gain_db, -24.0, 12.0)?;
        validate_db("Monitor gain", monitor_gain_db, -24.0, 12.0)?;
        validate_db("Noise gate", noise_gate_db, -80.0, -20.0)?;
        Ok(Self {
            input_gain_db,
            output_gain_db,
            monitor_gain_db,
            noise_gate_db,
        })
    }
}

impl Default for AudioProcessingSettings {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            monitor_gain_db: -6.0,
            noise_gate_db: -80.0,
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
    state: &'static str,
    processing: AudioProcessingSettings,
    telemetry: Arc<AudioTelemetry>,
    inference_worker: InferenceWorker,
}

struct AudioTelemetry {
    capture_ring: Arc<ArrayQueue<f32>>,
    playback_ring: Arc<ArrayQueue<f32>>,
    monitor_ring: Option<Arc<ArrayQueue<f32>>>,
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
        inference: Arc<InferenceTelemetry>,
    ) -> Self {
        Self {
            capture_ring,
            playback_ring,
            monitor_ring,
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
        let monitor_device = monitor_id
            .map(|id| find_device(&host, DeviceDirection::Output, id))
            .transpose()?;

        let input_supported = input_device
            .default_input_config()
            .map_err(|error| format!("Could not read the input format: {error}"))?;
        let output_supported = output_device
            .default_output_config()
            .map_err(|error| format!("Could not read the output format: {error}"))?;
        let monitor_supported = monitor_device
            .as_ref()
            .map(|(device, _)| {
                device
                    .default_output_config()
                    .map_err(|error| format!("Could not read the monitor format: {error}"))
            })
            .transpose()?;

        let input_rate = input_supported.sample_rate();
        let output_rate = output_supported.sample_rate();
        if input_rate != output_rate {
            return Err(format!(
                "Input runs at {input_rate} Hz but output runs at {output_rate} Hz. Set both Windows devices to the same sample rate before starting passthrough."
            ));
        }
        if let Some(monitor_rate) = monitor_supported
            .as_ref()
            .map(|config| config.sample_rate())
        {
            if input_rate != monitor_rate {
                return Err(format!(
                    "Input and output run at {input_rate} Hz but monitor runs at {monitor_rate} Hz. Set all selected Windows devices to the same sample rate."
                ));
            }
        }

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
            inference_worker.telemetry(),
        ));

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
        let monitor_stream = match (
            monitor_device.as_ref(),
            monitor_supported.as_ref(),
            monitor_config.as_ref(),
        ) {
            (Some((device, _)), Some(supported), Some(config)) => Some(build_monitor_stream(
                device,
                supported.sample_format(),
                config,
                Arc::clone(&telemetry),
                processing,
            )?),
            _ => None,
        };

        if let Some(stream) = &monitor_stream {
            stream
                .play()
                .map_err(|error| format!("Could not start the monitor stream: {error}"))?;
        }
        output_stream
            .play()
            .map_err(|error| format!("Could not start the output stream: {error}"))?;
        input_stream
            .play()
            .map_err(|error| format!("Could not start the input stream: {error}"))?;

        self.active = Some(ActiveAudioEngine {
            _input_stream: input_stream,
            _output_stream: output_stream,
            _monitor_stream: monitor_stream,
            input_device_id: input_id.to_owned(),
            output_device_id: output_id.to_owned(),
            monitor_device_id: monitor_id.map(str::to_owned),
            input_device_name: input_name,
            output_device_name: output_name,
            monitor_device_name: monitor_device.map(|(_, name)| name),
            sample_rate: input_rate,
            input_channels,
            output_channels,
            monitor_channels,
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
                input_peak: 0.0,
                output_peak: 0.0,
                monitor_peak: 0.0,
                input_gain_db: 0.0,
                output_gain_db: 0.0,
                monitor_gain_db: -6.0,
                noise_gate_db: -80.0,
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
            input_peak: f32::from_bits(telemetry.input_peak_bits.load(Ordering::Relaxed)),
            output_peak: f32::from_bits(telemetry.output_peak_bits.load(Ordering::Relaxed)),
            monitor_peak: f32::from_bits(telemetry.monitor_peak_bits.load(Ordering::Relaxed)),
            input_gain_db: active.processing.input_gain_db,
            output_gain_db: active.processing.output_gain_db,
            monitor_gain_db: active.processing.monitor_gain_db,
            noise_gate_db: active.processing.noise_gate_db,
            last_error: audio_error.or(inference.last_error),
        }
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
        if device_identifier(&device, direction, index) == requested_id {
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
                    let processed = input_processor.process(mono);
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

#[derive(Default)]
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
        } else {
            self.primed = false;
            self.stable_frames = 0;
            let previous_target = self.target_frames;
            self.target_frames = self
                .target_frames
                .saturating_add(WORKER_CHUNK_FRAMES)
                .min(MAX_PRIME_FRAMES);
            result.primed = false;
            result.underrun = true;
            result.reprime = true;
            result.target_changed = self.target_frames != previous_target;
            return result;
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
    device
        .build_output_stream(
            *config,
            move |data: &mut [T], _| {
                let mut peak = 0.0_f32;
                for frame in data.chunks_exact_mut(channels) {
                    let stable = stability.read(&telemetry.playback_ring);
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
                    if let Some(monitor_ring) = &telemetry.monitor_ring {
                        if monitor_ring.push(sample).is_err() {
                            let _ = monitor_ring.pop();
                            let _ = monitor_ring.push(sample);
                            telemetry.monitor_overruns.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let sample = (sample * output_gain).clamp(-1.0, 1.0);
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
    device
        .build_output_stream(
            *config,
            move |data: &mut [T], _| {
                let mut peak = 0.0_f32;
                for frame in data.chunks_exact_mut(channels) {
                    let stable = stability.read(&monitor_ring);
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

fn db_to_linear(value: f32) -> f32 {
    10.0_f32.powf(value / 20.0)
}

struct InputProcessor {
    input_gain: f32,
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

    fn process(&mut self, sample: f32) -> f32 {
        let amplified = sample * self.input_gain;
        self.envelope = amplified
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
        amplified * self.gate_gain
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
        assert!(AudioProcessingSettings::new(25.0, 0.0, -6.0, -60.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 13.0, -6.0, -60.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 0.0, 13.0, -60.0).is_err());
        assert!(AudioProcessingSettings::new(0.0, 0.0, -6.0, -19.0).is_err());
        assert!(AudioProcessingSettings::new(6.0, -3.0, -8.0, -55.0).is_ok());
    }

    #[test]
    fn noise_gate_at_minimum_is_transparent() {
        let mut processor = InputProcessor::new(AudioProcessingSettings::default(), 48_000);
        assert!((processor.process(0.25) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn input_gain_is_applied_before_capture() {
        let settings = AudioProcessingSettings::new(6.0, 0.0, -6.0, -80.0).unwrap();
        let mut processor = InputProcessor::new(settings, 48_000);
        assert!((processor.process(0.25) - 0.4988).abs() < 0.002);
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
        assert!(!underrun.primed);
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
}

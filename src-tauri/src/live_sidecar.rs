use std::{
    collections::VecDeque,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
        mpsc::{self, sync_channel, Receiver, SyncSender, TryRecvError, TrySendError},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use serde_json::{json, Value};

use crate::{
    inference::{BackendCapabilities, BackendConfig, InferenceBackend},
    sidecar,
};

const MAGIC: [u8; 4] = *b"VCN1";
const HEADER_BYTES: usize = 16;
const MAX_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;
const JSON_REQUEST: u8 = 1;
const JSON_RESPONSE: u8 = 2;
const AUDIO_REQUEST: u8 = 3;
const AUDIO_RESPONSE: u8 = 4;
const ERROR_RESPONSE: u8 = 5;
const SHUTDOWN: u8 = 6;
const LIVE_SAMPLE_RATE: u32 = 48_000;
pub const LIVE_CHUNK_FRAMES: usize = 9_600;
const LIVE_ANALYSIS_FRAMES: usize = 24_000;
const LIVE_CROSSFADE_FRAMES: usize = 1_920;
const LIVE_SOLA_SEARCH_FRAMES: usize = 576;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const CONTROL_TIMEOUT: Duration = Duration::from_secs(15);
const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(120);
const AUDIO_TIMEOUT: Duration = Duration::from_secs(5);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
const CLIENT_TIMEOUT_MARGIN: Duration = Duration::from_secs(5);
const HEALTHY: u8 = 1;
const RECOVERING: u8 = 2;
const FAILED: u8 = 3;
const _: () = {
    assert!(LIVE_CHUNK_FRAMES.is_multiple_of(480));
    assert!(LIVE_CROSSFADE_FRAMES < LIVE_CHUNK_FRAMES);
    assert!(
        LIVE_CHUNK_FRAMES + LIVE_CROSSFADE_FRAMES + LIVE_SOLA_SEARCH_FRAMES <= LIVE_ANALYSIS_FRAMES
    );
};

pub struct LiveWorkerHealth {
    state: AtomicU8,
    restarts: AtomicU64,
    last_error: Mutex<Option<String>>,
}

pub struct LiveWorkerHealthSnapshot {
    pub state: &'static str,
    pub restarts: u64,
    pub last_error: Option<String>,
}

impl LiveWorkerHealth {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(HEALTHY),
            restarts: AtomicU64::new(0),
            last_error: Mutex::new(None),
        }
    }

    fn begin_recovery(&self, error: String) {
        self.state.store(RECOVERING, Ordering::Release);
        if let Ok(mut last_error) = self.last_error.lock() {
            *last_error = Some(error);
        }
    }

    fn recovered(&self) {
        self.restarts.fetch_add(1, Ordering::Relaxed);
        self.state.store(HEALTHY, Ordering::Release);
    }

    fn failed(&self, error: String) {
        self.state.store(FAILED, Ordering::Release);
        if let Ok(mut last_error) = self.last_error.lock() {
            *last_error = Some(error);
        }
    }

    pub fn snapshot(&self) -> LiveWorkerHealthSnapshot {
        let state = match self.state.load(Ordering::Acquire) {
            RECOVERING => "recovering",
            FAILED => "failed",
            _ => "healthy",
        };
        LiveWorkerHealthSnapshot {
            state,
            restarts: self.restarts.load(Ordering::Relaxed),
            last_error: self.last_error.lock().ok().and_then(|error| error.clone()),
        }
    }
}

struct Frame {
    kind: u8,
    request_id: u32,
    payload: Vec<u8>,
}

fn write_frame(writer: &mut impl Write, frame: &Frame) -> Result<(), String> {
    if frame.payload.len() > MAX_PAYLOAD_BYTES {
        return Err("The live sidecar frame exceeds the payload limit.".to_owned());
    }
    let payload_size = u32::try_from(frame.payload.len())
        .map_err(|_| "The live sidecar payload length is invalid.".to_owned())?;
    let mut header = [0_u8; HEADER_BYTES];
    header[..4].copy_from_slice(&MAGIC);
    header[4] = frame.kind;
    header[8..12].copy_from_slice(&frame.request_id.to_le_bytes());
    header[12..16].copy_from_slice(&payload_size.to_le_bytes());
    writer
        .write_all(&header)
        .and_then(|_| writer.write_all(&frame.payload))
        .and_then(|_| writer.flush())
        .map_err(|error| format!("Could not write to the live sidecar: {error}"))
}

fn read_frame(reader: &mut impl Read) -> Result<Frame, String> {
    let mut header = [0_u8; HEADER_BYTES];
    reader
        .read_exact(&mut header)
        .map_err(|error| format!("Could not read the live sidecar frame header: {error}"))?;
    if header[..4] != MAGIC {
        return Err("The live sidecar returned invalid frame magic.".to_owned());
    }
    let request_id = u32::from_le_bytes(header[8..12].try_into().expect("fixed request ID"));
    let payload_size =
        u32::from_le_bytes(header[12..16].try_into().expect("fixed payload size")) as usize;
    if payload_size > MAX_PAYLOAD_BYTES {
        return Err("The live sidecar frame exceeds the payload limit.".to_owned());
    }
    let mut payload = vec![0_u8; payload_size];
    reader
        .read_exact(&mut payload)
        .map_err(|error| format!("Could not read the live sidecar frame payload: {error}"))?;
    Ok(Frame {
        kind: header[4],
        request_id,
        payload,
    })
}

struct FramedProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    responses: Receiver<Result<Frame, String>>,
    reader_handle: Option<JoinHandle<()>>,
    next_request_id: u32,
    healthy: bool,
}

impl FramedProcess {
    fn spawn() -> Result<Self, String> {
        let engine_dir = sidecar::engine_directory()?;
        let mut failures = Vec::new();
        for candidate in sidecar::python_candidates(&engine_dir) {
            match Self::spawn_candidate(&candidate, &engine_dir) {
                Ok(mut process) => match process.control("handshake", json!({})) {
                    Ok(handshake)
                        if handshake.get("protocolVersion").and_then(Value::as_u64) == Some(1)
                            && handshake.get("transport").and_then(Value::as_str)
                                == Some("framed-stdio")
                            && handshake.get("sampleRate").and_then(Value::as_u64)
                                == Some(u64::from(LIVE_SAMPLE_RATE))
                            && handshake.get("chunkFrames").and_then(Value::as_u64)
                                == Some(LIVE_CHUNK_FRAMES as u64) =>
                    {
                        return Ok(process);
                    }
                    Ok(_) => {
                        failures.push(format!("{}: incompatible handshake", candidate.label));
                        process.terminate();
                    }
                    Err(error) => {
                        failures.push(format!("{}: {error}", candidate.label));
                        process.terminate();
                    }
                },
                Err(error) => failures.push(format!("{}: {error}", candidate.label)),
            }
        }
        Err(format!(
            "No Python runtime could start the persistent RVC worker. {}",
            failures.join(" | ")
        ))
    }

    fn spawn_candidate(
        candidate: &sidecar::PythonCandidate,
        engine_dir: &Path,
    ) -> Result<Self, String> {
        let mut child = Command::new(&candidate.program)
            .args(&candidate.prefix_args)
            .args(["-m", "vc_next_sidecar", "--worker"])
            .current_dir(engine_dir)
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("could not start: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "The live sidecar standard input is unavailable.".to_owned())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "The live sidecar standard output is unavailable.".to_owned())?;
        let (response_sender, responses) = mpsc::channel();
        let reader_handle = thread::Builder::new()
            .name("vc-next-sidecar-reader".to_owned())
            .spawn(move || {
                let mut stdout = BufReader::new(stdout);
                loop {
                    match read_frame(&mut stdout) {
                        Ok(frame) => {
                            if response_sender.send(Ok(frame)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = response_sender.send(Err(error));
                            break;
                        }
                    }
                }
            })
            .map_err(|error| format!("Could not start the sidecar reader thread: {error}"))?;
        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            responses,
            reader_handle: Some(reader_handle),
            next_request_id: 1,
            healthy: true,
        })
    }

    fn next_id(&mut self) -> u32 {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        request_id
    }

    fn exchange(&mut self, kind: u8, payload: Vec<u8>, timeout: Duration) -> Result<Frame, String> {
        if !self.healthy {
            return Err("The live sidecar process is unavailable.".to_owned());
        }
        let request_id = self.next_id();
        if let Err(error) = write_frame(
            &mut self.stdin,
            &Frame {
                kind,
                request_id,
                payload,
            },
        ) {
            self.healthy = false;
            return Err(error);
        }
        let response = match self.responses.recv_timeout(timeout) {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                self.healthy = false;
                return Err(error);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.healthy = false;
                return Err(format!(
                    "The live sidecar did not respond within {:.0} seconds.",
                    timeout.as_secs_f32()
                ));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.healthy = false;
                return Err("The live sidecar response channel closed.".to_owned());
            }
        };
        if response.request_id != request_id {
            self.healthy = false;
            return Err("The live sidecar returned a mismatched request ID.".to_owned());
        }
        if response.kind == ERROR_RESPONSE {
            let error: Value = serde_json::from_slice(&response.payload)
                .map_err(|decode| format!("The live sidecar error was invalid JSON: {decode}"))?;
            return Err(error
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("The live sidecar rejected the request.")
                .to_owned());
        }
        Ok(response)
    }

    fn control(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let payload = serde_json::to_vec(&json!({ "method": method, "params": params }))
            .map_err(|error| format!("Could not encode the live control request: {error}"))?;
        let timeout = match method {
            "handshake" => HANDSHAKE_TIMEOUT,
            "load_model" => MODEL_LOAD_TIMEOUT,
            _ => CONTROL_TIMEOUT,
        };
        let response = self.exchange(JSON_REQUEST, payload, timeout)?;
        if response.kind != JSON_RESPONSE {
            self.healthy = false;
            return Err("The live sidecar returned the wrong control frame kind.".to_owned());
        }
        serde_json::from_slice(&response.payload)
            .map_err(|error| format!("The live sidecar returned invalid control JSON: {error}"))
    }

    fn process_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        if samples.is_empty() {
            return Err("A live RVC request cannot be empty.".to_owned());
        }
        let mut payload = Vec::with_capacity(size_of_val(samples));
        for sample in samples {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        let response = self.exchange(AUDIO_REQUEST, payload, AUDIO_TIMEOUT)?;
        if response.kind != AUDIO_RESPONSE {
            self.healthy = false;
            return Err("The live sidecar returned the wrong audio frame kind.".to_owned());
        }
        if response.payload.len() != samples.len() * size_of::<f32>() {
            self.healthy = false;
            return Err("The live sidecar returned the wrong number of audio bytes.".to_owned());
        }
        let mut output = Vec::with_capacity(LIVE_CHUNK_FRAMES);
        for bytes in response.payload.chunks_exact(size_of::<f32>()) {
            let sample = f32::from_le_bytes(bytes.try_into().expect("four-byte float"));
            if !sample.is_finite() {
                self.healthy = false;
                return Err("The live sidecar returned non-finite audio.".to_owned());
            }
            output.push(sample);
        }
        Ok(output)
    }

    fn terminate(&mut self) {
        if self.healthy {
            let _ = self.exchange(SHUTDOWN, Vec::new(), SHUTDOWN_TIMEOUT);
        }
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.join();
        }
        self.healthy = false;
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }
}

impl Drop for FramedProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.join();
        }
    }
}

enum WorkerRequest {
    Control {
        method: String,
        params: Value,
        response: mpsc::Sender<Result<Value, String>>,
    },
    Audio {
        samples: Vec<f32>,
        response: mpsc::Sender<Result<Vec<f32>, String>>,
    },
    Shutdown,
}

#[derive(Clone)]
pub struct LiveRvcClient {
    sender: SyncSender<WorkerRequest>,
    chunk_frames: Arc<AtomicUsize>,
    health: Arc<LiveWorkerHealth>,
}

impl LiveRvcClient {
    fn update_stream_shape(&self, status: &Value) -> Result<(), String> {
        let chunk_frames = status
            .get("chunkFrames")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| "The live RVC worker returned an invalid chunk size.".to_owned())?;
        if !(480..=480_000).contains(&chunk_frames) {
            return Err("The live RVC chunk must be between 480 and 480000 frames.".to_owned());
        }
        self.chunk_frames.store(chunk_frames, Ordering::Release);
        Ok(())
    }

    fn chunk_frames(&self) -> usize {
        self.chunk_frames.load(Ordering::Acquire)
    }

    fn control(&self, method: &str, params: Value) -> Result<Value, String> {
        let (response_sender, response_receiver) = mpsc::channel();
        match self.sender.try_send(WorkerRequest::Control {
            method: method.to_owned(),
            params,
            response: response_sender,
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err("The persistent RVC control queue is busy.".to_owned())
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err("The persistent RVC worker has stopped.".to_owned())
            }
        }
        let timeout = if method == "load_model" {
            MODEL_LOAD_TIMEOUT + CLIENT_TIMEOUT_MARGIN
        } else {
            CONTROL_TIMEOUT + CLIENT_TIMEOUT_MARGIN
        };
        response_receiver
            .recv_timeout(timeout)
            .map_err(|error| match error {
                mpsc::RecvTimeoutError::Timeout => {
                    "The persistent RVC worker request timed out.".to_owned()
                }
                mpsc::RecvTimeoutError::Disconnected => {
                    "The persistent RVC worker returned no response.".to_owned()
                }
            })?
    }

    fn process_audio_async(
        &self,
        samples: Vec<f32>,
    ) -> Result<Receiver<Result<Vec<f32>, String>>, String> {
        let (response_sender, response_receiver) = mpsc::channel();
        match self.sender.try_send(WorkerRequest::Audio {
            samples,
            response: response_sender,
        }) {
            Ok(()) => Ok(response_receiver),
            Err(TrySendError::Full(_)) => {
                Err("The persistent RVC request queue is full.".to_owned())
            }
            Err(TrySendError::Disconnected(_)) => {
                Err("The persistent RVC worker has stopped.".to_owned())
            }
        }
    }

    pub fn health_snapshot(&self) -> LiveWorkerHealthSnapshot {
        self.health.snapshot()
    }
}

struct LiveWorkerHandle {
    client: LiveRvcClient,
    handle: Option<JoinHandle<()>>,
}

fn recover_framed_process(
    process: &mut FramedProcess,
    health: &LiveWorkerHealth,
    last_load_params: Option<&Value>,
    cause: String,
) -> Result<(), String> {
    health.begin_recovery(cause);
    process.terminate();
    let mut replacement = FramedProcess::spawn().map_err(|error| {
        let message = format!("Could not restart the persistent RVC worker: {error}");
        health.failed(message.clone());
        message
    })?;
    if let Some(params) = last_load_params {
        if let Err(error) = replacement.control("load_model", params.clone()) {
            replacement.terminate();
            let message = format!("The restarted RVC worker could not restore the model: {error}");
            health.failed(message.clone());
            return Err(message);
        }
    }
    *process = replacement;
    health.recovered();
    Ok(())
}

fn control_with_recovery(
    process: &mut FramedProcess,
    health: &LiveWorkerHealth,
    last_load_params: Option<&Value>,
    method: &str,
    params: Value,
) -> Result<Value, String> {
    let first = process.control(method, params.clone());
    if first.is_ok() || process.is_healthy() {
        return first;
    }
    let cause = first
        .err()
        .unwrap_or_else(|| "The persistent RVC transport failed.".to_owned());
    recover_framed_process(process, health, last_load_params, cause)?;
    let retried = process.control(method, params);
    if let Err(error) = &retried {
        if !process.is_healthy() {
            health.failed(format!("The recovered RVC worker failed again: {error}"));
        }
    }
    retried
}

fn audio_with_recovery(
    process: &mut FramedProcess,
    health: &LiveWorkerHealth,
    last_load_params: Option<&Value>,
    samples: &[f32],
) -> Result<Vec<f32>, String> {
    let first = process.process_audio(samples);
    if first.is_ok() || process.is_healthy() {
        return first;
    }
    let cause = first
        .err()
        .unwrap_or_else(|| "The persistent RVC audio transport failed.".to_owned());
    let load_params = last_load_params.ok_or_else(|| {
        health.failed(cause.clone());
        "The RVC worker stopped before a model could be restored.".to_owned()
    })?;
    recover_framed_process(process, health, Some(load_params), cause)?;
    let retried = process.process_audio(samples);
    if let Err(error) = &retried {
        if !process.is_healthy() {
            health.failed(format!(
                "The recovered RVC audio worker failed again: {error}"
            ));
        }
    }
    retried
}

fn remember_successful_control(method: &str, params: &Value, last_load_params: &mut Option<Value>) {
    match method {
        "load_model" => *last_load_params = Some(params.clone()),
        "set_settings" => {
            if let (Some(load), Some(settings)) = (last_load_params.as_mut(), params.as_object()) {
                if let Some(load) = load.as_object_mut() {
                    for (key, value) in settings {
                        load.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        "unload" => *last_load_params = None,
        _ => {}
    }
}

impl LiveWorkerHandle {
    fn start() -> Result<Self, String> {
        let mut process = FramedProcess::spawn()?;
        let (sender, receiver) = sync_channel::<WorkerRequest>(2);
        let health = Arc::new(LiveWorkerHealth::new());
        let client = LiveRvcClient {
            sender,
            chunk_frames: Arc::new(AtomicUsize::new(LIVE_CHUNK_FRAMES)),
            health: Arc::clone(&health),
        };
        let handle = thread::Builder::new()
            .name("vc-next-live-sidecar".to_owned())
            .spawn(move || {
                let mut last_load_params = None;
                while let Ok(request) = receiver.recv() {
                    match request {
                        WorkerRequest::Control {
                            method,
                            params,
                            response,
                        } => {
                            let result = control_with_recovery(
                                &mut process,
                                &health,
                                last_load_params.as_ref(),
                                &method,
                                params.clone(),
                            );
                            if result.is_ok() {
                                remember_successful_control(
                                    &method,
                                    &params,
                                    &mut last_load_params,
                                );
                            }
                            let _ = response.send(result);
                        }
                        WorkerRequest::Audio { samples, response } => {
                            let result = audio_with_recovery(
                                &mut process,
                                &health,
                                last_load_params.as_ref(),
                                &samples,
                            );
                            let _ = response.send(result);
                        }
                        WorkerRequest::Shutdown => {
                            process.terminate();
                            break;
                        }
                    }
                }
            })
            .map_err(|error| format!("Could not start the live sidecar I/O thread: {error}"))?;
        Ok(Self {
            client,
            handle: Some(handle),
        })
    }

    fn shutdown(&mut self) {
        let _ = self.client.sender.send(WorkerRequest::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for LiveWorkerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct LiveRvcService {
    worker: Option<LiveWorkerHandle>,
    status: Value,
}

impl Default for LiveRvcService {
    fn default() -> Self {
        Self {
            worker: None,
            status: empty_status(),
        }
    }
}

impl LiveRvcService {
    fn status_with_health(&self) -> Value {
        let mut status = self.status.clone();
        let health = self
            .worker
            .as_ref()
            .map(|worker| worker.client.health_snapshot());
        if let Some(object) = status.as_object_mut() {
            object.insert(
                "workerState".to_owned(),
                json!(health.as_ref().map_or("stopped", |value| value.state)),
            );
            object.insert(
                "workerRestarts".to_owned(),
                json!(health.as_ref().map_or(0, |value| value.restarts)),
            );
            object.insert(
                "lastWorkerError".to_owned(),
                json!(health.and_then(|value| value.last_error)),
            );
        }
        status
    }

    pub fn load_model(
        &mut self,
        model_path: &str,
        index_path: Option<&str>,
        contentvec_path: Option<&str>,
        pitch_shift: f64,
        index_ratio: f64,
        protect_ratio: f64,
        speaker_id: i64,
        f0_threshold: f64,
        streaming_preset: &str,
        chunk_frames: usize,
        extra_frames: usize,
    ) -> Result<Value, String> {
        if self.worker.is_none() {
            self.worker = Some(LiveWorkerHandle::start()?);
        }
        let worker = self.worker.as_ref().expect("worker was initialized");
        self.status = worker.client.control(
            "load_model",
            json!({
                "modelPath": model_path,
                "indexPath": index_path,
                "contentvecPath": contentvec_path,
                "pitchShift": pitch_shift,
                "indexRatio": index_ratio,
                "protectRatio": protect_ratio,
                "speakerId": speaker_id,
                "f0Threshold": f0_threshold,
                "streamingPreset": streaming_preset,
                "chunkFrames": chunk_frames,
                "extraFrames": extra_frames,
            }),
        )?;
        worker.client.update_stream_shape(&self.status)?;
        Ok(self.status_with_health())
    }

    pub fn set_settings(
        &mut self,
        pitch_shift: f64,
        index_ratio: f64,
        protect_ratio: f64,
        speaker_id: i64,
        f0_threshold: f64,
        streaming_preset: &str,
        chunk_frames: usize,
        extra_frames: usize,
    ) -> Result<Value, String> {
        let worker = self
            .worker
            .as_ref()
            .ok_or_else(|| "No persistent RVC worker is running.".to_owned())?;
        self.status = worker.client.control(
            "set_settings",
            json!({
                "pitchShift": pitch_shift,
                "indexRatio": index_ratio,
                "protectRatio": protect_ratio,
                "speakerId": speaker_id,
                "f0Threshold": f0_threshold,
                "streamingPreset": streaming_preset,
                "chunkFrames": chunk_frames,
                "extraFrames": extra_frames,
            }),
        )?;
        worker.client.update_stream_shape(&self.status)?;
        Ok(self.status_with_health())
    }

    pub fn refresh_status(&mut self) -> Result<Value, String> {
        if let Some(worker) = &self.worker {
            if worker.client.health_snapshot().state != "healthy" {
                return Ok(self.status_with_health());
            }
            self.status = worker.client.control("status", json!({}))?;
        }
        Ok(self.status_with_health())
    }

    pub fn unload(&mut self) -> Result<Value, String> {
        if let Some(worker) = &self.worker {
            self.status = worker.client.control("unload", json!({}))?;
        } else {
            self.status = empty_status();
        }
        Ok(self.status_with_health())
    }

    pub fn ready_client(&self) -> Option<LiveRvcClient> {
        if self.status.get("state").and_then(Value::as_str) == Some("ready") {
            self.worker.as_ref().map(|worker| worker.client.clone())
        } else {
            None
        }
    }
}

fn empty_status() -> Value {
    json!({
        "state": "empty",
        "protocolVersion": 1,
        "modelPath": null,
        "sampleRate": LIVE_SAMPLE_RATE,
        "chunkFrames": LIVE_CHUNK_FRAMES,
        "chunkMilliseconds": 200.0,
        "analysisFrames": LIVE_ANALYSIS_FRAMES,
        "analysisMilliseconds": 500.0,
        "crossfadeFrames": LIVE_CROSSFADE_FRAMES,
        "crossfadeMilliseconds": 40.0,
        "solaSearchFrames": LIVE_SOLA_SEARCH_FRAMES,
        "solaSearchMilliseconds": 12.0,
        "streamPrimed": false,
        "pitchShift": 0.0,
        "speakerId": 0,
        "indexPath": null,
        "indexLoaded": false,
        "indexDimension": null,
        "indexVectorCount": 0,
        "indexType": null,
        "indexRatio": 0.0,
        "protectRatio": 0.5,
        "f0Method": "RMVPE",
        "f0Threshold": 0.03,
        "streamingPreset": "balanced",
        "processCalls": 0,
        "lastProcessMs": 0.0,
        "lastRetrievalMs": 0.0,
        "workerState": "stopped",
        "workerRestarts": 0,
        "lastWorkerError": null,
    })
}

pub struct LiveRvcInferenceBackend {
    client: LiveRvcClient,
    input_accumulator: Vec<f32>,
    output_queue: VecDeque<f32>,
    pending: Option<Receiver<Result<Vec<f32>, String>>>,
    live_chunk_frames: usize,
    max_backlog_frames: usize,
}

impl LiveRvcInferenceBackend {
    pub fn new(client: LiveRvcClient) -> Self {
        let live_chunk_frames = client.chunk_frames();
        Self {
            client,
            input_accumulator: Vec::with_capacity(live_chunk_frames * 2),
            output_queue: VecDeque::with_capacity(live_chunk_frames),
            pending: None,
            live_chunk_frames,
            max_backlog_frames: live_chunk_frames * 4,
        }
    }

    fn poll_pending(&mut self) -> Result<(), String> {
        let Some(pending) = &self.pending else {
            return Ok(());
        };
        match pending.try_recv() {
            Ok(result) => {
                self.pending = None;
                let converted = result?;
                if converted.len() != self.live_chunk_frames {
                    return Err("The live RVC worker returned an invalid frame count.".to_owned());
                }
                self.output_queue.extend(converted);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.pending = None;
                return Err("The live RVC audio response channel closed.".to_owned());
            }
        }
        Ok(())
    }
}

impl InferenceBackend for LiveRvcInferenceBackend {
    fn prepare(&mut self, config: &BackendConfig) -> Result<(), String> {
        if config.sample_rate != LIVE_SAMPLE_RATE {
            return Err(format!(
                "The live RVC prototype requires {LIVE_SAMPLE_RATE} Hz audio."
            ));
        }
        if !(480..=480_000).contains(&config.chunk_frames) {
            return Err(
                "The native worker chunk must be between 480 and 480000 frames.".to_owned(),
            );
        }
        Ok(())
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.poll_pending()?;
        self.input_accumulator.extend_from_slice(input);
        if self.input_accumulator.len() > self.max_backlog_frames {
            let overflow = self.input_accumulator.len() - self.max_backlog_frames;
            self.input_accumulator.drain(..overflow);
            return Err("The live RVC input backlog exceeded four streaming hops.".to_owned());
        }
        if self.pending.is_none() && self.input_accumulator.len() >= self.live_chunk_frames {
            let chunk = self
                .input_accumulator
                .drain(..self.live_chunk_frames)
                .collect::<Vec<_>>();
            self.pending = Some(self.client.process_audio_async(chunk)?);
        }
        for sample in output {
            *sample = self.output_queue.pop_front().unwrap_or(0.0);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.input_accumulator.clear();
        self.output_queue.clear();
        self.pending = None;
    }

    fn inspect_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "Python RVC live worker",
            stateful: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn binary_frame_round_trip_preserves_payload() {
        let expected = Frame {
            kind: AUDIO_REQUEST,
            request_id: 42,
            payload: vec![0, 1, 2, 255],
        };
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &expected).unwrap();
        let actual = read_frame(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(actual.kind, expected.kind);
        assert_eq!(actual.request_id, expected.request_id);
        assert_eq!(actual.payload, expected.payload);
    }

    #[test]
    fn invalid_binary_frame_magic_is_rejected() {
        let mut bytes = vec![0_u8; HEADER_BYTES];
        bytes[12..16].copy_from_slice(&0_u32.to_le_bytes());
        assert!(read_frame(&mut Cursor::new(bytes)).is_err());
    }

    #[test]
    fn empty_service_has_no_ready_client() {
        let service = LiveRvcService::default();
        assert!(service.ready_client().is_none());
    }

    #[test]
    fn client_accepts_valid_dynamic_stream_shape() {
        let (sender, _receiver) = sync_channel(1);
        let client = LiveRvcClient {
            sender,
            chunk_frames: Arc::new(AtomicUsize::new(LIVE_CHUNK_FRAMES)),
            health: Arc::new(LiveWorkerHealth::new()),
        };
        client
            .update_stream_shape(&json!({ "chunkFrames": 7_681 }))
            .unwrap();
        assert_eq!(client.chunk_frames(), 7_681);
        assert!(client
            .update_stream_shape(&json!({ "chunkFrames": 479 }))
            .is_err());
    }

    #[test]
    fn worker_health_tracks_recovery_without_losing_the_cause() {
        let health = LiveWorkerHealth::new();
        health.begin_recovery("pipe timeout".to_owned());
        assert_eq!(health.snapshot().state, "recovering");
        health.recovered();
        let snapshot = health.snapshot();
        assert_eq!(snapshot.state, "healthy");
        assert_eq!(snapshot.restarts, 1);
        assert_eq!(snapshot.last_error.as_deref(), Some("pipe timeout"));
    }

    #[test]
    fn successful_settings_are_replayed_during_recovery() {
        let mut load = Some(json!({
            "modelPath": "voice.pth",
            "pitchShift": 0.0,
            "indexRatio": 0.5
        }));
        remember_successful_control(
            "set_settings",
            &json!({ "pitchShift": 4.0, "indexRatio": 0.25 }),
            &mut load,
        );
        let load = load.unwrap();
        assert_eq!(
            load.get("modelPath").and_then(Value::as_str),
            Some("voice.pth")
        );
        assert_eq!(load.get("pitchShift").and_then(Value::as_f64), Some(4.0));
        assert_eq!(load.get("indexRatio").and_then(Value::as_f64), Some(0.25));
    }
}

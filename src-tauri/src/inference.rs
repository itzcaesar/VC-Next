use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam_queue::ArrayQueue;

pub const WORKER_CHUNK_FRAMES: usize = 480;

pub struct BackendConfig {
    pub sample_rate: u32,
    pub chunk_frames: usize,
}

pub struct BackendCapabilities {
    pub name: &'static str,
    pub stateful: bool,
}

pub trait InferenceBackend: Send + 'static {
    fn prepare(&mut self, config: &BackendConfig) -> Result<(), String>;
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String>;
    fn reset(&mut self);
    fn inspect_capabilities(&self) -> BackendCapabilities;
}

impl<T: InferenceBackend + ?Sized> InferenceBackend for Box<T> {
    fn prepare(&mut self, config: &BackendConfig) -> Result<(), String> {
        (**self).prepare(config)
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        (**self).process(input, output)
    }

    fn reset(&mut self) {
        (**self).reset();
    }

    fn inspect_capabilities(&self) -> BackendCapabilities {
        (**self).inspect_capabilities()
    }
}

#[derive(Default)]
pub struct NoopInferenceBackend;

impl InferenceBackend for NoopInferenceBackend {
    fn prepare(&mut self, config: &BackendConfig) -> Result<(), String> {
        if config.sample_rate == 0 || config.chunk_frames == 0 {
            return Err("The inference worker received an invalid audio format.".to_owned());
        }
        Ok(())
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if input.len() != output.len() {
            return Err("The inference input and output chunks do not match.".to_owned());
        }
        output.copy_from_slice(input);
        Ok(())
    }

    fn reset(&mut self) {}

    fn inspect_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "No-op inference seam",
            stateful: false,
        }
    }
}

pub struct InferenceTelemetry {
    backend_name: &'static str,
    stateful: bool,
    processed_frames: AtomicU64,
    process_calls: AtomicU64,
    last_process_micros: AtomicU64,
    max_process_micros: AtomicU64,
    missed_deadlines: AtomicU64,
    dropped_output_frames: AtomicU64,
    last_error: Mutex<Option<String>>,
}

pub struct InferenceSnapshot {
    pub backend_name: &'static str,
    pub stateful: bool,
    pub processed_frames: u64,
    pub process_calls: u64,
    pub last_process_micros: u64,
    pub max_process_micros: u64,
    pub missed_deadlines: u64,
    pub dropped_output_frames: u64,
    pub last_error: Option<String>,
}

impl InferenceTelemetry {
    fn new(capabilities: BackendCapabilities) -> Self {
        Self {
            backend_name: capabilities.name,
            stateful: capabilities.stateful,
            processed_frames: AtomicU64::new(0),
            process_calls: AtomicU64::new(0),
            last_process_micros: AtomicU64::new(0),
            max_process_micros: AtomicU64::new(0),
            missed_deadlines: AtomicU64::new(0),
            dropped_output_frames: AtomicU64::new(0),
            last_error: Mutex::new(None),
        }
    }

    pub fn snapshot(&self) -> InferenceSnapshot {
        InferenceSnapshot {
            backend_name: self.backend_name,
            stateful: self.stateful,
            processed_frames: self.processed_frames.load(Ordering::Relaxed),
            process_calls: self.process_calls.load(Ordering::Relaxed),
            last_process_micros: self.last_process_micros.load(Ordering::Relaxed),
            max_process_micros: self.max_process_micros.load(Ordering::Relaxed),
            missed_deadlines: self.missed_deadlines.load(Ordering::Relaxed),
            dropped_output_frames: self.dropped_output_frames.load(Ordering::Relaxed),
            last_error: self.last_error.lock().ok().and_then(|error| error.clone()),
        }
    }
}

pub struct InferenceWorker {
    stop_requested: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    telemetry: Arc<InferenceTelemetry>,
}

impl InferenceWorker {
    pub fn start<B: InferenceBackend>(
        mut backend: B,
        input: Arc<ArrayQueue<f32>>,
        output: Arc<ArrayQueue<f32>>,
        sample_rate: u32,
    ) -> Result<Self, String> {
        let config = BackendConfig {
            sample_rate,
            chunk_frames: WORKER_CHUNK_FRAMES,
        };
        let capabilities = backend.inspect_capabilities();
        backend.prepare(&config)?;

        let telemetry = Arc::new(InferenceTelemetry::new(capabilities));
        let worker_telemetry = Arc::clone(&telemetry);
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop_requested);
        let chunk_budget_micros = (WORKER_CHUNK_FRAMES as u64 * 1_000_000) / sample_rate as u64;

        let handle = thread::Builder::new()
            .name("vc-next-inference".to_owned())
            .spawn(move || {
                let mut input_chunk = vec![0.0_f32; WORKER_CHUNK_FRAMES];
                let mut output_chunk = vec![0.0_f32; WORKER_CHUNK_FRAMES];

                while !worker_stop.load(Ordering::Acquire) {
                    if input.len() < WORKER_CHUNK_FRAMES {
                        thread::sleep(Duration::from_micros(250));
                        continue;
                    }

                    for sample in &mut input_chunk {
                        *sample = input.pop().unwrap_or(0.0);
                    }

                    let started = Instant::now();
                    if let Err(error) = backend.process(&input_chunk, &mut output_chunk) {
                        if let Ok(mut last_error) = worker_telemetry.last_error.lock() {
                            *last_error = Some(error);
                        }
                        output_chunk.fill(0.0);
                    }
                    let elapsed_micros = started.elapsed().as_micros() as u64;

                    worker_telemetry
                        .last_process_micros
                        .store(elapsed_micros, Ordering::Relaxed);
                    worker_telemetry
                        .max_process_micros
                        .fetch_max(elapsed_micros, Ordering::Relaxed);
                    worker_telemetry
                        .process_calls
                        .fetch_add(1, Ordering::Relaxed);
                    worker_telemetry
                        .processed_frames
                        .fetch_add(WORKER_CHUNK_FRAMES as u64, Ordering::Relaxed);
                    if elapsed_micros > chunk_budget_micros {
                        worker_telemetry
                            .missed_deadlines
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    for sample in output_chunk.iter().copied() {
                        if output.push(sample).is_err() {
                            worker_telemetry
                                .dropped_output_frames
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                backend.reset();
            })
            .map_err(|error| format!("Could not start the inference worker: {error}"))?;

        Ok(Self {
            stop_requested,
            handle: Some(handle),
            telemetry,
        })
    }

    pub fn telemetry(&self) -> Arc<InferenceTelemetry> {
        Arc::clone(&self.telemetry)
    }

    pub fn stop(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_backend_preserves_samples() {
        let mut backend = NoopInferenceBackend;
        backend
            .prepare(&BackendConfig {
                sample_rate: 48_000,
                chunk_frames: 4,
            })
            .unwrap();
        let input = [-0.5, 0.0, 0.25, 1.0];
        let mut output = [0.0; 4];
        backend.process(&input, &mut output).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn invalid_backend_config_is_rejected() {
        let mut backend = NoopInferenceBackend;
        assert!(backend
            .prepare(&BackendConfig {
                sample_rate: 0,
                chunk_frames: WORKER_CHUNK_FRAMES,
            })
            .is_err());
    }

    #[test]
    fn worker_moves_a_complete_chunk_and_records_telemetry() {
        let input = Arc::new(ArrayQueue::new(WORKER_CHUNK_FRAMES * 2));
        let output = Arc::new(ArrayQueue::new(WORKER_CHUNK_FRAMES * 2));
        for index in 0..WORKER_CHUNK_FRAMES {
            input
                .push(index as f32 / WORKER_CHUNK_FRAMES as f32)
                .unwrap();
        }

        let mut worker = InferenceWorker::start(
            NoopInferenceBackend,
            Arc::clone(&input),
            Arc::clone(&output),
            48_000,
        )
        .unwrap();
        let telemetry = worker.telemetry();
        let deadline = Instant::now() + Duration::from_millis(250);
        while output.len() < WORKER_CHUNK_FRAMES && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        worker.stop();

        assert_eq!(output.len(), WORKER_CHUNK_FRAMES);
        for index in 0..WORKER_CHUNK_FRAMES {
            let expected = index as f32 / WORKER_CHUNK_FRAMES as f32;
            assert_eq!(output.pop(), Some(expected));
        }
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.processed_frames, WORKER_CHUNK_FRAMES as u64);
        assert_eq!(snapshot.process_calls, 1);
        assert_eq!(snapshot.missed_deadlines, 0);
    }
}

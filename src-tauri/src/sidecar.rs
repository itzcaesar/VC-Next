use std::{
    env,
    ffi::OsString,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use serde::Deserialize;
use serde_json::{json, Value};

const PROTOCOL_VERSION: u32 = 1;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SidecarResponse {
    protocol_version: u32,
    request_id: String,
    ok: bool,
    result: Option<Value>,
    error: Option<SidecarError>,
}

#[derive(Deserialize)]
struct SidecarError {
    code: String,
    message: String,
}

pub(crate) struct PythonCandidate {
    pub(crate) program: OsString,
    pub(crate) prefix_args: Vec<OsString>,
    pub(crate) label: String,
}

impl PythonCandidate {
    fn executable(path: PathBuf) -> Self {
        Self {
            label: path.display().to_string(),
            program: path.into_os_string(),
            prefix_args: Vec::new(),
        }
    }

    fn command(program: &str, prefix_args: &[&str]) -> Self {
        Self {
            program: OsString::from(program),
            prefix_args: prefix_args.iter().map(OsString::from).collect(),
            label: std::iter::once(program)
                .chain(prefix_args.iter().copied())
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

pub fn probe_runtime() -> Result<Value, String> {
    call("probe_runtime", json!({}))
}

pub fn inspect_model(path: &str) -> Result<Value, String> {
    call("inspect_model", json!({ "path": path }))
}

pub fn inspect_trusted_checkpoint(path: &str) -> Result<Value, String> {
    call("inspect_trusted_checkpoint", json!({ "path": path }))
}

fn call(method: &str, params: Value) -> Result<Value, String> {
    let engine_dir = engine_directory()?;
    let request_id = format!("tauri-{method}");
    let request = json!({
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "method": method,
        "params": params,
    });
    let request_line = serde_json::to_string(&request)
        .map_err(|error| format!("Could not serialize the sidecar request: {error}"))?;

    let mut failures = Vec::new();
    for candidate in python_candidates(&engine_dir) {
        match call_candidate(&candidate, &engine_dir, &request_line) {
            Ok(response) => {
                if response.protocol_version != PROTOCOL_VERSION {
                    return Err(format!(
                        "The Python sidecar returned protocol version {} instead of {}.",
                        response.protocol_version, PROTOCOL_VERSION
                    ));
                }
                if response.request_id != request_id {
                    return Err("The Python sidecar returned a mismatched request ID.".to_owned());
                }
                if response.ok {
                    return response
                        .result
                        .ok_or_else(|| "The Python sidecar returned no result.".to_owned());
                }
                let error = response.error.unwrap_or(SidecarError {
                    code: "unknown_error".to_owned(),
                    message: "The Python sidecar rejected the request.".to_owned(),
                });
                return Err(format!("{}: {}", error.code, error.message));
            }
            Err(error) => failures.push(format!("{}: {error}", candidate.label)),
        }
    }

    Err(format!(
        "No usable Python runtime could start the VC Next sidecar. {}",
        failures.join(" | ")
    ))
}

fn call_candidate(
    candidate: &PythonCandidate,
    engine_dir: &Path,
    request_line: &str,
) -> Result<SidecarResponse, String> {
    let mut command = Command::new(&candidate.program);
    command
        .args(&candidate.prefix_args)
        .args(["-m", "vc_next_sidecar", "--once"])
        .current_dir(engine_dir)
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("could not start: {error}"))?;
    let Some(mut stdin) = child.stdin.take() else {
        return Err("standard input was unavailable".to_owned());
    };
    stdin
        .write_all(format!("{request_line}\n").as_bytes())
        .map_err(|error| format!("could not send the request: {error}"))?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(|error| format!("could not read the response: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "exited with {} ({})",
            output.status,
            compact_message(&stderr)
        ));
    }

    let stdout =
        String::from_utf8(output.stdout).map_err(|_| "returned non-UTF-8 output".to_owned())?;
    let response_line = stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "returned an empty response".to_owned())?;
    serde_json::from_str(response_line).map_err(|error| format!("returned invalid JSON: {error}"))
}

pub(crate) fn engine_directory() -> Result<PathBuf, String> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Some(project_root) = manifest.parent() else {
        return Err("Could not resolve the VC Next project root.".to_owned());
    };
    let engine_dir = project_root.join("engine-python");
    if engine_dir.join("vc_next_sidecar").is_dir() {
        Ok(engine_dir)
    } else {
        Err(format!(
            "The Python sidecar package is missing at {}.",
            engine_dir.display()
        ))
    }
}

pub(crate) fn python_candidates(engine_dir: &Path) -> Vec<PythonCandidate> {
    let mut candidates = Vec::new();
    if let Some(configured) = env::var_os("VC_NEXT_PYTHON").filter(|path| !path.is_empty()) {
        candidates.push(PythonCandidate::executable(PathBuf::from(configured)));
    }

    let sidecar_venv = engine_dir.join(".venv").join("Scripts").join("python.exe");
    if sidecar_venv.is_file() {
        candidates.push(PythonCandidate::executable(sidecar_venv));
    }
    if let Some(project_root) = engine_dir.parent() {
        let project_venv = project_root
            .join(".venv")
            .join("Scripts")
            .join("python.exe");
        if project_venv.is_file() {
            candidates.push(PythonCandidate::executable(project_venv));
        }
    }

    if cfg!(windows) {
        candidates.push(PythonCandidate::command("py", &["-3.11"]));
    }
    candidates.push(PythonCandidate::command("python", &[]));
    candidates
}

fn compact_message(message: &str) -> String {
    let compact = message.split_whitespace().collect::<Vec<_>>().join(" ");
    compact.chars().take(300).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_sidecar_directory_is_present() {
        let directory = engine_directory().unwrap();
        assert!(directory.join("vc_next_sidecar").is_dir());
    }

    #[test]
    fn long_sidecar_errors_are_bounded() {
        let message = "failure ".repeat(100);
        assert_eq!(compact_message(&message).chars().count(), 300);
    }
}

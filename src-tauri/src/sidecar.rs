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

/// Open the visible first-run runtime bootstrap in a separate PowerShell
/// window. This is intentionally not run inside the Tauri command thread: pip
/// installs and CUDA package downloads can take several minutes, and the user
/// needs to see their progress and any driver/package error directly.
pub fn open_runtime_setup() -> Result<String, String> {
    let engine_dir = engine_directory()?;
    let script = runtime_setup_script(&engine_dir)?;
    if !cfg!(windows) {
        return Err("The VC Next runtime bootstrap currently supports Windows only.".to_owned());
    }

    Command::new("powershell.exe")
        .args([
            OsString::from("-NoProfile"),
            OsString::from("-ExecutionPolicy"),
            OsString::from("Bypass"),
            OsString::from("-NoExit"),
            OsString::from("-File"),
        ])
        .arg(&script)
        .current_dir(&engine_dir)
        .spawn()
        .map_err(|error| format!("Could not open the runtime setup window: {error}"))?;
    Ok(script.display().to_string())
}

/// Return a copyable command for the visible runtime bootstrap.
///
/// The source checkout can use `npm run runtime:setup`, but an installed
/// application has no project root or npm script. Resolve the staged script
/// beside the installed sidecar so the command is useful in both cases.
pub fn runtime_setup_command() -> Result<String, String> {
    let engine_dir = engine_directory()?;
    let script = runtime_setup_script(&engine_dir)?;
    let escaped = script.display().to_string().replace('\'', "''");
    if cfg!(windows) {
        Ok(format!(
            "powershell.exe -NoProfile -ExecutionPolicy Bypass -File '{escaped}'"
        ))
    } else {
        Ok(format!("pwsh -NoProfile -File '{escaped}'"))
    }
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
    let mut candidates = Vec::new();
    if let Some(project_root) = manifest.parent() {
        candidates.push(project_root.join("engine-python"));
    }
    if let Some(configured) = env::var_os("VC_NEXT_ENGINE_DIR").filter(|path| !path.is_empty()) {
        candidates.insert(0, PathBuf::from(configured));
    }
    if let Ok(executable) = env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("engine-python"));
            candidates.push(parent.join("resources").join("engine-python"));
        }
    }
    if let Ok(current) = env::current_dir() {
        candidates.push(current.join("engine-python"));
    }

    for candidate in &candidates {
        if candidate.join("vc_next_sidecar").is_dir() {
            return Ok(candidate.clone());
        }
    }
    let searched = candidates
        .iter()
        .map(|candidate| candidate.display().to_string())
        .collect::<Vec<_>>()
        .join("; ");
    Err(format!(
        "The Python sidecar package could not be found. Searched: {searched}"
    ))
}

fn runtime_setup_script(engine_dir: &Path) -> Result<PathBuf, String> {
    let mut candidates = vec![engine_dir.join("setup-runtime.ps1")];
    if let Some(project_root) = engine_dir.parent() {
        candidates.push(project_root.join("scripts").join("setup-runtime.ps1"));
    }
    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            format!(
                "The VC Next runtime setup script could not be found beside {}.",
                engine_dir.display()
            )
        })
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

    // A packaged installer may live under Program Files, where the bundled
    // resources are intentionally read-only. setup-runtime.ps1 then places
    // the venv under the user's LocalAppData directory instead.
    if let Some(local_app_data) = env::var_os("LOCALAPPDATA").filter(|path| !path.is_empty()) {
        let user_venv = PathBuf::from(local_app_data)
            .join("VC Next")
            .join("engine-python")
            .join(".venv")
            .join("Scripts")
            .join("python.exe");
        if user_venv.is_file() {
            candidates.push(PythonCandidate::executable(user_venv));
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

    #[test]
    fn runtime_setup_script_is_available_from_source_checkout() {
        let engine = engine_directory().unwrap();
        assert!(runtime_setup_script(&engine).unwrap().is_file());
    }

    #[test]
    fn runtime_setup_command_points_at_the_staged_script() {
        let command = runtime_setup_command().unwrap();
        assert!(command.contains("setup-runtime.ps1"));
        assert!(command.contains("ExecutionPolicy") || command.contains("-File"));
    }
}

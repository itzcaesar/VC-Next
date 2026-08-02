param(
    [string]$ModelPath = "",
    [string]$IndexPath = "",
    [switch]$SkipModelSmoke
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    npm run build
    cargo test --manifest-path src-tauri\Cargo.toml
    cargo test --manifest-path src-tauri\Cargo.toml --features native-validation --bin native-route-validation
    $python = Join-Path $repoRoot "engine-python\.venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The project Python environment is missing at $python. Run npm run runtime:setup first."
    }
    & $python -m unittest discover -s engine-python\tests -p "test_*.py" -q

    $probeRequest = '{"protocolVersion":1,"requestId":"release-check","method":"probe_runtime","params":{}}'
    $probeOutput = $probeRequest | & $python -m vc_next_sidecar --once
    if ($LASTEXITCODE -ne 0) {
        throw "The Python runtime probe failed with exit code $LASTEXITCODE. Run npm run runtime:setup first."
    }
    $probe = $probeOutput | ConvertFrom-Json
    if (-not $probe.ok) {
        throw "The Python runtime probe returned an error: $($probe.error.message)"
    }
    if (-not $probe.result.readyForRvc) {
        throw "The selected runtime is not RVC-ready: $($probe.result.blockers -join '; ')"
    }
    Write-Output "Runtime probe passed: CUDA=$($probe.result.torchRuntime.cudaAvailable), ONNX CUDA=$($probe.result.onnxRuntime.cudaProviderAvailable)."

    if (-not $SkipModelSmoke) {
        if (-not $ModelPath -or -not $IndexPath) {
            throw "Pass -ModelPath and -IndexPath for the real-model smoke test, or use -SkipModelSmoke."
        }
        & (Join-Path $repoRoot "engine-python\.venv\Scripts\python.exe") engine-python\tools\live_worker_smoke.py `
            --model $ModelPath --index $IndexPath --index-ratio 0.5 --streaming-preset balanced --chunks 3
    }
    Write-Output "VC Next release checks passed."
} finally {
    Pop-Location
}

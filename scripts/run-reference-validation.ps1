param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$IndexPath = "",
    [string]$OutputDirectory = "outputs\reference-validation",
    [string]$InputDevice = "",
    [string]$OutputDevice = "",
    [int]$InputSampleRate = 0,
    [int]$OutputSampleRate = 0,
    [int]$AudioSeconds = 10,
    [int]$ImpulseCount = 0,
    [double]$SoakSeconds = 0,
    [int]$SoakChunkFrames = 0,
    [int]$SoakExtraFrames = 0,
    [switch]$SoakRealtime,
    [string]$ConvertedInputDevice = "",
    [string]$ConvertedOutputDevice = "",
    [double]$ConvertedRouteSeconds = 0,
    [int]$ConvertedRouteBlockSize = 480,
    [ValidateSet("quality", "balanced", "latency")]
    [string]$ConvertedRoutePreset = "quality",
    [switch]$ConvertedRouteStrict,
    [switch]$SkipAudio
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "engine-python\.venv\Scripts\python.exe"
$model = (Resolve-Path -LiteralPath $ModelPath -ErrorAction Stop).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The project Python environment is missing. Run npm run runtime:setup first."
}
if ($IndexPath) {
    $index = (Resolve-Path -LiteralPath $IndexPath -ErrorAction Stop).Path
} else {
    $index = $null
}
if ($AudioSeconds -le 0) {
    throw "AudioSeconds must be positive."
}
if ($ImpulseCount -lt 0) {
    throw "ImpulseCount cannot be negative."
}
if ($SoakSeconds -lt 0) {
    throw "SoakSeconds cannot be negative."
}
if ($SoakChunkFrames -lt 0 -or $SoakExtraFrames -lt 0) {
    throw "Soak chunk and extra values cannot be negative."
}
if ($InputSampleRate -lt 0 -or $OutputSampleRate -lt 0) {
    throw "InputSampleRate and OutputSampleRate cannot be negative."
}
if ($ConvertedRouteSeconds -lt 0) {
    throw "ConvertedRouteSeconds cannot be negative."
}
if ($ConvertedRouteBlockSize -le 0) {
    throw "ConvertedRouteBlockSize must be positive."
}
if (($ConvertedInputDevice -and -not $ConvertedOutputDevice) -or (-not $ConvertedInputDevice -and $ConvertedOutputDevice)) {
    throw "Pass both ConvertedInputDevice and ConvertedOutputDevice for converted-route validation."
}
if ($ConvertedRouteSeconds -gt 0 -and (-not $ConvertedInputDevice -or -not $ConvertedOutputDevice)) {
    throw "ConvertedRouteSeconds requires ConvertedInputDevice and ConvertedOutputDevice."
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

function Read-JsonFile {
    param([string]$Path)
    return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
}

function Get-Basename {
    param([object]$Value)
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return $null
    }
    return [System.IO.Path]::GetFileName([string]$Value)
}

Write-Host "==> Probing Python/CUDA/ONNX runtime"
$probeRequest = '{"protocolVersion":1,"requestId":"reference-runtime","method":"probe_runtime","params":{}}'
$probeFile = Join-Path $outputPath "runtime.json"
$probeRequest | & $python -m vc_next_sidecar --once | Set-Content -LiteralPath $probeFile -Encoding UTF8
if ($LASTEXITCODE -ne 0) {
    throw "Runtime probe failed with exit code $LASTEXITCODE."
}
$runtime = Read-JsonFile $probeFile
if (-not $runtime.ok) {
    throw "Runtime probe rejected the request: $($runtime.error.message)"
}

Write-Host "==> Running paired-model live worker smoke"
$smokeFile = Join-Path $outputPath "live-worker.json"
$smokeErrorFile = Join-Path $outputPath "live-worker.stderr.txt"
$smokeArgs = @(
    "engine-python\tools\live_worker_smoke.py",
    "--model", $model,
    "--streaming-preset", "balanced",
    "--chunks", "3"
)
if ($index) { $smokeArgs += @("--index", $index, "--index-ratio", "0.3") }
$commandPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
try {
    & $python @smokeArgs 2> $smokeErrorFile | Set-Content -LiteralPath $smokeFile -Encoding UTF8
} finally {
    $ErrorActionPreference = $commandPreference
}
if ($LASTEXITCODE -ne 0) {
    throw "Live worker smoke failed with exit code $LASTEXITCODE. See $smokeErrorFile"
}
$smoke = Read-JsonFile $smokeFile

# Keep the portable report useful without copying a user's local directory
# layout into a diagnostic artifact.
if ($runtime.result.python) {
    $runtime.result.python.executable = Get-Basename $runtime.result.python.executable
}
if ($smoke.status) {
    foreach ($property in @("modelPath", "contentvecPath", "rmvpePath", "indexPath")) {
        if ($smoke.status.PSObject.Properties.Name -contains $property) {
            $smoke.status.$property = Get-Basename $smoke.status.$property
        }
    }
}

$audio = $null
if (-not $SkipAudio) {
    if (-not $InputDevice -or -not $OutputDevice) {
        throw "Pass -InputDevice and -OutputDevice for audio validation, or use -SkipAudio."
    }
    Write-Host "==> Running endpoint loopback validation"
    $audioFile = Join-Path $outputPath "audio-loopback.json"
    $audioArgs = @(
        "engine-python\tools\audio_validation.py",
        "--mode", "loopback",
        "--input-device", $InputDevice,
        "--output-device", $OutputDevice,
        "--seconds", $AudioSeconds,
        "--report", $audioFile
    )
    if ($ImpulseCount -gt 0) { $audioArgs += @("--impulse-count", $ImpulseCount) }
    if ($InputSampleRate -gt 0) { $audioArgs += @("--input-sample-rate", $InputSampleRate) }
    if ($OutputSampleRate -gt 0) { $audioArgs += @("--output-sample-rate", $OutputSampleRate) }
    $commandPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $python @audioArgs
    } finally {
        $ErrorActionPreference = $commandPreference
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Audio loopback validation failed. See the command output and $audioFile"
    }
    $audio = Read-JsonFile $audioFile
    if ($ImpulseCount -gt 0) {
        $actualImpulses = [int]$audio.loopback.impulses
        $detectedImpulses = [int]$audio.loopback.detected
        if ($actualImpulses -ne $ImpulseCount -or $detectedImpulses -lt $ImpulseCount) {
            throw "Loopback acceptance failed: requested $ImpulseCount impulses, generated $actualImpulses, detected $detectedImpulses. See $audioFile"
        }
    }
}

$convertedSoak = $null
if ($SoakSeconds -gt 0) {
    Write-Host "==> Running converted-worker soak ($SoakSeconds seconds)"
    $soakFile = Join-Path $outputPath "converted-worker-soak.json"
    $soakErrorFile = Join-Path $outputPath "converted-worker-soak.stderr.txt"
    $soakArgs = @(
        "engine-python\tools\live_worker_soak.py",
        "--model", $model,
        "--seconds", $SoakSeconds,
        "--status-interval", "30",
        "--report", $soakFile
    )
    if ($index) {
        $soakArgs += @("--index", $index, "--index-ratio", "0.3")
    } else {
        $soakArgs += @("--index-ratio", "0")
    }
    if ($SoakChunkFrames -gt 0) { $soakArgs += @("--chunk-frames", $SoakChunkFrames) }
    if ($SoakExtraFrames -gt 0) { $soakArgs += @("--extra-frames", $SoakExtraFrames) }
    if ($SoakRealtime) { $soakArgs += "--realtime" }
    $commandPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & $python @soakArgs 2> $soakErrorFile | Set-Content -LiteralPath $soakFile -Encoding UTF8
    } finally {
        $ErrorActionPreference = $commandPreference
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Converted-worker soak failed with exit code $LASTEXITCODE. See $soakErrorFile"
    }
    $convertedSoak = Read-JsonFile $soakFile
    if (-not $convertedSoak.finite -or [int]$convertedSoak.deadlineMisses -gt 0) {
        throw "Converted-worker soak acceptance failed: finite=$($convertedSoak.finite), deadlineMisses=$($convertedSoak.deadlineMisses). See $soakFile"
    }
}

$convertedRoute = $null
if ($ConvertedRouteSeconds -gt 0) {
    Write-Host "==> Running duplex converted-route validation ($ConvertedRouteSeconds seconds)"
    $routeFile = Join-Path $outputPath "converted-route.json"
    $routeErrorFile = Join-Path $outputPath "converted-route.stderr.txt"
    $routeArgs = @(
        "engine-python\tools\live_route_validation.py",
        "--model", $model,
        "--input-device", $ConvertedInputDevice,
        "--output-device", $ConvertedOutputDevice,
        "--seconds", $ConvertedRouteSeconds,
        "--block-size", $ConvertedRouteBlockSize,
        "--streaming-preset", $ConvertedRoutePreset,
        "--report", $routeFile
    )
    if ($index) { $routeArgs += @("--index", $index, "--index-ratio", "0.3") }
    else { $routeArgs += @("--index-ratio", "0") }
    if ($ConvertedRouteStrict) { $routeArgs += "--strict" }
    $commandPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & $python @routeArgs 2> $routeErrorFile | Set-Content -LiteralPath $routeFile -Encoding UTF8
    } finally {
        $ErrorActionPreference = $commandPreference
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Duplex converted-route validation failed with exit code $LASTEXITCODE. See $routeErrorFile"
    }
    $convertedRoute = Read-JsonFile $routeFile
    if (-not $convertedRoute.acceptancePassed) {
        throw "Duplex converted-route acceptance failed. See $routeFile"
    }
}

$report = [ordered]@{
    product = "VC Next"
    generatedAt = [DateTime]::UtcNow.ToString("o")
    modelPath = [System.IO.Path]::GetFileName($model)
    indexPath = if ($index) { [System.IO.Path]::GetFileName($index) } else { $null }
    runtime = $runtime.result
    liveWorker = $smoke
    audioLoopback = $audio
    convertedSoak = $convertedSoak
    convertedRoute = $convertedRoute
}
$reportFile = Join-Path $outputPath "reference-validation.json"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportFile -Encoding UTF8

Write-Host ""
Write-Host "Reference validation passed."
Write-Host "Report: $reportFile"
Write-Host "Model deadline met: $($smoke.deadlineMet)"
if ($audio) {
    Write-Host "Loopback P50: $($audio.loopback.delayMs.p50) ms"
    Write-Host "Loopback P95: $($audio.loopback.delayMs.p95) ms"
}
if ($convertedSoak) {
    Write-Host "Converted soak calls: $($convertedSoak.calls)"
    Write-Host "Converted soak P95: $($convertedSoak.processMs.p95) ms"
    Write-Host "Converted soak deadline misses: $($convertedSoak.deadlineMisses)"
}
if ($convertedRoute) {
    Write-Host "Converted route P95: $($convertedRoute.processMs.p95) ms"
    Write-Host "Converted route first played: $($convertedRoute.firstPlayedLatencyMs) ms"
    Write-Host "Converted route underruns: $($convertedRoute.outputUnderruns)"
}

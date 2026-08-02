param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$IndexPath = "",
    [string]$ContentVecPath = "",
    [Parameter(Mandatory = $true)]
    [string]$FixturePath,
    [string]$InputDevice = "CABLE-A Output (VB-Audio Cable A)",
    [string]$FixtureOutputDevice = "CABLE-A Input (VB-Audio Cable A)",
    [string]$OutputDevice = "CABLE-B Input (VB-Audio Cable B)",
    [double]$Seconds = 12,
    [double]$PitchShift = 14,
    [double]$IndexRatio = 0.5,
    [double]$ProtectRatio = 0.5,
    [int]$ChunkFrames = 9600,
    [int]$ExtraFrames = 24000,
    [ValidateSet("quality", "balanced", "latency")]
    [string]$Preset = "balanced",
    [string]$ReportPath = "outputs\native-speech-loopback.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "engine-python\.venv\Scripts\python.exe"
$model = (Resolve-Path -LiteralPath $ModelPath -ErrorAction Stop).Path
$fixture = (Resolve-Path -LiteralPath $FixturePath -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The project Python environment is missing. Run npm run runtime:setup first."
}

$native = Join-Path $repoRoot "src-tauri\target\debug\native-route-validation.exe"
if (-not (Test-Path -LiteralPath $native -PathType Leaf)) {
    Push-Location $repoRoot
    try {
        & cargo build --manifest-path src-tauri\Cargo.toml --features native-validation --bin native-route-validation
        if ($LASTEXITCODE -ne 0) { throw "The native validation binary could not be built." }
    } finally {
        Pop-Location
    }
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    [System.IO.Path]::GetFullPath($ReportPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ReportPath))
}
New-Item -ItemType Directory -Path (Split-Path -Parent $reportFullPath) -Force | Out-Null
$playerSeconds = [Math]::Max($Seconds + 20, 30)
# Start-Process receives a single Windows command line.  Passing an array here
# silently drops the grouping around paths/device names that contain spaces,
# which made the fixture process fail before it opened the cable.
function Quote-ProcessArgument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}
$playerArgs = @(
    (Quote-ProcessArgument "engine-python\tools\playback_fixture.py"),
    "--input", (Quote-ProcessArgument $fixture),
    "--device", (Quote-ProcessArgument $FixtureOutputDevice),
    "--seconds", (Quote-ProcessArgument ([string]$playerSeconds))
) -join " "
$playerStdout = Join-Path $repoRoot "outputs\native-speech-loopback-player.stdout.txt"
$playerStderr = Join-Path $repoRoot "outputs\native-speech-loopback-player.stderr.txt"
$player = Start-Process -FilePath $python -ArgumentList $playerArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $playerStdout -RedirectStandardError $playerStderr

try {
    # Give PortAudio a moment to open the virtual output before the native
    # validator starts its model warm-up.
    Start-Sleep -Milliseconds 500
    $nativeArgs = @(
        "--model", $model,
        "--input", $InputDevice,
        "--output", $OutputDevice,
        "--seconds", $Seconds,
        "--pitch", $PitchShift,
        "--index-ratio", $IndexRatio,
        "--protect", $ProtectRatio,
        "--chunk", $ChunkFrames,
        "--extra", $ExtraFrames,
        "--preset", $Preset,
        "--report", $reportFullPath
    )
    if ($IndexPath) { $nativeArgs += @("--index", (Resolve-Path -LiteralPath $IndexPath -ErrorAction Stop).Path) }
    if ($ContentVecPath) { $nativeArgs += @("--contentvec", (Resolve-Path -LiteralPath $ContentVecPath -ErrorAction Stop).Path) }

    & $native @nativeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Native speech loopback failed with exit code $LASTEXITCODE. See $reportFullPath"
    }
} finally {
    if (-not $player.HasExited) {
        Stop-Process -Id $player.Id -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $reportFullPath -PathType Leaf)) {
    throw "The native speech loopback did not produce a report: $reportFullPath"
}
Write-Host "Native speech loopback report: $reportFullPath"

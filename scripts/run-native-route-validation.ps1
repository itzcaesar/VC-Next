param(
    [switch]$List,
    [Parameter(Mandatory = $false)]
    [string]$ModelPath = "",
    [string]$IndexPath = "",
    [string]$ContentVecPath = "",
    [string]$InputDevice = "",
    [string]$OutputDevice = "",
    [string]$MonitorDevice = "",
    [double]$Seconds = 5,
    [double]$PitchShift = 14,
    [double]$IndexRatio = 0.3,
    [double]$ProtectRatio = 0.5,
    [int]$ChunkFrames = 24000,
    [int]$ExtraFrames = 32768,
    [string]$ReportPath = "",
    [switch]$HighPass,
    [ValidateSet("quality", "balanced", "latency")]
    [string]$Preset = "quality"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$args = @()
if ($List) {
    $args += "--list"
} else {
    if (-not $ModelPath) { throw "ModelPath is required unless -List is used." }
    if (-not $InputDevice) { throw "InputDevice is required unless -List is used." }
    if (-not $OutputDevice) { throw "OutputDevice is required unless -List is used." }

    $args += @(
        "--model", (Resolve-Path -LiteralPath $ModelPath -ErrorAction Stop).Path,
        "--input", $InputDevice,
        "--output", $OutputDevice,
        "--seconds", $Seconds,
        "--pitch", $PitchShift,
        "--index-ratio", $IndexRatio,
        "--protect", $ProtectRatio,
        "--chunk", $ChunkFrames,
        "--extra", $ExtraFrames,
        "--preset", $Preset
    )
    if ($IndexPath) { $args += @("--index", (Resolve-Path -LiteralPath $IndexPath -ErrorAction Stop).Path) }
    if ($ContentVecPath) { $args += @("--contentvec", (Resolve-Path -LiteralPath $ContentVecPath -ErrorAction Stop).Path) }
    if ($MonitorDevice) { $args += @("--monitor", $MonitorDevice) }
    if ($HighPass) { $args += "--high-pass" }
    if ($ReportPath) {
        $reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
            [System.IO.Path]::GetFullPath($ReportPath)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ReportPath))
        }
        $reportDirectory = Split-Path -Parent $reportFullPath
        New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
        $args += @("--report", $reportFullPath)
    }
}

Push-Location $repoRoot
try {
    & cargo run --manifest-path src-tauri\Cargo.toml --features native-validation --bin native-route-validation -- @args
    if ($LASTEXITCODE -ne 0) {
        throw "Native route validation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

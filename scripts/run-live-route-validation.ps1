param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$IndexPath = "",
    [Parameter(Mandatory = $true)]
    [string]$InputDevice,
    [Parameter(Mandatory = $true)]
    [string]$OutputDevice,
    [double]$Seconds = 60,
    [int]$BlockSize = 480,
    [ValidateSet("quality", "balanced", "latency")]
    [string]$StreamingPreset = "quality",
    [int]$ChunkFrames = 0,
    [int]$ExtraFrames = 0,
    [double]$IndexRatio = 0.3,
    [switch]$UsePackageDefaults,
    [switch]$Strict,
    [string]$OutputDirectory = "outputs\reference-validation"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "engine-python\.venv\Scripts\python.exe"
$model = (Resolve-Path -LiteralPath $ModelPath -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The project Python environment is missing. Run npm run runtime:setup first."
}
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$report = Join-Path $outputPath "live-route-validation.json"
$args = @(
    "engine-python\tools\live_route_validation.py",
    "--model", $model,
    "--index-ratio", $IndexRatio,
    "--input-device", $InputDevice,
    "--output-device", $OutputDevice,
    "--seconds", $Seconds,
    "--block-size", $BlockSize,
    "--streaming-preset", $StreamingPreset,
    "--report", $report
)
if ($IndexPath) { $args += @("--index", (Resolve-Path -LiteralPath $IndexPath -ErrorAction Stop).Path) }
if ($ChunkFrames -gt 0) { $args += @("--chunk-frames", $ChunkFrames) }
if ($ExtraFrames -gt 0) { $args += @("--extra-frames", $ExtraFrames) }
if ($UsePackageDefaults) { $args += "--use-package-defaults" }
if ($Strict) { $args += "--strict" }

& $python @args
if ($LASTEXITCODE -ne 0) {
    throw "Converted route validation failed with exit code $LASTEXITCODE. See $report"
}
Write-Host "Converted route report: $report"

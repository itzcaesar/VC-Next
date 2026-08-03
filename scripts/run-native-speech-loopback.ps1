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
    [switch]$RequireSignal,
    [double]$MinimumPeak = 0.005,
    [string]$CaptureDevice = "",
    [string]$CapturePath = "",
    [int]$CaptureSampleRate = 48000,
    [int]$CaptureBlockSize = 480,
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
$nativePlayer = Join-Path $repoRoot "src-tauri\target\debug\native-fixture-playback.exe"
if (-not (Test-Path -LiteralPath $native -PathType Leaf) -or -not (Test-Path -LiteralPath $nativePlayer -PathType Leaf)) {
    Push-Location $repoRoot
    try {
        & cargo build --manifest-path src-tauri\Cargo.toml --features native-validation --bin native-route-validation --bin native-fixture-playback
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
$capture = $null
$captureSummary = $null
$capturePathFull = ""
$captureReadyFile = "$reportFullPath.capture.ready.json"
$captureStopFile = "$reportFullPath.capture.stop"
if ($CaptureDevice) {
    if ($CaptureSampleRate -le 0 -or $CaptureBlockSize -le 0) {
        throw "CaptureSampleRate and CaptureBlockSize must be positive."
    }
    $capturePathFull = if ($CapturePath) {
        if ([System.IO.Path]::IsPathRooted($CapturePath)) {
            [System.IO.Path]::GetFullPath($CapturePath)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $repoRoot $CapturePath))
        }
    } else {
        [System.IO.Path]::ChangeExtension($reportFullPath, ".captured.wav")
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $capturePathFull) -Force | Out-Null
    foreach ($marker in @($captureReadyFile, $captureStopFile)) {
        if (Test-Path -LiteralPath $marker -PathType Leaf) {
            Remove-Item -LiteralPath $marker -Force
        }
    }
}
$playerSeconds = [Math]::Max($Seconds + 20, 30)
$readyFile = "$reportFullPath.player.ready.json"
if (Test-Path -LiteralPath $readyFile -PathType Leaf) {
    Remove-Item -LiteralPath $readyFile -Force
}
# Start-Process receives a single Windows command line.  Passing an array here
# silently drops the grouping around paths/device names that contain spaces,
# which made the fixture process fail before it opened the cable.
function Quote-ProcessArgument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}
$captureStdout = Join-Path $repoRoot "outputs\native-speech-loopback-capture.stdout.txt"
$captureStderr = Join-Path $repoRoot "outputs\native-speech-loopback-capture.stderr.txt"
$effectiveIndexRatio = $IndexRatio
if (-not $IndexPath -and $effectiveIndexRatio -gt 0) {
    Write-Warning "No .index file was supplied; retrieval is disabled and IndexRatio is set to 0."
    $effectiveIndexRatio = 0
}
if ($CaptureDevice) {
    $captureSeconds = [Math]::Max($Seconds + 30, 45)
    $captureArgs = @(
        (Quote-ProcessArgument "engine-python\tools\record_device.py"),
        "--device", (Quote-ProcessArgument $CaptureDevice),
        "--output", (Quote-ProcessArgument $capturePathFull),
        "--seconds", (Quote-ProcessArgument ([string]$captureSeconds)),
        "--sample-rate", (Quote-ProcessArgument ([string]$CaptureSampleRate)),
        "--block-size", (Quote-ProcessArgument ([string]$CaptureBlockSize)),
        "--ready-file", (Quote-ProcessArgument $captureReadyFile),
        "--stop-file", (Quote-ProcessArgument $captureStopFile)
    ) -join " "
    $capture = Start-Process -FilePath $python -ArgumentList $captureArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $captureStdout -RedirectStandardError $captureStderr
}
$playerArgs = @(
    "--input", (Quote-ProcessArgument $fixture),
    "--output", (Quote-ProcessArgument $FixtureOutputDevice),
    "--seconds", (Quote-ProcessArgument ([string]$playerSeconds)),
    "--ready-file", (Quote-ProcessArgument $readyFile)
) -join " "
$playerStdout = Join-Path $repoRoot "outputs\native-speech-loopback-player.stdout.txt"
$playerStderr = Join-Path $repoRoot "outputs\native-speech-loopback-player.stderr.txt"
$player = Start-Process -FilePath $nativePlayer -ArgumentList $playerArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $playerStdout -RedirectStandardError $playerStderr

try {
    if ($capture) {
        # Open the far-end recorder before the fixture and native engine. This
        # captures startup silence as well as the converted speech window and
        # avoids confusing a late recorder with a missing output signal.
        $captureReadyDeadline = (Get-Date).AddSeconds(8)
        while (-not (Test-Path -LiteralPath $captureReadyFile -PathType Leaf)) {
            $capture.Refresh()
            if ($capture.HasExited) {
                $captureError = if (Test-Path -LiteralPath $captureStderr) { Get-Content -LiteralPath $captureStderr -Raw } else { "" }
                throw "The far-end recorder exited before opening $CaptureDevice. $captureError"
            }
            if ((Get-Date) -ge $captureReadyDeadline) {
                throw "The far-end recorder did not open $CaptureDevice within 8 seconds."
            }
            Start-Sleep -Milliseconds 100
        }
    }
    # Wait for the fixture stream to open before the native validator starts
    # its model warm-up. A fixed sleep hid player/device failures as silence.
    $readyDeadline = (Get-Date).AddSeconds(8)
    while (-not (Test-Path -LiteralPath $readyFile -PathType Leaf)) {
        if ($player.HasExited) {
            $playerError = if (Test-Path -LiteralPath $playerStderr) { Get-Content -LiteralPath $playerStderr -Raw } else { "" }
            throw "The fixture player exited before opening $FixtureOutputDevice. $playerError"
        }
        if ((Get-Date) -ge $readyDeadline) {
            throw "The fixture player did not open $FixtureOutputDevice within 8 seconds."
        }
        Start-Sleep -Milliseconds 100
    }
    $nativeArgs = @(
        "--model", $model,
        "--input", $InputDevice,
        "--output", $OutputDevice,
        "--seconds", $Seconds,
        "--pitch", $PitchShift,
        "--index-ratio", $effectiveIndexRatio,
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
    if ($capture) {
        # Let record_device finish its WAV write cleanly instead of killing the
        # process and losing the captured samples.
        New-Item -ItemType File -Path $captureStopFile -Force | Out-Null
        try { Wait-Process -Id $capture.Id -Timeout 15 -ErrorAction SilentlyContinue } catch { }
        $capture.Refresh()
        if (-not $capture.HasExited) {
            Stop-Process -Id $capture.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path -LiteralPath $reportFullPath -PathType Leaf)) {
    throw "The native speech loopback did not produce a report: $reportFullPath"
}
$captureSummaryPath = $captureStdout
if ($capture) {
    if (Test-Path -LiteralPath $captureStdout -PathType Leaf) {
        try { $captureSummary = Get-Content -LiteralPath $captureStdout -Raw | ConvertFrom-Json } catch { $captureSummary = $null }
    }
    $nativeReport = Get-Content -LiteralPath $reportFullPath -Raw | ConvertFrom-Json
    $nativeReport | Add-Member -NotePropertyName capture -NotePropertyValue ([pscustomobject]@{
        device = $CaptureDevice
        sampleRate = $CaptureSampleRate
        blockSize = $CaptureBlockSize
        path = $capturePathFull
        summary = $captureSummary
        summaryPath = $captureSummaryPath
    }) -Force
    $nativeReport | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $reportFullPath -Encoding utf8
}
if ($RequireSignal) {
    $nativeReport = Get-Content -LiteralPath $reportFullPath -Raw | ConvertFrom-Json
    # Keep this wrapper compatible with the Windows PowerShell 5.1 that is
    # present on a stock Windows installation. PowerShell 7's null-coalescing
    # operator (??) is not available there.
    $inputPeak = 0.0
    if ($nativeReport -and $nativeReport.PSObject.Properties.Name -contains "maxInputPeak" -and $null -ne $nativeReport.maxInputPeak) {
        $inputPeak = [double]$nativeReport.maxInputPeak
    }
    $outputPeak = 0.0
    if ($nativeReport -and $nativeReport.PSObject.Properties.Name -contains "maxOutputPeak" -and $null -ne $nativeReport.maxOutputPeak) {
        $outputPeak = [double]$nativeReport.maxOutputPeak
    }
    $capturePeak = $null
    if ($captureSummary) {
        $capturePeak = 0.0
        if ($captureSummary.PSObject.Properties.Name -contains "peak" -and $null -ne $captureSummary.peak) {
            $capturePeak = [double]$captureSummary.peak
        }
    }
    if ($inputPeak -lt $MinimumPeak -or $outputPeak -lt $MinimumPeak) {
        $captureDetail = if ($captureSummary) { ", far-end capture peak $capturePeak" } else { "" }
        throw "The speech route was silent (max input peak $inputPeak, max output peak $outputPeak$captureDetail; minimum $MinimumPeak). Check the fixture bus and selected endpoints. Report: $reportFullPath"
    }
    if ($capture -and $captureSummary -and $capturePeak -lt $MinimumPeak) {
        throw "The far-end capture was silent (peak $capturePeak; minimum $MinimumPeak). Native output was generated, but the downstream route did not carry it. Capture: $capturePathFull"
    }
}
Write-Host "Native speech loopback report: $reportFullPath"
if ($capture) { Write-Host "Far-end capture: $capturePathFull" }

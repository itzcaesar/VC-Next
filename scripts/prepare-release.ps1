param(
    [string]$OutputDirectory = "release\engine-python",
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$engineSource = Join-Path $repoRoot "engine-python"
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))

if (-not (Test-Path -LiteralPath (Join-Path $engineSource "vc_next_sidecar") -PathType Container)) {
    throw "The engine source package is missing: $engineSource"
}
if ($outputPath -eq [System.IO.Path]::GetFullPath($engineSource) -or $outputPath.StartsWith([System.IO.Path]::GetFullPath($engineSource) + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must not be inside engine-python."
}

function Resolve-PythonExecutable {
    param([string]$Requested)
    if ($Requested) {
        $candidate = (Resolve-Path -LiteralPath $Requested -ErrorAction Stop).Path
        return $candidate
    }
    $configured = $env:VC_NEXT_PYTHON
    if ($configured -and (Test-Path -LiteralPath $configured -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $configured).Path
    }
    $localVenv = Join-Path $engineSource ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $localVenv -PathType Leaf) {
        return (Resolve-Path -LiteralPath $localVenv).Path
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { return $pythonCommand.Source }
    throw "Python 3.11 was not found. Pass -PythonExecutable or set VC_NEXT_PYTHON."
}

$python = Resolve-PythonExecutable $PythonExecutable
$pythonInfo = & $python -c "import platform, sys; print(sys.version.split()[0]); print(platform.machine())"
if ($LASTEXITCODE -ne 0 -or $pythonInfo.Count -lt 2) {
    throw "The selected Python executable could not run: $python"
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $engineSource "vc_next_sidecar") -Destination $outputPath -Recurse -Force
foreach ($file in @("pyproject.toml", "requirements-rvc-core.txt", "requirements-rvc-optional.txt", "README.md")) {
    Copy-Item -LiteralPath (Join-Path $engineSource $file) -Destination $outputPath -Force
}
# Include the same first-run bootstrap used by the source checkout.  The script
# derives the installed resource root from its own location, so it can create
# `<install>\engine-python\.venv` without requiring the repository layout.
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "setup-runtime.ps1") -Destination (Join-Path $outputPath "setup-runtime.ps1") -Force
# Never ship interpreter caches from a developer checkout.
Get-ChildItem -LiteralPath $outputPath -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $outputPath -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue |
    Remove-Item -Force

$commit = "unknown"
try {
    $commit = (& git -C $repoRoot rev-parse --short HEAD).Trim()
} catch {
    # A source archive may not contain Git metadata.
}

$manifest = [ordered]@{
    product = "VC Next"
    engineProtocol = 1
    generatedAt = [DateTime]::UtcNow.ToString("o")
    sourceCommit = $commit
    sourceDirectory = "engine-python"
    pythonExecutableAtBuild = $python
    pythonVersion = [string]$pythonInfo[0]
    architecture = [string]$pythonInfo[1]
    virtualEnvironmentBundled = $false
    runtimeSetupScript = "setup-runtime.ps1"
    note = "The staged package contains the VC Next Python module and a first-run setup script. Run setup-runtime.ps1 from the installed app to create a per-user runtime when resources are protected, or provide VC_NEXT_PYTHON."
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $outputPath "runtime-manifest.json") -Encoding UTF8

Write-Output "Prepared VC Next engine resources at: $outputPath"
Write-Output "Python: $python ($($pythonInfo[0]), $($pythonInfo[1]))"
Write-Output "Source commit: $commit"
Write-Output "Tauri maps this staged directory to the installed engine-python resource. VC_NEXT_ENGINE_DIR remains available for portable or relocated builds."

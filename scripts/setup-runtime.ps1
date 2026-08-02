param(
    [string]$PythonExecutable = "",
    [string]$VenvDirectory = "",
    [switch]$SkipTorch,
    [switch]$SkipOptional,
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$enginePath = Join-Path $repoRoot "engine-python"
$programRoots = @(
    [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFiles),
    [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86)
) | Where-Object { $_ }

function Resolve-VenvPath {
    param([string]$Requested)

    if ($Requested) {
        if ([System.IO.Path]::IsPathRooted($Requested)) {
            return [System.IO.Path]::GetFullPath($Requested)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Requested))
    }

    foreach ($programRoot in $programRoots) {
        $rootWithSeparator = $programRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
        if ($repoRoot.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase) -and $env:LOCALAPPDATA) {
            return [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "VC Next\engine-python\.venv"))
        }
    }
    return [System.IO.Path]::GetFullPath((Join-Path $enginePath ".venv"))
}

$venvPath = Resolve-VenvPath $VenvDirectory

$enginePathWithSeparator = [System.IO.Path]::GetFullPath($enginePath).TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
$userRuntimeRoot = if ($env:LOCALAPPDATA) {
    [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "VC Next"))
} else {
    ""
}
$userRuntimeRootWithSeparator = if ($userRuntimeRoot) {
    $userRuntimeRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
} else {
    ""
}
$insideEngine = $venvPath.StartsWith($enginePathWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)
$insideUserRuntime = $userRuntimeRootWithSeparator -and $venvPath.StartsWith($userRuntimeRootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)

if (-not ($insideEngine -or $insideUserRuntime)) {
    throw "VenvDirectory must stay inside engine-python or the per-user VC Next runtime directory."
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Program,
        [string[]]$Arguments
    )
    Write-Host "==> $Label"
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Resolve-Python {
    if ($PythonExecutable) {
        $resolved = (Resolve-Path -LiteralPath $PythonExecutable -ErrorAction Stop).Path
        return @{ Program = $resolved; Prefix = @() }
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        return @{ Program = $launcher.Source; Prefix = @("-3.11") }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        $version = & $python.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
        if ($version -eq "3.11") {
            return @{ Program = $python.Source; Prefix = @() }
        }
    }
    throw "Python 3.11 was not found. Install it from python.org or pass -PythonExecutable."
}

$selectedPython = Resolve-Python
if ($ForceRecreate -and (Test-Path -LiteralPath $venvPath)) {
    Write-Host "Removing the requested virtual environment: $venvPath"
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

if (-not (Test-Path -LiteralPath (Join-Path $venvPath "Scripts\python.exe") -PathType Leaf)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $venvPath) -Force | Out-Null
    Invoke-Checked "Creating Python 3.11 virtual environment" $selectedPython.Program ($selectedPython.Prefix + @("-m", "venv", $venvPath))
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
Invoke-Checked "Upgrading pip" $venvPython @("-m", "pip", "install", "--upgrade", "pip")
$engineIsProtected = $false
foreach ($programRoot in $programRoots) {
    $rootWithSeparator = $programRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if ($repoRoot.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
        $engineIsProtected = $true
        break
    }
}
if ($engineIsProtected) {
    Invoke-Checked "Installing VC Next sidecar" $venvPython @("-m", "pip", "install", $enginePath)
} else {
    Invoke-Checked "Installing VC Next sidecar" $venvPython @("-m", "pip", "install", "-e", $enginePath)
}
Invoke-Checked "Installing core RVC dependencies" $venvPython @("-m", "pip", "install", "-r", (Join-Path $enginePath "requirements-rvc-core.txt"))

if (-not $SkipTorch) {
    Invoke-Checked "Installing verified PyTorch CUDA baseline" $venvPython @(
        "-m", "pip", "install", "torch==2.9.0", "torchaudio==2.9.0",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    )
}
if (-not $SkipOptional) {
    Invoke-Checked "Installing ONNX Runtime GPU" $venvPython @("-m", "pip", "install", "-r", (Join-Path $enginePath "requirements-rvc-optional.txt"))
}

$probeRequest = '{"protocolVersion":1,"requestId":"setup","method":"probe_runtime","params":{}}'
$probeOutput = $probeRequest | & $venvPython -m vc_next_sidecar --once
if ($LASTEXITCODE -ne 0) {
    throw "The post-install runtime probe failed with exit code $LASTEXITCODE."
}
$probe = $probeOutput | ConvertFrom-Json
if (-not $probe.ok) {
    throw "The post-install runtime probe returned an error: $($probe.error.message)"
}

$result = $probe.result
Write-Host ""
Write-Host "VC Next runtime probe"
Write-Host "  Python: $($result.python.version)"
Write-Host "  PyTorch CUDA: $($result.torchRuntime.cudaAvailable)"
Write-Host "  ONNX providers: $($result.onnxRuntime.availableProviders -join ', ')"
Write-Host "  RVC ready: $($result.readyForRvc)"
if (-not $result.readyForRvc) {
    Write-Warning (($result.blockers -join "`n") + "`nRun the probe again after fixing the listed dependency or driver issue.")
    exit 2
}

Write-Host "Runtime setup complete: $venvPath"

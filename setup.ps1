#Requires -Version 5.1
<#
.SYNOPSIS
    VoltVandal dependency setup (official doloMing stress + native NVAPI backend).

.DESCRIPTION
    Installs Python packages required by VoltVandal:
      - nvidia-ml-py, psutil, matplotlib, numpy, PyOpenGL, PyOpenGL_accelerate
      - cupy (CUDA build auto-detected or forced via -CudaVersion)
      - CUDA runtime pip packages for cuda12x (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12)

    The project now uses:
      - `nvapi_curve.py` as the only VF-curve backend
      - `gpu-cpu-stress-tests/nvidia_gpu_stress_test.py` from:
        https://github.com/doloMing/gpu-cpu-stress-tests
#>

[CmdletBinding()]
param(
    [string] $InstallDir = (Split-Path $MyInvocation.MyCommand.Path -Parent),
    [string] $CudaVersion = "",
    [switch] $SkipCupy
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Off

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg) { Write-Host "    [OK]  $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "    [!!]  $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg) { Write-Host "    [XX]  $msg" -ForegroundColor Red }
function Write-Info([string]$msg) { Write-Host "          $msg" -ForegroundColor Gray }

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║      VoltVandal  —  Dependency Setup (Official) ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host "  Install dir : $InstallDir"

Write-Step "Checking Python installation"
$pythonExe = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 8) {
                $pythonExe = $candidate
                Write-OK "Found $ver  ($candidate)"
                break
            }
        }
    } catch { }
}
if (-not $pythonExe) {
    Write-Err "Python 3.8+ not found on PATH."
    exit 1
}

Write-Step "Checking pip"
try {
    $pipVer = & $pythonExe -m pip --version 2>&1
    Write-OK $pipVer
} catch {
    Write-Err "pip not found. Run: $pythonExe -m ensurepip --upgrade"
    exit 1
}

Write-Step "Installing core pip packages"
$oldPynvml = & $pythonExe -m pip show pynvml 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Info "  Removing deprecated 'pynvml' wrapper..."
    & $pythonExe -m pip uninstall pynvml -y --quiet 2>&1 | Out-Null
}

$pipPkgs = @("nvidia-ml-py", "psutil", "matplotlib", "numpy", "PyOpenGL", "PyOpenGL_accelerate")
foreach ($pkg in $pipPkgs) {
    Write-Info "  pip install $pkg ..."
    & $pythonExe -m pip install --quiet --upgrade $pkg
    if ($LASTEXITCODE -ne 0) { Write-Warn "pip install $pkg returned non-zero ($LASTEXITCODE)" }
    else { Write-OK $pkg }
}

if (-not $SkipCupy) {
    Write-Step "Installing cupy"
    if ($CudaVersion -eq "") {
        try {
            $nvccOut = (& nvcc --version 2>&1) -join "`n"
            if ($nvccOut -match "release (\d+)\.(\d+)") {
                $cudaMaj = [int]$Matches[1]; $cudaMin = [int]$Matches[2]
                if     ($cudaMaj -ge 12)                      { $CudaVersion = "cuda12x" }
                elseif ($cudaMaj -eq 11 -and $cudaMin -ge 2) { $CudaVersion = "cuda11x" }
                else                                          { $CudaVersion = "cuda12x" }
                Write-Info "  Detected CUDA $cudaMaj.$cudaMin  ->  cupy-$CudaVersion"
            } else {
                $CudaVersion = "cuda12x"
            }
        } catch {
            $CudaVersion = "cuda12x"
            Write-Warn "nvcc not found; defaulting to cupy-cuda12x."
        }
    }

    $cupyPkg = "cupy-$CudaVersion"
    Write-Info "  pip install $cupyPkg ..."
    & $pythonExe -m pip install --quiet --upgrade $cupyPkg
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "cupy install failed; integrated stress modes will not run."
    } else {
        Write-OK "cupy ($cupyPkg)"
    }

    if ($CudaVersion -eq "cuda12x") {
        Write-Info "  Installing CUDA 12 runtime libraries via pip..."
        foreach ($pkg in @("nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12")) {
            Write-Info "    pip install $pkg ..."
            & $pythonExe -m pip install --quiet --upgrade $pkg
            if ($LASTEXITCODE -eq 0) { Write-OK "    $pkg" }
            else { Write-Warn "    pip install $pkg returned non-zero" }
        }
    }
} else {
    Write-Warn "Skipping cupy (--SkipCupy). GPU stress modes will be unavailable."
}

Write-Step "Ensuring official gpu-cpu-stress-tests repository"
$stressRepoUrl = "https://github.com/doloMing/gpu-cpu-stress-tests.git"
$stressRepoDir = Join-Path $InstallDir "gpu-cpu-stress-tests"
$gitExe = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitExe) {
    Write-Warn "git not found on PATH; cannot clone/update $stressRepoUrl"
} else {
    if (Test-Path (Join-Path $stressRepoDir ".git")) {
        Write-Info "  Updating existing repository at $stressRepoDir ..."
        & $gitExe.Source -C $stressRepoDir pull --ff-only
        if ($LASTEXITCODE -eq 0) { Write-OK "Updated gpu-cpu-stress-tests" }
        else { Write-Warn "git pull failed for $stressRepoDir" }
    } elseif (Test-Path $stressRepoDir) {
        Write-Warn "Directory exists but is not a git repo: $stressRepoDir"
        Write-Warn "Remove it manually, then re-run setup to clone official source."
    } else {
        Write-Info "  Cloning $stressRepoUrl ..."
        & $gitExe.Source clone $stressRepoUrl $stressRepoDir
        if ($LASTEXITCODE -eq 0) { Write-OK "Cloned gpu-cpu-stress-tests" }
        else { Write-Warn "git clone failed for $stressRepoUrl" }
    }
}

Write-Step "Verifying local runtime files"
foreach ($f in @("voltvandal.py", "nvapi_curve.py", "gpu-cpu-stress-tests\nvidia_gpu_stress_test.py")) {
    $p = Join-Path $InstallDir $f
    if (Test-Path $p) { Write-OK "$f found" }
    else { Write-Warn "$f missing at $p" }
}

Write-Step "Quick import checks"
$nvmlCheck = & $pythonExe -c "import pynvml; pynvml.nvmlInit(); print('pynvml OK'); pynvml.nvmlShutdown()" 2>&1
if ($LASTEXITCODE -eq 0) { Write-OK $nvmlCheck } else { Write-Warn $nvmlCheck }

Write-Host ""
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  Setup complete — quick start" -ForegroundColor Magenta
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "  python voltvandal.py dump --gpu 0 --out artifacts" -ForegroundColor White
Write-Host ""
Write-Host "  python voltvandal.py run --gpu 0 --out artifacts ``" -ForegroundColor White
Write-Host "      --mode uv --bin-min-mv 850 --bin-max-mv 950 --step-mv 5 ``" -ForegroundColor White
Write-Host "      --stress-seconds 120 --doloming-mode ray" -ForegroundColor White
Write-Host ""

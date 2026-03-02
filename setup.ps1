#Requires -Version 5.1
<#
.SYNOPSIS
    VoltVandal setup — installs all Python dependencies and third-party stress tools.

.DESCRIPTION
    This script automates the full VoltVandal dependency setup on Windows:

      1. Python dependency check  (>= 3.7 required)
      2. pip packages             nvidia-ml-py, psutil, numpy, PyOpenGL
      3. cupy (CUDA-accelerated)  auto-detects CUDA version, or use -CudaVersion
      4. doloMing stress tool     cloned from GitHub; stress.py shim created

    VF-curve backend (nvapi_curve.py) is pure Python — no extra binary needed.
    Run from an elevated (Administrator) PowerShell terminal because NVAPI
    VF-curve operations require elevated privileges at runtime.

.PARAMETER InstallDir
    Root directory where voltvandal.py lives.
    Defaults to the folder that contains this script.

.PARAMETER CudaVersion
    cupy CUDA suffix, e.g. "cuda12x" (CUDA 12.x), "cuda11x" (CUDA 11.2+).
    Run  nvcc --version  to find your installed CUDA version.
    If omitted the script will detect automatically; if detection fails
    it falls back to "cuda12x" (broad CUDA 12 wheel).

.PARAMETER SkipCupy
    Skip cupy installation entirely (e.g. if you only want ray/matrix modes
    that don't need cupy, or you plan to install it manually).

.PARAMETER SkipDoloMing
    Skip cloning and configuring doloMing.

.EXAMPLE
    # Typical — run from elevated PowerShell inside the VoltVandal folder:
    .\setup.ps1

.EXAMPLE
    # Explicit CUDA 11.x build of cupy:
    .\setup.ps1 -CudaVersion cuda11x

.EXAMPLE
    # Skip cupy (e.g. CUDA not installed yet):
    .\setup.ps1 -SkipCupy

.LINK
    nvapi_curve.py source : https://github.com/buswedg/nvapi-cmd  (reference C++)
    doloMing stress tool  : https://github.com/doloMing/gpu-cpu-stress-tests
    nvidia-ml-py          : https://pypi.org/project/nvidia-ml-py/
    cupy install guide    : https://docs.cupy.dev/en/stable/install.html
#>

[CmdletBinding()]
param(
    [string] $InstallDir   = (Split-Path $MyInvocation.MyCommand.Path -Parent),
    [string] $CudaVersion  = "",
    [switch] $SkipCupy,
    [switch] $SkipDoloMing
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Off   # keep loose for compatibility with PS 5.1

# ─── Colour helpers ──────────────────────────────────────────────────────────
function Write-Step  ([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan   }
function Write-OK    ([string]$msg) { Write-Host "    [OK]  $msg" -ForegroundColor Green  }
function Write-Warn  ([string]$msg) { Write-Host "    [!!]  $msg" -ForegroundColor Yellow }
function Write-Err   ([string]$msg) { Write-Host "    [XX]  $msg" -ForegroundColor Red    }
function Write-Info  ([string]$msg) { Write-Host "          $msg" -ForegroundColor Gray   }

# ─── Banner ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║          VoltVandal  —  Dependency Setup         ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host "  Install dir : $InstallDir"

# ─── 0. Administrator check ───────────────────────────────────────────────────
Write-Step "Checking privilege level"
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if ($isAdmin) {
    Write-OK "Running as Administrator."
} else {
    Write-Warn "Not running as Administrator."
    Write-Info "nvapi-cmd.exe requires Admin rights at runtime."
    Write-Info "Re-run this script from an elevated terminal if tool calls fail."
}

# ─── 1. Python check ─────────────────────────────────────────────────────────
Write-Step "Checking Python installation"

$pythonExe = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 7) {
                $pythonExe = $candidate
                Write-OK "Found $ver  ($candidate)"
                break
            } else {
                Write-Warn "Found $ver but VoltVandal requires Python >= 3.7"
            }
        }
    } catch { }
}

if (-not $pythonExe) {
    Write-Err "Python 3.7+ not found on PATH."
    Write-Info "Download from https://www.python.org/downloads/"
    exit 1
}

# ─── 2. pip check ────────────────────────────────────────────────────────────
Write-Step "Checking pip"
try {
    $pipVer = & $pythonExe -m pip --version 2>&1
    Write-OK $pipVer
} catch {
    Write-Err "pip not found. Install it:"
    Write-Info "  $pythonExe -m ensurepip --upgrade"
    exit 1
}

# ─── 3. Core pip packages ────────────────────────────────────────────────────
Write-Step "Installing pip packages (nvidia-ml-py, psutil, matplotlib, numpy, PyOpenGL)"

# Remove deprecated pynvml wrapper if present (nvidia-ml-py replaces it)
$oldPynvml = & $pythonExe -m pip show pynvml 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Info "  Removing deprecated 'pynvml' package (replaced by nvidia-ml-py)..."
    & $pythonExe -m pip uninstall pynvml -y --quiet 2>&1 | Out-Null
}

$pipPkgs = @("nvidia-ml-py", "psutil", "matplotlib", "numpy", "PyOpenGL", "PyOpenGL_accelerate")
foreach ($pkg in $pipPkgs) {
    Write-Info "  pip install $pkg ..."
    & $pythonExe -m pip install --quiet --upgrade $pkg
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "pip install $pkg returned non-zero ($LASTEXITCODE) — check output above."
    } else {
        Write-OK $pkg
    }
}

# ─── 4. cupy (CUDA-accelerated NumPy) ────────────────────────────────────────
if (-not $SkipCupy) {
    Write-Step "Installing cupy (CUDA-accelerated NumPy for doloMing)"

    # Auto-detect CUDA version via nvcc if not supplied
    if ($CudaVersion -eq "") {
        try {
            $nvccOut = (& nvcc --version 2>&1) -join "`n"
            if ($nvccOut -match "release (\d+)\.(\d+)") {
                $cudaMaj = [int]$Matches[1]; $cudaMin = [int]$Matches[2]
                # Map to cupy wheel suffix (PyPI names: cupy-cuda12x, cupy-cuda11x, etc.)
                if     ($cudaMaj -ge 12)                        { $CudaVersion = "cuda12x" }
                elseif ($cudaMaj -eq 11 -and $cudaMin -ge 2)   { $CudaVersion = "cuda11x" }
                elseif ($cudaMaj -eq 11)                        { $CudaVersion = "cuda111" }
                elseif ($cudaMaj -eq 10 -and $cudaMin -ge 2)   { $CudaVersion = "cuda102" }
                else                                             { $CudaVersion = "cuda12x" }
                Write-Info "  Detected CUDA $cudaMaj.$cudaMin  →  cupy-$CudaVersion"
            } else {
                Write-Warn "Could not parse CUDA version from nvcc output; defaulting to cupy-cuda12x."
                $CudaVersion = "cuda12x"
            }
        } catch {
            Write-Warn "nvcc not found; defaulting to cupy-cuda12x. Use -CudaVersion cudaXXX to override."
            $CudaVersion = "cuda12x"
        }
    }

    $cupyPkg = "cupy-$CudaVersion"
    Write-Info "  pip install $cupyPkg ..."
    & $pythonExe -m pip install --quiet --upgrade $cupyPkg
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "cupy install failed — doloMing may not work for GPU stress modes."
        Write-Info "  See https://docs.cupy.dev/en/stable/install.html for manual steps."
        Write-Info "  Or re-run:  .\setup.ps1 -CudaVersion cuda12x"
    } else {
        Write-OK "cupy ($cupyPkg)"
    }

    # ── 4b. CUDA runtime DLLs (cublas, cudart) via pip ───────────────────────
    # cupy-cudaXXX ships only the Python bindings; the actual CUDA compute
    # libraries (cublas64_12.dll, cudart64_12.dll …) must come from either the
    # CUDA Toolkit or these nvidia-* pip packages.  cuda-pathfinder (already a
    # cupy dependency) will call os.add_dll_directory() to register them so
    # Python 3.8+ finds them without any PATH change required.
    #
    # Only the cuda12x packages exist on PyPI today; for older CUDA the user
    # needs the full Toolkit — we warn but don't abort.
    if ($CudaVersion -eq "cuda12x") {
        Write-Info "  Installing CUDA 12 runtime libraries (cublas, cudart) via pip..."
        $cudaRtPkgs = @("nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12")
        foreach ($pkg in $cudaRtPkgs) {
            Write-Info "    pip install $pkg ..."
            & $pythonExe -m pip install --quiet --upgrade $pkg
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "    pip install $pkg returned non-zero — check output above."
            } else {
                Write-OK "    $pkg"
            }
        }
    } else {
        Write-Warn "  CUDA runtime pip packages only available for cuda12x."
        Write-Info "  For CUDA 11.x / 10.x ensure the CUDA Toolkit is installed and on PATH."
    }

    # ── 4c. Functional cupy verification (exercises cublas DLL load) ─────────
    Write-Info "  Verifying cupy can perform GPU matrix ops (tests cublas DLL load)..."
    $cupyVerify = & $pythonExe -c "import cupy as cp, sys; a=cp.ones((64,64),dtype=cp.float32); cp.dot(a,a); print('cupy dot OK')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK "cupy GPU matrix ops verified — cublas loads correctly."
    } else {
        Write-Warn "cupy matrix test failed: $cupyVerify"
        Write-Warn "doloMing 'matrix' mode will not function until this is resolved."
        Write-Info "  Most likely fix: install CUDA Toolkit 12.x from:"
        Write-Info "    https://developer.nvidia.com/cuda-downloads"
        Write-Info "  (Windows → x86_64 → exe local; adds CUDA bin to PATH automatically)"
    }
} else {
    Write-Warn "Skipping cupy (--SkipCupy). doloMing GPU modes will not work without it."
}

# ─── 5. doloMing ─────────────────────────────────────────────────────────────
if (-not $SkipDoloMing) {
    Write-Step "Setting up doloMing GPU stress tool"

    $doloDir = Join-Path $InstallDir "doloMing"

    # Try git clone first, fall back to zip download
    if (Test-Path $doloDir) {
        Write-Info "  doloMing directory already exists; pulling latest..."
        try {
            Push-Location $doloDir
            & git pull --quiet origin main 2>&1 | Out-Null
            Pop-Location
            Write-OK "doloMing updated via git."
        } catch {
            Pop-Location -ErrorAction SilentlyContinue
            Write-Warn "git pull failed; keeping existing directory as-is."
        }
    } else {
        $gitOk = $false

        # Check if git is available
        try {
            $null = & git --version 2>&1
            $gitOk = $true
        } catch { }

        if ($gitOk) {
            Write-Info "  Cloning doloMing/gpu-cpu-stress-tests ..."
            & git clone --quiet "https://github.com/doloMing/gpu-cpu-stress-tests.git" $doloDir
            if ($LASTEXITCODE -eq 0) {
                Write-OK "Cloned via git."
            } else {
                $gitOk = $false
                Write-Warn "git clone failed; trying zip download..."
            }
        }

        if (-not $gitOk) {
            Write-Info "  Downloading doloMing zip from GitHub..."
            $zipUrl  = "https://github.com/doloMing/gpu-cpu-stress-tests/archive/refs/heads/main.zip"
            $zipPath = Join-Path $env:TEMP "doloming_main.zip"
            $unzipTo = Join-Path $env:TEMP "doloming_unzip"

            try {
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
                Expand-Archive -Path $zipPath -DestinationPath $unzipTo -Force
                # GitHub zips unpack as <repo>-<branch>/
                $inner = Get-ChildItem $unzipTo -Directory | Select-Object -First 1
                if ($inner) {
                    Move-Item $inner.FullName $doloDir
                    Write-OK "Downloaded and extracted doloMing."
                } else {
                    throw "Could not find extracted folder."
                }
            } catch {
                Write-Err "Could not download doloMing: $_"
                Write-Info "  Download manually from:"
                Write-Info "  https://github.com/doloMing/gpu-cpu-stress-tests"
                Write-Info "  Extract into: $doloDir"
                $SkipDoloMing = $true
            } finally {
                Remove-Item $zipPath   -ErrorAction SilentlyContinue
                Remove-Item $unzipTo   -ErrorAction SilentlyContinue -Recurse
            }
        }
    }

    # ── 5a. Create stress.py compatibility shim ──────────────────────────────
    if (-not $SkipDoloMing) {
        $shimPath = Join-Path $doloDir "stress.py"
        Write-Info "  Writing VoltVandal compatibility shim -> stress.py ..."

        # The shim translates VoltVandal's two calling conventions:
        #   stress.py --mode ray --seconds 90    (VoltVandal primary)
        #   stress.py ray 90                     (VoltVandal fallback)
        # into doloMing's actual interface:
        #   nvidia_gpu_stress_test.py -m ray -d 90
        $shimContent = @'
#!/usr/bin/env python3
"""
VoltVandal <-> doloMing compatibility shim.

VoltVandal calls this script as:
    stress.py --mode <mode> --seconds <secs>    (primary attempt)
    stress.py <mode> <secs>                      (fallback attempt)

This shim translates both forms to doloMing's actual CLI:
    nvidia_gpu_stress_test.py -m <mode> -d <secs>

Supported modes (from doloMing): matrix, simple, ray, frequency-max
"""
import subprocess
import sys
from pathlib import Path


def _parse_args(argv):
    mode = "ray"
    duration = 60
    rest = list(argv)

    # --mode / --seconds (VoltVandal primary)
    for flag, attr in (("--mode", "mode"), ("--seconds", "duration")):
        if flag in rest:
            idx = rest.index(flag)
            if idx + 1 < len(rest):
                val = rest[idx + 1]
                if attr == "mode":
                    mode = val
                else:
                    try:
                        duration = int(val)
                    except ValueError:
                        pass
                rest = rest[:idx] + rest[idx + 2:]

    # Positional fallback: stress.py <mode> <seconds>
    positional = [a for a in rest if not a.startswith("-")]
    if positional:
        mode = positional[0]
        rest.remove(positional[0])
    if positional and len(positional) > 1:
        try:
            duration = int(positional[1])
            rest.remove(positional[1])
        except ValueError:
            pass

    return mode, duration, rest


def main() -> int:
    mode, duration, extra = _parse_args(sys.argv[1:])
    script = Path(__file__).resolve().parent / "nvidia_gpu_stress_test.py"

    if not script.exists():
        print(
            f"ERROR: nvidia_gpu_stress_test.py not found at {script}",
            file=sys.stderr,
        )
        return 1

    cmd = [sys.executable, str(script), "-m", mode, "-d", str(duration)] + extra
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
'@
        Set-Content -Path $shimPath -Value $shimContent -Encoding UTF8
        Write-OK "stress.py shim written to $shimPath"

        # ── 5b. Verify nvidia_gpu_stress_test.py is present ─────────────────
        $mainScript = Join-Path $doloDir "nvidia_gpu_stress_test.py"
        if (Test-Path $mainScript) {
            Write-OK "nvidia_gpu_stress_test.py found."
        } else {
            Write-Warn "nvidia_gpu_stress_test.py not found in $doloDir"
            Write-Info "  The clone/download may have placed files differently."
            Write-Info "  Expected path: $mainScript"
        }
    }
} else {
    Write-Warn "Skipping doloMing setup."
}

# ─── 6. Verify nvapi_curve.py (native Python VF-curve backend) ───────────────
Write-Step "Verifying nvapi_curve.py  (native Python NVAPI backend)"

$nvapiCurvePath = Join-Path $InstallDir "nvapi_curve.py"
if (Test-Path $nvapiCurvePath) {
    Write-OK "nvapi_curve.py found — no nvapi-cmd.exe binary needed."
    Write-Info "  VoltVandal will use the native Python backend automatically."
} else {
    Write-Warn "nvapi_curve.py not found at expected location: $nvapiCurvePath"
    Write-Info "  Make sure you cloned the full VoltVandal repository."
}

# Keep nvapi-cmd.exe as an optional legacy fallback (pass --nvapi-cmd if needed)
$nvapiExePath = Join-Path $InstallDir "nvapi-cmd.exe"
if (Test-Path $nvapiExePath) {
    Write-Info "  Legacy nvapi-cmd.exe also present (subprocess fallback available)."
}

# ─── 7. Quick sanity-import of pynvml ────────────────────────────────────────
Write-Step "Verifying pynvml import"
$pynvmlCheck = & $pythonExe -c "import pynvml; pynvml.nvmlInit(); print('pynvml (nvidia-ml-py) OK'); pynvml.nvmlShutdown()" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-OK $pynvmlCheck
} else {
    Write-Warn "pynvml import failed: $pynvmlCheck"
}

# ─── 8. Final summary ────────────────────────────────────────────────────────
$doloRelPath = ".\doloMing\stress.py"

Write-Host ""
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  Setup complete — quick-start reference" -ForegroundColor Magenta
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "  # 1. Dump stock VF curve (uses nvapi_curve.py automatically):"
Write-Host "  python voltvandal.py dump --gpu 0 --out artifacts" -ForegroundColor White
Write-Host ""
Write-Host "  # 2. Run UV sweep, stress with doloMing ray mode:"
Write-Host "  python voltvandal.py run ``" -ForegroundColor White
Write-Host "      --gpu 0 --out artifacts ``" -ForegroundColor White
Write-Host "      --mode uv --bin-min-mv 850 --bin-max-mv 950 --step-mv 5 ``" -ForegroundColor White
Write-Host "      --stress-seconds 120 --stress-timeout 180 ``" -ForegroundColor White
Write-Host "      --doloming $doloRelPath --doloming-mode ray ``" -ForegroundColor White
Write-Host "      --temp-limit-c 83 --power-limit-w 350" -ForegroundColor White
Write-Host ""
Write-Host "  # 3. Resume from checkpoint:"
Write-Host "  python voltvandal.py resume --out artifacts" -ForegroundColor White
Write-Host ""
Write-Host "  # 4. Revert GPU to last known-good curve:"
Write-Host "  python voltvandal.py restore --gpu 0 --out artifacts" -ForegroundColor White
Write-Host ""

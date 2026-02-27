#Requires -Version 5.1
<#
.SYNOPSIS
    VoltVandal setup — installs all Python dependencies and third-party stress tools.

.DESCRIPTION
    This script automates the full VoltVandal dependency setup on Windows:

      1. Python dependency check  (>= 3.7 required)
      2. pip packages             pynvml, psutil, numpy, PyOpenGL
      3. cupy (CUDA-accelerated)  auto-detects CUDA version, or use -CudaVersion
      4. doloMing stress tool     cloned from GitHub; stress.py shim created
      5. nvapi-cmd.exe            downloaded from buswedg/nvapi-cmd (GitHub)

    Run from an elevated (Administrator) PowerShell terminal because
    nvapi-cmd.exe requires Admin rights at runtime.

.PARAMETER InstallDir
    Root directory where voltvandal.py lives.
    Defaults to the folder that contains this script.

.PARAMETER CudaVersion
    cupy CUDA suffix, e.g. "cu121" (CUDA 12.1), "cu118" (CUDA 11.8).
    Run  nvcc --version  to find your installed CUDA version.
    If omitted the script will detect automatically; if detection fails
    it falls back to "cu12x" (broad CUDA 12 wheel).

.PARAMETER SkipCupy
    Skip cupy installation entirely (e.g. if you only want ray/matrix modes
    that don't need cupy, or you plan to install it manually).

.PARAMETER SkipDoloMing
    Skip cloning and configuring doloMing.

.EXAMPLE
    # Typical — run from elevated PowerShell inside the VoltVandal folder:
    .\setup.ps1

.EXAMPLE
    # Explicit CUDA 11.8 build of cupy:
    .\setup.ps1 -CudaVersion cu118

.EXAMPLE
    # Skip cupy (e.g. CUDA not installed yet):
    .\setup.ps1 -SkipCupy

.LINK
    nvapi-cmd             : https://github.com/buswedg/nvapi-cmd
    doloMing stress tool  : https://github.com/doloMing/gpu-cpu-stress-tests
    pynvml                : https://pypi.org/project/pynvml/
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
Write-Step "Installing pip packages (pynvml, psutil, numpy, PyOpenGL)"

$pipPkgs = @("pynvml", "psutil", "numpy", "PyOpenGL", "PyOpenGL_accelerate")
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
            $nvccOut = & nvcc --version 2>&1
            if ($nvccOut -match "release (\d+)\.(\d+)") {
                $cudaMaj = [int]$Matches[1]; $cudaMin = [int]$Matches[2]
                # Map to cupy wheel suffix
                if     ($cudaMaj -ge 12)                        { $CudaVersion = "cu12x" }
                elseif ($cudaMaj -eq 11 -and $cudaMin -ge 8)   { $CudaVersion = "cu118" }
                elseif ($cudaMaj -eq 11 -and $cudaMin -ge 6)   { $CudaVersion = "cu116" }
                elseif ($cudaMaj -eq 11 -and $cudaMin -ge 2)   { $CudaVersion = "cu112" }
                elseif ($cudaMaj -eq 11)                        { $CudaVersion = "cu111" }
                elseif ($cudaMaj -eq 10 -and $cudaMin -ge 2)   { $CudaVersion = "cu102" }
                elseif ($cudaMaj -eq 10)                        { $CudaVersion = "cu101" }
                else                                             { $CudaVersion = "cu12x" }
                Write-Info "  Detected CUDA $cudaMaj.$cudaMin  →  cupy-$CudaVersion"
            }
        } catch {
            Write-Warn "nvcc not found; defaulting to cupy-cuda12x. Use -CudaVersion cuXXX to override."
            $CudaVersion = "cu12x"
        }
    }

    $cupyPkg = "cupy-cuda$CudaVersion"
    Write-Info "  pip install $cupyPkg ..."
    & $pythonExe -m pip install --quiet --upgrade $cupyPkg
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "cupy install failed — doloMing may not work for GPU stress modes."
        Write-Info "  See https://docs.cupy.dev/en/stable/install.html for manual steps."
        Write-Info "  Or re-run:  .\setup.ps1 -CudaVersion cu<your-version>"
    } else {
        Write-OK "cupy ($cupyPkg)"
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

# ─── 6. nvapi-cmd.exe ─────────────────────────────────────────────────────────
# Source: https://github.com/buswedg/nvapi-cmd
# The repo ships a pre-compiled nvapi-cmd.exe in its root directory.
Write-Step "Installing nvapi-cmd.exe  (buswedg/nvapi-cmd)"

$nvapiPath  = Join-Path $InstallDir "nvapi-cmd.exe"
$nvapiRepo  = "https://github.com/buswedg/nvapi-cmd.git"
$nvapiRawExe = "https://raw.githubusercontent.com/buswedg/nvapi-cmd/master/nvapi-cmd.exe"

if (Test-Path $nvapiPath) {
    Write-OK "nvapi-cmd.exe already present — skipping download."
} else {
    $downloaded = $false

    # ── Attempt 1: git clone then copy exe ───────────────────────────────────
    try {
        $null = & git --version 2>&1
        $gitAvailable = $true
    } catch {
        $gitAvailable = $false
    }

    if ($gitAvailable) {
        $nvapiCloneDir = Join-Path $env:TEMP "nvapi-cmd-clone"
        Write-Info "  Cloning buswedg/nvapi-cmd ..."
        try {
            Remove-Item $nvapiCloneDir -Recurse -Force -ErrorAction SilentlyContinue
            & git clone --quiet --depth 1 $nvapiRepo $nvapiCloneDir 2>&1 | Out-Null
            $clonedExe = Join-Path $nvapiCloneDir "nvapi-cmd.exe"
            if (Test-Path $clonedExe) {
                Copy-Item $clonedExe $nvapiPath
                Write-OK "nvapi-cmd.exe cloned and placed at $nvapiPath"
                $downloaded = $true
            } else {
                Write-Warn "Clone succeeded but nvapi-cmd.exe not found in repo root."
            }
        } catch {
            Write-Warn "git clone failed: $_"
        } finally {
            Remove-Item $nvapiCloneDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    # ── Attempt 2: direct raw download of the compiled exe ───────────────────
    if (-not $downloaded) {
        Write-Info "  Downloading nvapi-cmd.exe directly from GitHub raw ..."
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $nvapiRawExe -OutFile $nvapiPath -UseBasicParsing
            if (Test-Path $nvapiPath) {
                Write-OK "nvapi-cmd.exe downloaded to $nvapiPath"
                $downloaded = $true
            }
        } catch {
            Write-Err "Direct download also failed: $_"
        }
    }

    if (-not $downloaded) {
        Write-Warn "Could not obtain nvapi-cmd.exe automatically."
        Write-Info "  Download manually from: https://github.com/buswedg/nvapi-cmd"
        Write-Info "  Place nvapi-cmd.exe in:  $InstallDir"
    }
}

# ─── 7. Quick sanity-import of pynvml ────────────────────────────────────────
Write-Step "Verifying pynvml import"
$pynvmlCheck = & $pythonExe -c "import pynvml; print('pynvml', pynvml.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-OK $pynvmlCheck
} else {
    Write-Warn "pynvml import failed: $pynvmlCheck"
}

# ─── 8. Final summary ────────────────────────────────────────────────────────
$doloRelPath  = ".\doloMing\stress.py"
$nvapiRelPath = ".\nvapi-cmd.exe"

Write-Host ""
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  Setup complete — quick-start reference" -ForegroundColor Magenta
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "  # 1. Dump stock VF curve (requires nvapi-cmd.exe):"
Write-Host "  python voltvandal.py dump ``" -ForegroundColor White
Write-Host "      --gpu 0 --nvapi-cmd $nvapiRelPath --out artifacts" -ForegroundColor White
Write-Host ""
Write-Host "  # 2. Run UV sweep, stress with doloMing ray mode:"
Write-Host "  python voltvandal.py run ``" -ForegroundColor White
Write-Host "      --gpu 0 --nvapi-cmd $nvapiRelPath --out artifacts ``" -ForegroundColor White
Write-Host "      --mode uv --bin-min-mv 850 --bin-max-mv 950 --step-mv 5 ``" -ForegroundColor White
Write-Host "      --stress-seconds 120 --stress-timeout 180 ``" -ForegroundColor White
Write-Host "      --doloming $doloRelPath --doloming-mode ray ``" -ForegroundColor White
Write-Host "      --temp-limit-c 83 --power-limit-w 350" -ForegroundColor White
Write-Host ""
Write-Host "  # 3. Resume from checkpoint:"
Write-Host "  python voltvandal.py resume --out artifacts" -ForegroundColor White
Write-Host ""
Write-Host "  # 4. Revert GPU to last known-good curve:"
Write-Host "  python voltvandal.py restore --gpu 0 --nvapi-cmd $nvapiRelPath --out artifacts" -ForegroundColor White
Write-Host ""

if (-not (Test-Path $nvapiPath)) {
    Write-Host "  [!!] nvapi-cmd.exe missing — download from:" -ForegroundColor Yellow
    Write-Host "       https://github.com/buswedg/nvapi-cmd" -ForegroundColor Yellow
    Write-Host "       Place nvapi-cmd.exe in: $InstallDir" -ForegroundColor Yellow
}
Write-Host ""

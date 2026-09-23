<#
.SYNOPSIS
    One-shot Windows setup for Flipster, then starts the app.

.DESCRIPTION
    1. Finds Python >= 3.10 and creates .venv
    2. Builds the native engine (C++/OpenMP, plus CUDA when an NVIDIA GPU and the
       CUDA Toolkit are present) with Visual Studio's C++ compiler. Without a
       compiler, or if the build fails, it falls back to the pure-NumPy engine so
       the app still runs.
    3. Builds the React app (npm ci + npm run build)
    4. Runs the Python test suite
    5. Starts the server and opens http://127.0.0.1:8000

    Everything is also written to data\setup\setup.log.

.EXAMPLE
    scripts\windows\setup.cmd              (double-click)
    powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 -NoRun
#>
param(
    [switch]$NoRun,          # set up only
    [switch]$SkipTests,
    [switch]$ReferenceOnly,  # skip the C++ build and use the NumPy engine
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
Set-Content -Path $Log -Value "Flipster setup $(Get-Date -Format s)" -Encoding UTF8
Set-Location $Root

# ------------------------------------------------------------------ system info
Write-Step 'Checking this PC'
Write-Log ("   Windows: " + [System.Environment]::OSVersion.VersionString)
try {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    Write-Log ("   CPU: " + $cpu.Name.Trim() + " (" + $cpu.NumberOfLogicalProcessors + " threads)")
    Get-CimInstance Win32_VideoController | ForEach-Object { Write-Log ("   GPU: " + $_.Name) }
} catch { }

# ------------------------------------------------------------------ python
function Find-Python {
    $candidates = @()
    foreach ($name in @('py', 'python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $candidates += $cmd.Source }
    }
    $candidates += @('C:\Python314\python.exe', 'C:\Python313\python.exe', 'C:\Python312\python.exe', 'C:\Python311\python.exe')
    foreach ($exe in $candidates) {
        if (-not (Test-Path $exe)) { continue }
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $ver = & $exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        $ErrorActionPreference = $old
        if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.(\d+)$' -and [int]$Matches[1] -ge 10) {
            return @{ Exe = $exe; Version = $ver }
        }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Log 'Python 3.10+ was not found. Install it from https://www.python.org/downloads/ (tick "Add to PATH") and run this again.' 'Red'
    exit 1
}
Write-Log ("   Python " + $py.Version + " at " + $py.Exe)

$node = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($node) { Write-Log ("   npm at " + $node.Source) } else { Write-Log '   npm not found (install Node.js 20+ from https://nodejs.org to build the web app)' 'Yellow' }

# Visual Studio C++ tools
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = $null
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if ($vsPath) {
    Write-Log "   Visual Studio C++ tools: $vsPath"
} else {
    Write-Log ('   Visual Studio C++ tools: not found. For the fast C++ engine, open Visual Studio Installer > Modify > ' +
               'tick "Desktop development with C++", then run this again.') 'Yellow'
}

# CUDA
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc -and $env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
    $nvcc = Get-Item (Join-Path $env:CUDA_PATH 'bin\nvcc.exe')
}
$useCuda = [bool]($nvsmi -and $nvcc -and $vsPath)
if ($nvsmi) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $gpu = & nvidia-smi '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader' 2>$null
    $ErrorActionPreference = $old
    Write-Log "   NVIDIA GPU: $gpu"
} else {
    Write-Log '   NVIDIA GPU: none detected (the CUDA backend needs an NVIDIA card; the C++ CPU engine will be used)' 'Yellow'
}
if ($nvsmi -and -not $nvcc) {
    Write-Log '   CUDA Toolkit (nvcc) not found - install it from https://developer.nvidia.com/cuda-downloads to build the GPU engine.' 'Yellow'
}

# ------------------------------------------------------------------ venv
Write-Step 'Creating the Python environment (.venv)'
if (-not (Test-Path $VenvPy)) {
    if ((Invoke-Logged $py.Exe @('-m', 'venv', '.venv')) -ne 0) { Write-Log 'Could not create .venv' 'Red'; exit 1 }
}
Invoke-Logged $VenvPy @('-m', 'pip', 'install', '--upgrade', 'pip') | Out-Null

# ------------------------------------------------------------------ native engine
$native = $false
if ($vsPath -and -not $ReferenceOnly) {
    Write-Step ('Building the native engine (C++/OpenMP' + $(if ($useCuda) { ' + CUDA' } else { '' }) + ') - first build takes a few minutes')
    # Load the MSVC environment so CMake + Ninja can find cl.exe (and nvcc can find its host compiler).
    # (via a temp .cmd file: quoting a path with spaces through cmd /c from PowerShell 5.1 is unreliable)
    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    $envCmd = Join-Path $env:TEMP 'flipster-vcvars.cmd'
    Set-Content -Path $envCmd -Value "@call `"$vcvars`" >nul 2>&1`r`n@set" -Encoding ASCII
    & cmd.exe /c $envCmd | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path ("env:" + $Matches[1]) -Value $Matches[2] }
    }
    Remove-Item $envCmd -ErrorAction SilentlyContinue
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { Write-Log '   warning: cl.exe still not on PATH after vcvars64' 'Yellow' }
    $env:CMAKE_GENERATOR = 'Ninja'
    $env:FLIPSTER_ENABLE_CUDA = $(if ($useCuda) { 'ON' } else { 'OFF' })
    $code = Invoke-Logged $VenvPy @('-m', 'pip', 'install', '-e', '.[server,dev]')
    if ($code -eq 0) {
        $native = $true
        $site = (& $VenvPy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
        Remove-Item (Join-Path $site 'flipster-src.pth') -ErrorAction SilentlyContinue  # left over from a fallback run
    } else {
        Write-Log 'Native build failed (details above and in data\setup\setup.log). Falling back to the NumPy engine.' 'Yellow'
    }
}

if (-not $native) {
    Write-Step 'Installing Python packages (NumPy reference engine only)'
    $pkgs = @('numpy', 'opencv-python-headless', 'pillow', 'fastapi', 'uvicorn[standard]', 'python-multipart',
              'imageio', 'imageio-ffmpeg', 'pytest', 'httpx')
    if ((Invoke-Logged $VenvPy (@('-m', 'pip', 'install') + $pkgs)) -ne 0) { Write-Log 'pip install failed' 'Red'; exit 1 }
    # Make the in-repo package importable without compiling it.
    $site = (& $VenvPy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
    Set-Content -Path (Join-Path $site 'flipster-src.pth') -Value (Join-Path $Root 'python') -Encoding ASCII
}

Write-Step 'Checking the engine'
Invoke-Logged $VenvPy @('-c', 'import flipster, json; print(json.dumps(flipster.device_info()))') | Out-Null

# ------------------------------------------------------------------ web app
if ($node) {
    Write-Step 'Building the web app'
    $web = Join-Path $Root 'web'
    if ((Invoke-Logged $node.Source @('ci', '--no-audit', '--no-fund') $web) -ne 0) { Write-Log 'npm ci failed' 'Red'; exit 1 }
    if ((Invoke-Logged $node.Source @('run', 'build') $web) -ne 0) { Write-Log 'web build failed' 'Red'; exit 1 }
}

# ------------------------------------------------------------------ tests
if (-not $SkipTests) {
    Write-Step 'Running the Python test suite'
    $code = Invoke-Logged $VenvPy @('-m', 'pytest', '-q', '-p', 'no:cacheprovider')
    if ($code -eq 0) { Write-Log '   all tests passed' 'Green' } else { Write-Log '   some tests failed (see above) - continuing' 'Yellow' }
}

Write-Log ''
Write-Log ('Setup complete. Engine: ' + $(if ($native) { if ($useCuda) { 'CUDA + C++' } else { 'C++ (OpenMP)' } } else { 'NumPy reference' })) 'Green'
Write-Log 'Next time, start it with scripts\windows\run.cmd'

if (-not $NoRun) { Start-Flipster -Port $Port }

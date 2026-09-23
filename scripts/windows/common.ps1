# Shared helpers for setup.ps1 / run.ps1 (Windows PowerShell 5.1 compatible).

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LogDir = Join-Path $Root 'data\setup'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'setup.log'
$VenvPy = Join-Path $Root '.venv\Scripts\python.exe'

function Write-Log([string]$Text, [string]$Color = '') {
    Add-Content -Path $Log -Value $Text -Encoding UTF8
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
}

function Write-Step([string]$Title) {
    Write-Log ''
    Write-Log "==> $Title" 'Cyan'
}

# Run a native command, stream its output to the console and the log, return its exit code.
function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$WorkDir = $Root) {
    Write-Log ("   $ " + $Exe + ' ' + ($Arguments -join ' ')) 'DarkGray'
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Push-Location $WorkDir
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { Write-Log ("   " + "$_") }
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $old
    }
    return $code
}

function Start-Flipster([int]$Port = 8000) {
    if (-not (Test-Path $VenvPy)) {
        Write-Log 'No .venv found - run scripts\windows\setup.cmd first.' 'Red'
        return
    }
    $url = "http://127.0.0.1:$Port"
    Write-Step "Starting Flipster on $url"
    $uvArgs = @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'server', '--host', '127.0.0.1', '--port', "$Port")
    $server = Start-Process -FilePath $VenvPy -ArgumentList $uvArgs -WorkingDirectory $Root -NoNewWindow -PassThru
    $up = $false
    for ($i = 0; $i -lt 90; $i++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$url/api/health" -TimeoutSec 2 | Out-Null
            $up = $true
            break
        } catch {
            if ($server.HasExited) { break }
            Start-Sleep -Seconds 1
        }
    }
    if (-not $up) {
        Write-Log 'The server did not come up; see the messages above.' 'Red'
        return
    }
    try {
        $info = Invoke-RestMethod -Uri "$url/api/system" -TimeoutSec 5
        Write-Log ("   backends: " + ($info.backends -join ', ') + "   GPU: " + $(if ($info.cuda_device) { $info.cuda_device } else { 'none' }))
    } catch { }
    Start-Process $url
    Write-Log "Flipster is running at $url  -  close this window (or press Ctrl+C) to stop it." 'Green'
    Wait-Process -Id $server.Id
}

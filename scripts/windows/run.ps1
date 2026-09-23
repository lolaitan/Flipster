<#
.SYNOPSIS
    Start Flipster (after setup.ps1 has run once) and open it in the browser.
#>
param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
Add-Content -Path $Log -Value "`nFlipster run $(Get-Date -Format s)" -Encoding UTF8
Start-Flipster -Port $Port

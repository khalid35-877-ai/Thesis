# One-click launcher for the Thesis Streamlit demo
# Usage: Right-click -> "Run with PowerShell"  OR  .\launch_app.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Suppress TF/oneDNN noise and slow initialisation logs before the Python process starts
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host ""
Write-Host "=== Thesis Demo Launcher ===" -ForegroundColor Cyan
Write-Host "Launching Streamlit app on http://localhost:8501"
Write-Host ""

# Kill anything already on port 8501
$occupied = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($occupied) {
    $occupied | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# Launch streamlit from repo root; web_app.py does os.chdir internally so paths resolve correctly
$proc = Start-Process -FilePath "streamlit" `
    -ArgumentList @("run", "tcontext\web_app.py", "--server.port", "8501", "--server.headless", "false") `
    -PassThru

# Wait up to 20 s for port to open, then open browser
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue) {
        $ready = $true
        break
    }
}

if ($ready) {
    Write-Host "App is ready -> opening http://localhost:8501" -ForegroundColor Green
    Start-Process "http://localhost:8501"
} else {
    Write-Host "App may still be starting. Check http://localhost:8501 in your browser." -ForegroundColor Yellow
}

# Keep the window open so Streamlit logs are visible
Wait-Process -Id $proc.Id

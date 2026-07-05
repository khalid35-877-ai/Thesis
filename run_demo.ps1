param(
    [int]$Seed = 777,
    [string]$QueryImage = "",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = "logs\thesis_run_${timestamp}.log"
$latestRunLog = "logs\thesis_run_latest.log"
$summaryPath = "tcontext\artifacts\sampled25_summary.json"
$comparisonPath = "tcontext\artifacts\sampled25_comparison_metrics.csv"
$modelPath = "tcontext\model\cats_vs_dogs_model_quick500.keras"
$vectorDbRoot = "tcontext\vector_db"

Write-Host "== Thesis comparative demo =="
Write-Host "Seed: $Seed"
Write-Host ""

function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments,
        [string]$StdOutPath,
        [string]$StepName
    )
    Write-Host "Running $StepName..."
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
    $stderrPath = "$StdOutPath.err.log"
    $allArgs = @($ScriptPath) + $Arguments
    $proc = Start-Process -FilePath $pythonExe -ArgumentList $allArgs -RedirectStandardOutput $StdOutPath -RedirectStandardError $stderrPath -Wait -PassThru
    $stderrText = if (Test-Path $stderrPath) { (Get-Content -Path $stderrPath -Raw).Trim() } else { "" }
    if ($stderrText) {
        Add-Content -Path $StdOutPath -Value "`n[stderr]`n$stderrText"
    }
    $null = Remove-Item -Path $stderrPath -Force -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Get-Content -Path $StdOutPath -Tail 100
        throw "$StepName failed with exit code $($proc.ExitCode). Check log: $StdOutPath"
    }
}

$hasRunnableAssets = (Test-Path $modelPath) -and (Test-Path $vectorDbRoot)
$hasComparativeMetrics = (Test-Path $summaryPath) -and (Test-Path $comparisonPath)

if (-not $hasRunnableAssets -or -not $hasComparativeMetrics) {
    $expArgs = @("--seed", "$Seed")
    if ($QueryImage -ne "") {
        $expArgs += @("--query-image", $QueryImage)
    }
    Invoke-PythonScript -ScriptPath "tcontext/quick500_experiment.py" -Arguments $expArgs -StdOutPath $runLog -StepName "experiment pipeline"
    Copy-Item -Path $runLog -Destination $latestRunLog -Force
    $edaLog = "logs\comparative_eda_${timestamp}.log"
    Invoke-PythonScript -ScriptPath "tcontext/comparative_eda.py" -Arguments @() -StdOutPath $edaLog -StepName "comparative EDA generation"
}
else {
    Write-Host "Comparative outputs are ready. Launching dashboard..."
}

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    $pidsToStop = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pidToStop in $pidsToStop) {
        Stop-Process -Id $pidToStop -Force -ErrorAction SilentlyContinue
    }
}

$streamlitArgs = @("run", "tcontext/web_app.py", "--server.port", "$Port", "--server.headless", "true")
$streamlitProc = Start-Process -FilePath "streamlit" -ArgumentList $streamlitArgs -PassThru

$maxWaitSeconds = 30
$isReady = $false
for ($i = 0; $i -lt $maxWaitSeconds; $i++) {
    Start-Sleep -Seconds 1
    $ready = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($ready) {
        $isReady = $true
        break
    }
}

if ($isReady) {
    $url = "http://localhost:$Port"
    Write-Host "Dashboard ready: $url"
    Start-Process $url | Out-Null
}
else {
    Write-Warning "Dashboard did not start within $maxWaitSeconds seconds."
}

Wait-Process -Id $streamlitProc.Id

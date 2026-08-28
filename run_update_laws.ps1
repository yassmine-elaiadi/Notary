# Wrapper invoked by the "NotaryRAG_UpdateLaws" scheduled task.
# Runs update_laws.py with the project venv and appends output to a log
# file, since a scheduled task has no console to watch.

Set-Location -Path $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$logFile = Join-Path $logDir "update_laws.log"
$start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logFile -Value "`n=== Run started: $start ==="

$env:PYTHONIOENCODING = "utf-8"
& ".\.venv\Scripts\python.exe" -u update_laws.py 2>&1 |
    Out-String -Stream |
    Out-File -FilePath $logFile -Append -Encoding utf8
$exitCode = $LASTEXITCODE

$end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logFile -Value "=== Run finished: $end (exit code $exitCode) ==="

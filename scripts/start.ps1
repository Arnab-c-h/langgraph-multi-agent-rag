param(
    [ValidateSet("full", "test")]
    [string]$Profile = "full",
    [int]$Port = 8000,
    [Alias("Host")]
    [string]$BindAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$env:APP_PROFILE = $Profile
$env:PORT = [string]$Port
& "$PSScriptRoot\..\.venv\Scripts\python.exe" -m uvicorn rag_app.main:app `
    --host $BindAddress --port $Port --workers 1 --limit-concurrency 2 `
    --timeout-keep-alive 5 --timeout-graceful-shutdown 15

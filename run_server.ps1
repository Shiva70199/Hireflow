# HireFlow API — Windows-friendly launcher (uvicorn is not always on PATH).
Set-Location $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = "1"
Write-Host "Starting: http://127.0.0.1:8000/docs" -ForegroundColor Green
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

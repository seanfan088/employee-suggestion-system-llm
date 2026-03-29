$ErrorActionPreference = "SilentlyContinue"
$proc = Start-Process -FilePath "cmd" -ArgumentList "/c", "cd /d C:\Users\364328\.config\opencode\employee-suggestion-system && npm run server" -NoNewWindow -PassThru
Start-Sleep 3
if ($proc -and !$proc.HasExited) {
    Write-Host "Server started, PID: $($proc.Id)"
} else {
    Write-Host "Failed to start server"
}

# Launch the orchestrator fully detached from the calling shell (survives the interactive session).
$code = "C:\Users\Mamun-Pro\Desktop\Research Books\study2026\code"
Start-Process -FilePath "C:\Python314\python.exe" -ArgumentList "run_all.py" -WorkingDirectory $code -WindowStyle Hidden
Write-Output "orchestrator launched"

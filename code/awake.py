"""Keep Windows from idle-sleeping while the detached orchestrator runs (per-process execution-state request, released on exit).
Exits when results/orchestrator.log ends with 'all jobs finished' or when no run_all.py process is alive. Windows only."""
import os, sys, time, ctypes, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); LOG = os.path.join(ROOT, "results", "orchestrator.log")
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
def orchestrator_alive():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\").CommandLine"], capture_output=True, text=True).stdout
    return "run_all.py" in out
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
while True:
    time.sleep(120)
    tail = open(LOG, encoding="utf-8", errors="ignore").read()[-200:] if os.path.exists(LOG) else ""
    if "all jobs finished" in tail or not orchestrator_alive():
        break
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

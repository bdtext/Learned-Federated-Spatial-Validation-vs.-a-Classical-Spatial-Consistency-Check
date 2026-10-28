"""Detached orchestrator: runs the remaining experiment jobs with bounded concurrency (thread and memory budgets, live free-memory
check), per-job thread counts and dependencies, logging to results/orchestrator.log. Every job is resumable, so the orchestrator
can simply be relaunched after an interruption; jobs already running outside this instance are recognized and never duplicated.
Launch with PowerShell Start-Process (see launch.ps1) so that it survives the interactive session.
"""
import os, sys, time, subprocess, threading
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE = os.path.join(ROOT, "code"); RES = os.path.join(ROOT, "results")
LOG = open(os.path.join(RES, "orchestrator.log"), "a", buffering=1)
MAX_THREADS = 27   # 16 physical cores / 24 logical; the model's small operations use ~1.5 cores per process, so many 3-thread processes
MAX_MEM_GB = 20.0  # sum of per-job memory estimates allowed at once (host: 32 GB)
MIN_FREE_GB = 1.5  # live available memory (physical or commit headroom, whichever is smaller) that must remain after starting a job
MAX_RETRIES = 2    # a job that exits with an error (e.g. out of memory) is re-queued; runners resume from their saved partial results

def log(m):
    LOG.write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")

def exists(p):
    return os.path.exists(os.path.join(RES, p))

def detect_done(corpus, seeds):
    return lambda: all(exists(f"{corpus}/seed{s}/metrics.json") for s in seeds)

RULES = ("mean", "median", "multikrum", "fltrust")
TUNE_GRID = [("32", "0.001"), ("32", "0.003")]  # the width-64 configuration is evaluated post hoc (see paper): never selected on NL/NA, 4x slower on DE
def cfg_cached(corpus, seed, ch, lr):
    return exists(f"{corpus}/seed{seed}/model_ch{ch}_lr{float(lr)}.pt") and exists(f"{corpus}/seed{seed}/history_ch{ch}_lr{float(lr)}.json")

# job: (name, argv, ready() predicate, done() predicate, threads, memory estimate in GB); roughly longest-first within a dependency level
JOBS = [(f"DE_cfg_{ch}_{lr}", ["run_detection.py", "DE", "0", "--only-cfg", f"{ch},{lr}"], lambda: True, (lambda ch=ch, lr=lr: cfg_cached("DE", 0, ch, lr)), 6, 3.0) for ch, lr in TUNE_GRID]
JOBS += [
    ("DE_tune", ["run_detection.py", "DE", "0", "--tune", "--grid", ";".join(f"{ch},{lr}" for ch, lr in TUNE_GRID)], lambda: all(cfg_cached("DE", 0, ch, lr) for ch, lr in TUNE_GRID), detect_done("DE", [0]), 6, 3.5),
    ("NA_seeds", ["run_detection.py", "NA", "0,1,2,3,4"], lambda: True, detect_done("NA", range(5)), 4, 1.5),
    ("NL_tune", ["run_detection.py", "NL", "0", "--tune"], lambda: True, detect_done("NL", [0]), 4, 1.5),
]
TH = 3
for s in range(3):
    JOBS.append((f"resid_EU_s{s}", ["run_residency.py", "EU", str(s), "--stride", "3"], lambda: exists("DE/tuning.json"), (lambda s=s: exists(f"residency_EU/seed{s}/residency.json")), TH, 2.5))
for s in range(3):
    JOBS.append((f"resid_DE_s{s}", ["run_residency.py", "DE", str(s)], lambda: exists("DE/tuning.json"), (lambda s=s: exists(f"residency_DE/seed{s}/residency.json")), TH, 2.0))
for s in range(1, 5):
    JOBS.append((f"DE_s{s}", ["run_detection.py", "DE", str(s)], lambda: exists("DE/tuning.json"), detect_done("DE", [s]), TH, 3.0))
for s in range(2):
    JOBS.append((f"DE_subtle_s{s}", ["run_detection.py", "DE", str(s), "--regime", "subtle", "--tag", "_subtle"], lambda: exists("DE/tuning.json"), detect_done("DE_subtle", [s]), TH, 3.0))
for s in range(1):
    for r in RULES:
        JOBS.append((f"robust_s{s}_{r}", ["run_robustness.py", str(s), "--rules", r, "--rounds", "5", "--stride", "4"], lambda: exists("DE/tuning.json"), (lambda s=s, r=r: exists(f"robustness_DE/seed{s}/done_{r}.flag")), TH, 2.0))
for s in range(1, 5):
    JOBS.append((f"NL_s{s}", ["run_detection.py", "NL", str(s)], lambda: exists("NL/tuning.json") and exists("NL/seed0/metrics.json"), detect_done("NL", [s]), TH, 1.5))

FAILED = {}
def run_job(name, argv, threads):
    log(f"start {name}: {' '.join(argv)} [{threads} threads]")
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), QC_THREADS=str(threads))
    with open(os.path.join(RES, f"log_{name}.txt"), "a", buffering=1) as f:
        f.write(time.strftime("\n==== %Y-%m-%d %H:%M:%S start\n"))
        p = subprocess.Popen([sys.executable] + argv, cwd=CODE, stdout=f, stderr=subprocess.STDOUT, env=env)
        rc = p.wait()
    log(f"end {name} rc={rc}")
    if rc != 0:
        FAILED[name] = FAILED.get(name, 0) + 1

def host_state():
    """(command lines of all python processes, available memory in GB) or None when the query fails (then nothing is started)."""
    cmd = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\").CommandLine; "
           "'AVAILMB=' + [string](Get-Counter '\\Memory\\Available MBytes').CounterSamples[0].CookedValue; "
           "'COMMITFREEMB=' + [string](((Get-Counter '\\Memory\\Commit Limit').CounterSamples[0].CookedValue - (Get-Counter '\\Memory\\Committed Bytes').CounterSamples[0].CookedValue)/1MB)")
    for _ in range(3):
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120).stdout
            if "run_all.py" in out and "AVAILMB=" in out and "COMMITFREEMB=" in out:
                avail = float(out.split("AVAILMB=")[1].strip().split()[0]) / 1024.0
                commit_free = float(out.split("COMMITFREEMB=")[1].strip().split()[0]) / 1024.0
                return out, min(avail, commit_free)  # binding constraint: physical availability or commit headroom
        except Exception:
            pass
        time.sleep(5)
    return None

def main():
    running = {}
    pending = [j for j in JOBS if not j[3]()]
    log(f"pending jobs: {[j[0] for j in pending]}")
    while pending or running:
        for name in list(running):
            if not running[name][0].is_alive():
                job = running.pop(name)[3]
                if FAILED.get(name, 0) and FAILED[name] <= MAX_RETRIES and not job[3]():
                    log(f"re-queue {name} (attempt {FAILED[name] + 1})"); pending.append(job)
        st = host_state()
        if st is None:
            log("host query failed; no job started this round"); time.sleep(20); continue
        cmds, free_gb = st
        external = [j for j in pending if " ".join(j[1]) in cmds]
        used_th = sum(r[1] for r in running.values()) + sum(j[4] for j in external)
        used_mem = sum(r[2] for r in running.values()) + sum(j[5] for j in external)
        for j in list(pending):
            name, argv, ready, done, threads, mem = j
            if done():
                pending.remove(j); continue
            if j in external:
                continue  # already running outside this orchestrator (e.g. started before a restart)
            if name not in running and ready() and used_th + threads <= MAX_THREADS and used_mem + mem <= MAX_MEM_GB and free_gb - mem >= MIN_FREE_GB:
                t = threading.Thread(target=run_job, args=(name, argv, threads), daemon=False); t.start()
                running[name] = (t, threads, mem, j); used_th += threads; used_mem += mem; free_gb -= mem; pending.remove(j)
        time.sleep(20)
    log("all jobs finished")

if __name__ == "__main__":
    main()

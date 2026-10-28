"""Timeline evidence for the pre-fixed decision criteria: file modification times of the manuscript drafts that state the research
questions and criteria, and the first/last timestamps of the experiment logs. Writes PROVENANCE_TIMELINE.md at the study root.
Usage: python provenance_timeline.py
"""
import os, re, glob, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def mtime(p):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))

def main():
    lines = ["# Provenance timeline (local time of the analysis host)", "",
             "The research questions and their decision criteria (Section I) were written into the manuscript draft before the benchmark runs. "
             "They were not deposited in a public registry; the evidence below is the file system and the experiment logs included in the release.", "",
             "## Draft files", ""]
    for f in ("paper/paper_front_draft.md", "paper/paper_methods.md", "paper/paper_methods_draft.md"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p): lines.append(f"- `{f}`: last modified {mtime(p)}")
    lines += ["", "## Experiment outputs", ""]
    firsts = []
    for pat in ("results/*/seed*/metrics.json", "results/residency_*/seed*/residency.json", "results/robustness_DE/seed*/robustness*.json", "results/govsim/govsim.json"):
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            firsts.append((os.path.getmtime(p), os.path.relpath(p, ROOT)))
    firsts.sort()
    if firsts:
        lines.append(f"- first result file: `{firsts[0][1]}` ({time.strftime('%Y-%m-%d %H:%M', time.localtime(firsts[0][0]))})")
        lines.append(f"- last result file: `{firsts[-1][1]}` ({time.strftime('%Y-%m-%d %H:%M', time.localtime(firsts[-1][0]))})")
    log = os.path.join(ROOT, "results", "orchestrator.log")
    if os.path.exists(log):
        stamps = [l[:19] for l in open(log, encoding="utf-8", errors="ignore") if re.match(r"\d{4}-\d{2}-\d{2} ", l)]
        if stamps: lines.append(f"- orchestrator log: {stamps[0]} to {stamps[-1]}")
    lines += ["", "Note: the draft files were edited after the runs for wording (e.g. protocol reductions and the IEEE Access layout); the research questions and criteria in Section I are unchanged from the 14 October 2026 draft, which is included as `paper/paper_front_draft.md`.", ""]
    open(os.path.join(ROOT, "PROVENANCE_TIMELINE.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()

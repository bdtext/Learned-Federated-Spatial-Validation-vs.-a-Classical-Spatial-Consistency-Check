"""Assemble the deliverable zips: code (with results and figures) and data (raw + processed + manifests + provenance).
Also regenerates the GHCNh download manifest and a top-level SHA-256 list.
Usage: python package.py
"""
import os, sys, json, hashlib, zipfile, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "deliverables"); os.makedirs(OUT, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def add_tree(z, base, rel_root, skip_ext=(), skip_dirs=()):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in skip_dirs and not d.startswith("__pycache__")]
        for f in fn:
            if f.endswith(skip_ext): continue
            p = os.path.join(dp, f)
            z.write(p, os.path.relpath(p, rel_root))

def main():
    # GHCNh manifest
    gh = os.path.join(ROOT, "data", "raw", "ghcnh", "files")
    cities = json.load(open(os.path.join(ROOT, "data", "raw", "isdlite", "city_ghcnh.json")))
    man = []
    for c in cities:
        for p in sorted(glob.glob(os.path.join(gh, f"GHCNh_{c['ghcnh']}_*.psv"))):
            man.append(dict(city=c["city"], ghcnh=c["ghcnh"], file=os.path.basename(p), bytes=os.path.getsize(p), sha256=sha(p)))
    json.dump(man, open(os.path.join(ROOT, "data", "raw", "ghcnh", "download_manifest.json"), "w"), indent=1)
    # code zip
    with zipfile.ZipFile(os.path.join(OUT, "code_and_results.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, os.path.join(ROOT, "code"), ROOT)
        add_tree(z, os.path.join(ROOT, "results"), ROOT)
        add_tree(z, os.path.join(ROOT, "figures"), ROOT)
        add_tree(z, os.path.join(ROOT, "paper"), ROOT)
        for f in ("README_DELIVERABLE.md", "PROVENANCE_TIMELINE.md"):
            if os.path.exists(os.path.join(ROOT, f)): z.write(os.path.join(ROOT, f), f)
    # data zip (raw + processed)
    with zipfile.ZipFile(os.path.join(OUT, "data.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, os.path.join(ROOT, "data"), ROOT, skip_ext=(".html",))
        for f in ("README_DELIVERABLE.md",):
            if os.path.exists(os.path.join(ROOT, f)): z.write(os.path.join(ROOT, f), f)
    # checksums
    with open(os.path.join(OUT, "SHA256SUMS.txt"), "w") as f:
        for name in ("code_and_results.zip", "data.zip"):
            p = os.path.join(OUT, name); f.write(f"{sha(p)}  {name}  {os.path.getsize(p)} bytes\n")
    print(open(os.path.join(OUT, "SHA256SUMS.txt")).read())

if __name__ == "__main__":
    main()

"""Download GHCNh (NOAA NCEI Global Historical Climatology Network hourly) by-year PSV files for the mapped city airports, 2023-2026.
Source: https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/<year>/psv/GHCNh_<ID>_<year>.psv
NOAA data: US Government work, public domain. Keeps only the columns needed (see prep_ghcnh.py).
"""
import os, json, time, hashlib
import requests

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "isdlite")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ghcnh")
BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year"
YEARS = [2023, 2024, 2025, 2026]

def main():
    os.makedirs(os.path.join(OUT, "files"), exist_ok=True)
    cities = json.load(open(os.path.join(RAW, "city_ghcnh.json")))
    manifest = []
    sess = requests.Session()
    for c in cities:
        for y in YEARS:
            fn = f"GHCNh_{c['ghcnh']}_{y}.psv"
            out = os.path.join(OUT, "files", fn)
            url = f"{BASE}/{y}/psv/{fn}"
            if not os.path.exists(out) or os.path.getsize(out) < 1000:
                for attempt in range(4):
                    try:
                        with sess.get(url, timeout=600, stream=True) as r:
                            if r.status_code == 200:
                                with open(out, "wb") as f:
                                    for chunk in r.iter_content(1 << 20):
                                        f.write(chunk)
                                break
                            elif r.status_code == 404:
                                print("missing", fn, flush=True); break
                    except Exception as e:
                        time.sleep(5 * (attempt + 1))
            if os.path.exists(out):
                manifest.append(dict(city=c["city"], ghcnh=c["ghcnh"], year=y, file=fn, url=url, bytes=os.path.getsize(out), sha256=hashlib.sha256(open(out, "rb").read()).hexdigest()))
        print("done", c["city"], c["ghcnh"], flush=True)
    json.dump(manifest, open(os.path.join(OUT, "download_manifest.json"), "w"), indent=1)
    print("files", len(manifest), "MB", sum(m["bytes"] for m in manifest) / 1e6)

if __name__ == "__main__":
    main()

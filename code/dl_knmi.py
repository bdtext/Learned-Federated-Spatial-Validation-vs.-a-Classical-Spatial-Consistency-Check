"""Download KNMI hourly station observations 2023-2026 per station per year via the script API.
Variables: T (0.1 degC), U (relative humidity %), P (sea-level pressure 0.1 hPa), FH (hourly mean wind 0.1 m/s), TD (dew point 0.1 degC).
Licence: CC BY 4.0 (KNMI). Source: https://www.daggegevens.knmi.nl/klimatologie/uurgegevens
"""
import os, re, json, time, hashlib
import requests

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "knmi")
URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
YEARS = [(2023, "2023010101", "2023123124"), (2024, "2024010101", "2024123124"), (2025, "2025010101", "2025123124"), (2026, "2026010101", "2026101224")]

def station_list():
    r = requests.post(URL, data="stns=ALL&vars=T&start=2024010101&end=2024010101", timeout=120)
    stns = []
    for line in r.text.splitlines():
        m = re.match(r"# (\d{3})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(.*)$", line)
        if m:
            stns.append(dict(id=m.group(1), lon=float(m.group(2)), lat=float(m.group(3)), elev=float(m.group(4)), name=m.group(5).strip()))
    return stns

def main():
    os.makedirs(os.path.join(RAW, "files"), exist_ok=True)
    stns = station_list()
    json.dump(stns, open(os.path.join(RAW, "stations.json"), "w"), indent=1)
    print("stations", len(stns), flush=True)
    manifest = []
    for s in stns:
        for y, a, b in YEARS:
            out = os.path.join(RAW, "files", f"knmi_{s['id']}_{y}.csv")
            if not os.path.exists(out) or os.path.getsize(out) < 500:
                for attempt in range(4):
                    try:
                        r = requests.post(URL, data=f"stns={s['id']}&vars=T:U:P:FH:TD&start={a}&end={b}", timeout=300)
                        if r.status_code == 200 and "# STN" in r.text:
                            open(out, "w", encoding="utf-8").write(r.text); break
                    except Exception:
                        time.sleep(5 * (attempt + 1))
            if os.path.exists(out):
                manifest.append(dict(station=s["id"], year=y, file=os.path.basename(out), bytes=os.path.getsize(out), sha256=hashlib.sha256(open(out, "rb").read()).hexdigest()))
        print("done", s["id"], s["name"], flush=True)
    json.dump(manifest, open(os.path.join(RAW, "download_manifest.json"), "w"), indent=1)
    print("files", len(manifest), "MB", sum(m["bytes"] for m in manifest) / 1e6)

if __name__ == "__main__":
    main()

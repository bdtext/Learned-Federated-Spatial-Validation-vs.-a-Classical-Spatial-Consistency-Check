"""Select DWD hourly stations in a regional box and download hist+recent zips for TU, TD, FF, P0.
Usage: python dl_dwd.py [select|download]
Region box (lat/lon) chosen to cover NW Germany (NRW, Lower Saxony, Hesse, Rhineland-Palatinate) and the Dutch border.
Licence: DWD Climate Data Center, CC BY 4.0 (https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf).
"""
import os, re, sys, json, hashlib, time
import requests

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
PROD = {"TU": "air_temperature", "TD": "dew_point", "FF": "wind", "P0": "pressure"}
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dwd")
BOX = dict(lat_min=50.0, lat_max=53.2, lon_min=6.0, lon_max=9.6)
MIN_END = 20261001  # station must still report in Oct 2026
MAX_START = 20221231  # and must have started before 2023

def parse_station_list(path):
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f.readlines()[2:]:
            if not line.strip():
                continue
            sid = line[0:5]; von = int(line[6:14]); bis = int(line[15:23])
            hoehe = int(line[24:38].strip() or 0)
            lat = float(line[39:50]); lon = float(line[51:60])
            rest = line[61:]
            name = rest[0:41].strip(); land = rest[41:82].strip()
            rows.append(dict(id=sid, start=von, end=bis, elev=hoehe, lat=lat, lon=lon, name=name, state=land))
    return rows

def select():
    tu = parse_station_list(os.path.join(RAW, "TU_stations.txt"))
    hist_idx = open(os.path.join(RAW, "TU_hist_index.html"), encoding="latin-1").read()
    rec_idx = open(os.path.join(RAW, "TU_recent_index.html"), encoding="latin-1").read()
    sel = []
    for s in tu:
        if not (BOX["lat_min"] <= s["lat"] <= BOX["lat_max"] and BOX["lon_min"] <= s["lon"] <= BOX["lon_max"]):
            continue
        if s["end"] < MIN_END or s["start"] > MAX_START:
            continue
        m = re.search(r"stundenwerte_TU_%s_(\d{8})_20251231_hist\.zip" % s["id"], hist_idx)
        if not m or f"stundenwerte_TU_{s['id']}_akt.zip" not in rec_idx:
            continue
        s["tu_hist"] = m.group(0)
        sel.append(s)
    # attach other products if available for the same station id
    for code in ("TD", "FF", "P0"):
        hi = open(os.path.join(RAW, f"{code}_hist_index.html"), encoding="latin-1").read()
        ri = open(os.path.join(RAW, f"{code}_recent_index.html"), encoding="latin-1").read()
        for s in sel:
            m = re.search(r"stundenwerte_%s_%s_(\d{8})_20251231_hist\.zip" % (code, s["id"]), hi)
            s[f"{code.lower()}_hist"] = m.group(0) if (m and f"stundenwerte_{code}_{s['id']}_akt.zip" in ri) else None
    json.dump(sel, open(os.path.join(RAW, "selected_stations.json"), "w"), indent=1)
    print("selected", len(sel), "stations; with TD", sum(1 for s in sel if s["td_hist"]), "FF", sum(1 for s in sel if s["ff_hist"]), "P0", sum(1 for s in sel if s["p0_hist"]))
    from collections import Counter
    print(Counter(s["state"] for s in sel))

def download():
    sel = json.load(open(os.path.join(RAW, "selected_stations.json")))
    manifest = []
    sess = requests.Session()
    for s in sel:
        for code in ("TU", "TD", "FF", "P0"):
            hist = s.get(f"{code.lower()}_hist")
            if hist is None:
                continue
            for kind, fn in (("historical", hist), ("recent", f"stundenwerte_{code}_{s['id']}_akt.zip")):
                url = f"{BASE}/{PROD[code]}/{kind}/{fn}"
                out = os.path.join(RAW, "files", fn)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if not os.path.exists(out) or os.path.getsize(out) == 0:
                    for attempt in range(4):
                        try:
                            r = sess.get(url, timeout=120)
                            if r.status_code == 200 and len(r.content) > 0:
                                open(out, "wb").write(r.content); break
                        except Exception as e:
                            time.sleep(3 * (attempt + 1))
                if os.path.exists(out):
                    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
                    manifest.append(dict(station=s["id"], product=code, kind=kind, file=fn, url=url, bytes=os.path.getsize(out), sha256=h))
        print("done station", s["id"], s["name"], flush=True)
    json.dump(manifest, open(os.path.join(RAW, "download_manifest.json"), "w"), indent=1)
    print("files:", len(manifest), "total MB:", sum(m["bytes"] for m in manifest) / 1e6)

if __name__ == "__main__":
    {"select": select, "download": download}[sys.argv[1]]()

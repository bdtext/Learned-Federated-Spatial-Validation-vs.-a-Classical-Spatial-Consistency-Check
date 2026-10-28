"""Parse the downloaded DWD hourly zips (TU, TD, FF, P0; historical + recent) into one hourly UTC table.
Output: data/processed/dwd_hourly.parquet (long format: time, station, TEMP, RH, DEWP, WSPM, PRES, PRES0 + QN flags)
        data/processed/dwd_stations.csv (id, name, state, lat, lon, elev)
Missing value in DWD files: -999.
"""
import os, io, json, zipfile, glob
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "dwd")
OUT = os.path.join(ROOT, "data", "processed")
T0, T1 = "2023-01-01 00:00", "2026-10-12 23:00"
PRODUCTS = {
    "TU": dict(cols={"TT_TU": "TEMP", "RF_TU": "RH"}, qn="QN_9"),
    "TD": dict(cols={"TD": "DEWP"}, qn="QN_8"),
    "FF": dict(cols={"F": "WSPM"}, qn="QN_3"),
    "P0": dict(cols={"P": "PRES", "P0": "PRES0"}, qn="QN_8"),
}

def read_product_zip(path):
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.startswith("produkt")][0]
    df = pd.read_csv(io.BytesIO(z.read(name)), sep=";", encoding="latin-1", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["MESS_DATUM"].astype(str), format="%Y%m%d%H", utc=True)
    return df

def main():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(os.path.join(RAW, "selected_stations.json")))
    idx = pd.date_range(T0, T1, freq="h", tz="UTC")
    frames = []
    meta = []
    for s in sel:
        sid = s["id"]
        st = pd.DataFrame(index=idx)
        have_any = False
        for code, spec in PRODUCTS.items():
            hist = s.get(f"{code.lower()}_hist")
            if hist is None:
                continue
            parts = []
            for fn in (hist, f"stundenwerte_{code}_{sid}_akt.zip"):
                p = os.path.join(RAW, "files", fn)
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    try:
                        parts.append(read_product_zip(p))
                    except Exception as e:
                        print("bad zip", fn, e)
            if not parts:
                continue
            df = pd.concat(parts).drop_duplicates("time", keep="last").set_index("time").sort_index()
            df = df[(df.index >= idx[0]) & (df.index <= idx[-1])]
            for src, dst in spec["cols"].items():
                v = pd.to_numeric(df[src], errors="coerce").replace(-999.0, np.nan)
                st[dst] = v.reindex(idx)
                have_any = True
            st[f"QN_{code}"] = pd.to_numeric(df[spec["qn"]], errors="coerce").reindex(idx)
        if have_any:
            st["station"] = sid
            frames.append(st.reset_index().rename(columns={"index": "time"}))
            meta.append(dict(id=sid, name=s["name"], state=s["state"], lat=s["lat"], lon=s["lon"], elev=s["elev"],
                             has_TEMP=int("TEMP" in st), has_DEWP=int("DEWP" in st), has_WSPM=int("WSPM" in st), has_PRES=int("PRES" in st)))
    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(os.path.join(OUT, "dwd_hourly.parquet"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "dwd_stations.csv"), index=False)
    cov = long.groupby("station")[[c for c in ("TEMP", "RH", "DEWP", "WSPM", "PRES") if c in long]].apply(lambda d: d.notna().mean())
    print("stations", len(meta), "rows", len(long))
    print(cov.describe().round(3))

if __name__ == "__main__":
    main()

"""Parse GHCNh by-year PSV files into one hourly UTC table (one observation per station-hour: the report closest to the top of the hour,
as in ISD-Lite), keeping NCEI quality codes for the real-flag endpoint.
Variables: temperature (degC) -> TEMP; relative_humidity (%) -> RH; dew_point_temperature -> DEWP; wind_speed (m/s) -> WSPM;
sea_level_pressure (hPa) -> PRES; station_level_pressure -> PRES0. Quality codes kept as Q_<var>.
Output: data/processed/ghcnh_hourly.parquet, data/processed/ghcnh_stations.csv
"""
import os, json, glob
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "ghcnh")
OUT = os.path.join(ROOT, "data", "processed")
T0, T1 = "2023-01-01 00:00", "2026-10-12 23:00"
VARS = {"temperature": "TEMP", "relative_humidity": "RH", "dew_point_temperature": "DEWP", "wind_speed": "WSPM",
        "sea_level_pressure": "PRES", "station_level_pressure": "PRES0"}

def read_psv(path):
    usecols = ["STATION", "DATE"] + [v for v in VARS] + [f"{v}_Quality_Code" for v in VARS]
    df = pd.read_csv(path, sep="|", usecols=lambda c: c in usecols, dtype=str, low_memory=False)
    df["time"] = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    for v in VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    # nearest-to-the-hour selection
    hour = df["time"].dt.round("h")
    dist = (df["time"] - hour).abs()
    df = df.assign(hour=hour, dist=dist).sort_values(["hour", "dist"]).drop_duplicates("hour", keep="first")
    return df

def main():
    os.makedirs(OUT, exist_ok=True)
    cities = json.load(open(os.path.join(ROOT, "data", "raw", "isdlite", "city_ghcnh.json")))
    idx = pd.date_range(T0, T1, freq="h", tz="UTC")
    frames, meta = [], []
    for c in cities:
        files = sorted(glob.glob(os.path.join(RAW, "files", f"GHCNh_{c['ghcnh']}_*.psv")))
        if not files:
            continue
        df = pd.concat([read_psv(f) for f in files]).drop_duplicates("hour", keep="first").set_index("hour").sort_index()
        df = df[(df.index >= idx[0]) & (df.index <= idx[-1])]
        st = pd.DataFrame(index=idx)
        for v, dst in VARS.items():
            st[dst] = df[v].reindex(idx)
            st[f"Q_{dst}"] = df[f"{v}_Quality_Code"].reindex(idx)
        st["station"] = c["ghcnh"]
        frames.append(st.reset_index().rename(columns={"index": "time"}))
        meta.append(dict(id=c["ghcnh"], name=c["name"], city=c["city"], state=c["ctry"], lat=c["lat"], lon=c["lon"], elev=c["elev"],
                         has_TEMP=int(st["TEMP"].notna().mean() > 0.5), has_DEWP=int(st["DEWP"].notna().mean() > 0.5),
                         has_WSPM=int(st["WSPM"].notna().mean() > 0.5), has_PRES=int(st["PRES"].notna().mean() > 0.5)))
        print("parsed", c["city"], len(df), flush=True)
    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(os.path.join(OUT, "ghcnh_hourly.parquet"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "ghcnh_stations.csv"), index=False)
    print("stations", len(meta), "rows", len(long))
    print(long.groupby("station")[["TEMP", "RH", "DEWP", "WSPM", "PRES"]].apply(lambda d: d.notna().mean()).describe().round(3))

if __name__ == "__main__":
    main()

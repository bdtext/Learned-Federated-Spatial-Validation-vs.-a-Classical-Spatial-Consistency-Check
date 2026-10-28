"""Parse KNMI hourly CSVs into one hourly UTC table.
KNMI hour HH = 1..24 denotes the hour interval ending at HH UT (hour 1 = 00:00-01:00 UT); we timestamp each row at the END of the
interval minus one hour? No: following KNMI's convention, HH=1 means 00-01 UT, so the observation time is (HH-1):00 UT when
the variable is an instantaneous reading at the end of the hour? KNMI states T is the temperature "at the time of observation"
within hour HH, i.e. at HH:00 UT. We therefore map HH -> hour HH (UTC), with HH=24 -> 00:00 of the next day.
Variables: T 0.1 degC -> TEMP; U % -> RH; P 0.1 hPa (sea level) -> PRES; FH 0.1 m/s -> WSPM; TD 0.1 degC -> DEWP.
Output: data/processed/knmi_hourly.parquet, data/processed/knmi_stations.csv
"""
import os, io, json, glob
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "knmi")
OUT = os.path.join(ROOT, "data", "processed")
T0, T1 = "2023-01-01 00:00", "2026-10-12 23:00"

def read_file(path):
    lines = [l for l in open(path, encoding="utf-8") if not l.startswith("#")]
    if not lines:
        return None
    df = pd.read_csv(io.StringIO("".join(lines)), header=None, skipinitialspace=True)
    df.columns = ["STN", "YYYYMMDD", "HH", "T", "U", "P", "FH", "TD"][: df.shape[1]]
    return df

def main():
    os.makedirs(OUT, exist_ok=True)
    stns = json.load(open(os.path.join(RAW, "stations.json")))
    idx = pd.date_range(T0, T1, freq="h", tz="UTC")
    frames, meta = [], []
    for s in stns:
        parts = [read_file(p) for p in sorted(glob.glob(os.path.join(RAW, "files", f"knmi_{s['id']}_*.csv")))]
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            continue
        df = pd.concat(parts)
        t = pd.to_datetime(df["YYYYMMDD"].astype(int).astype(str), format="%Y%m%d", utc=True) + pd.to_timedelta(df["HH"].astype(int), unit="h")
        df = df.assign(time=t).drop_duplicates("time").set_index("time").sort_index()
        st = pd.DataFrame(index=idx)
        st["TEMP"] = pd.to_numeric(df["T"], errors="coerce").reindex(idx) / 10.0
        st["RH"] = pd.to_numeric(df["U"], errors="coerce").reindex(idx)
        st["PRES"] = pd.to_numeric(df["P"], errors="coerce").reindex(idx) / 10.0
        st["WSPM"] = pd.to_numeric(df["FH"], errors="coerce").reindex(idx) / 10.0
        st["DEWP"] = pd.to_numeric(df["TD"], errors="coerce").reindex(idx) / 10.0
        if st["TEMP"].notna().mean() < 0.5:
            continue  # stations without temperature (e.g., precipitation-only or offshore) are dropped
        st["station"] = f"NL{s['id']}"
        frames.append(st.reset_index().rename(columns={"index": "time"}))
        meta.append(dict(id=f"NL{s['id']}", name=s["name"], state="Netherlands", lat=s["lat"], lon=s["lon"], elev=s["elev"],
                         has_TEMP=1, has_DEWP=int(st["DEWP"].notna().mean() > 0.5), has_WSPM=int(st["WSPM"].notna().mean() > 0.5), has_PRES=int(st["PRES"].notna().mean() > 0.5)))
    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(os.path.join(OUT, "knmi_hourly.parquet"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "knmi_stations.csv"), index=False)
    print("stations", len(meta), "rows", len(long))
    print(long.groupby("station")[["TEMP", "RH", "DEWP", "WSPM", "PRES"]].apply(lambda d: d.notna().mean()).describe().round(3))

if __name__ == "__main__":
    main()

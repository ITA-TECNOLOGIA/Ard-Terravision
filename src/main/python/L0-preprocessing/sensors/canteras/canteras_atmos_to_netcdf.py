import os
import re
import unicodedata
import glob
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

def safe_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")  # remove accents
    s = s.replace("/", "_per_")  # fix slashes
    s = re.sub(r"\([^)]*\)", "", s)  # remove units in parentheses
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s



# 1) files (adjust path/pattern)
data_path = os.getenv("CANTERAS_WEATHER_DATA_PATH")
files = [os.path.join(data_path, f"Weathercloud CANTERAS 2 2025-{i:02d}.csv") for i in range(5, 12)]

# 2) read + concat
dfs = []
for f in files:
    # Read each df
    df = pd.read_csv(f, sep=";", encoding="utf-16le", engine="python", on_bad_lines="skip")
    # Clean df
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["time"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce")
    df = df.drop(columns=df.columns[0]).set_index("time")
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce"))
    df = df.rename(columns={c: safe_name(c) for c in df.columns})
    dfs.append(df)

data = pd.concat(dfs).sort_index()
data = data[~data.index.duplicated(keep="last")]  # remove overlaps if any

# 3) to NetCDF
ds = xr.Dataset.from_dataframe(data)
ds.to_netcdf(os.path.join(data_path, "canteras_atmos.nc"))
print(ds)

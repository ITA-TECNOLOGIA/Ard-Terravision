import os
import re
import unicodedata
import glob
import pandas as pd
import xarray as xr

SENSOR_COORDINATES = {
    #### NEW ####
    #Tharsis
    "181139E5":(667141.902, 4163281.618, 264.619),
    "187046AB":(666952.000, 4163424.000, 273.467),
    "18704AC1":(666377.686, 4162553.681, 282.624),
    "16402035":(667279.947, 4162623.522, 252.352), #GATEWAY FILÓN NORTE
    #La Zarza
    "187045A2":(689422.391, 4175177.646, 246.800),
    "18704AAF":(689646.792, 4175692.696, 295.236),
    "18704B7E":(689398.433, 4175924.098, 279.980),
    "18704C66":(689055.305, 4174823.136, 209.062),
    "164020A0":(688950.655, 4174953.163, 211.765), #GATEWAY LOS CEPOS

    "16222309": (667299.385,4162669.941,252.668),
    "16221F19": (667280.590,4162671.903,252.254),
    "16221F12": (667294.134,4162647.590,252.374),
    "16221D04": (667272.154,4162650.206,251.201),
    "16221F59": (688950.527,4174961.024,211.970),
    "16221EFA": (685950.642,4174953.175,211.765),
    "16221E6C": (688950.821,4174948.136,211.284),
    "16221F10": (688950.815,4174968.052,212.537),

}
def safe_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")  # remove accents
    s = s.replace("/", "_per_")  # fix slashes
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s

# files (adjust path/pattern)
data_path = "/datassd/proyectos/terravision/terravision_sensor_data/tharsis/tilt"

all_files = glob.glob(os.path.join(data_path, "*.csv"))
data_files = glob.glob(os.path.join(data_path, "*_data.csv"))
gateway_files = [f for f in all_files if not f.endswith("_data.csv")]

# data
dfs = []
for f in data_files:
    # Read each df
    df = pd.read_csv(f, sep=",", encoding="utf", engine="python", on_bad_lines="skip")
    # Clean df
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["time"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df["time"] = pd.to_datetime(df['time'], unit="s", dayfirst=True)
    df.rename(columns={"Serial Number": "sensor"}, inplace=True)
    df = df.drop(columns=df.columns[:1]).set_index(["time", "sensor"])
    df = df.rename(columns={c: safe_name(c) for c in df.columns})
    dfs.append(df)

data = pd.concat(dfs).sort_index()
data = data[~data.index.duplicated(keep="last")]  # remove overlaps if any

# Metadata
dfs = []
for f in gateway_files:
    # Read each df
    df = pd.read_csv(f, sep=",", encoding="utf", engine="python", on_bad_lines="skip")
    # Clean df
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["time"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df["time"] = pd.to_datetime(df['time'], unit="s", dayfirst=True)
    df.rename(columns={"Serial Number": "sensor"}, inplace=True)
    df = df.drop(columns=df.columns[:1]).set_index(["time", "sensor"])
    df = df.rename(columns={c: safe_name(c) for c in df.columns})
    dfs.append(df)

gateway = pd.concat(dfs).sort_index()
gateway = gateway[~gateway.index.duplicated(keep="last")]  # remove overlaps if any

# NetCDF
ds = xr.Dataset.from_dataframe(data)
ds = ds.assign_coords(
    x=("sensor", [SENSOR_COORDINATES[s][0] for s in ds.sensor.values]),
    y=("sensor", [SENSOR_COORDINATES[s][1] for s in ds.sensor.values]),
)
ds.attrs["crs"] = "EPSG:32629"
ds.to_netcdf(os.path.join(data_path, "tharsis_tilt_geolocated.nc"))

print("Data:", ds)

ds = xr.Dataset.from_dataframe(gateway)
ds = ds.assign_coords(
    x=("sensor", [SENSOR_COORDINATES[s][0] for s in ds.sensor.values]),
    y=("sensor", [SENSOR_COORDINATES[s][1] for s in ds.sensor.values]),
)
ds.attrs["crs"] = "EPSG:32629"
ds.to_netcdf(os.path.join(data_path, "tharsis_tilt_gateway.nc"))
print("Metadata:", ds)


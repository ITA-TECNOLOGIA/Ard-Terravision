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

VARIABLE_RENAME_MAP = {
    "temperatura_interior": "indoor_temperature",
    "temperatura": "temperature",
    "sensacion_termica": "apparent_temperature",
    "punto_de_rocio_interior": "indoor_dew_point_temperature",
    "punto_de_rocio": "dew_point_temperature",
    "indice_de_calor_interior": "indoor_heat_index",
    "indice_de_calor": "heat_index",
    "humedad_interior": "indoor_humidity",
    "humedad": "humidity",
    "rafaga_maxima_de_viento": "wind_gust_speed",
    "velocidad_media_del_viento": "wind_mean_speed",
    "direccion_media_del_viento": "wind_direction",
    "presion_atmosferica": "air_pressure",
    "lluvia": "precipitation",
    "evapotranspiracion": "evapotranspiration",
    "intensidad_de_lluvia": "precipitation_rate",
    "radiacion_solar": "solar_radiation",
    "indice_uv": "uv_index",
}

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
ds = ds.rename_vars(VARIABLE_RENAME_MAP)
ds.to_netcdf(os.path.join(data_path, "canteras_atmos_2.nc"))
print(ds)

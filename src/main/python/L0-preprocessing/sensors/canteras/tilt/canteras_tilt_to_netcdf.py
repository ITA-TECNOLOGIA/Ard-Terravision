import os
import json
import pandas as pd
import numpy as np
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv("CANTERAS_TILT_DATA_PATH")
file_path = os.path.join(data_path, "worldsensing_flat_historic.jsonl")
output_nc = os.path.join(data_path, "canteras_tilt_historic.nc")

rows = []
with open(file_path, "r") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

# ---- Parsear time (tiempo real)
df["time"] = pd.to_datetime(df["date-and-time"], errors="coerce", utc=True).dt.tz_convert(None)

# Quitar filas sin time o sin sensor_id (no sirven para serie temporal)
df = df.dropna(subset=["time", "sensor_id"])

# Renombrar sensor_id a sensor (como tu pipeline)
df = df.rename(columns={"sensor_id": "sensor"})

# Convertir a numérico todas las columnas que se pueda (sin romper strings)
for col in df.columns:
    if col not in ["device_id", "device_name", "device_model", "sensor", "msg_type", "topic", "date-and-time"]:
        df[col] = pd.to_numeric(df[col], errors="ignore")

# Ordenar
df = df.sort_values(["sensor", "time"])

# ---- Index correcto
df = df.set_index(["time", "sensor"])

# ---- Resolver duplicados SOLO dentro del mismo sensor y timestamp
# (esto es inevitable para tener una grilla time×sensor)
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

df_num = df[numeric_cols].groupby(level=["time", "sensor"]).mean()

if len(non_numeric_cols) > 0:
    df_meta = df[non_numeric_cols].groupby(level=["time", "sensor"]).first()
    df_final = pd.concat([df_num, df_meta], axis=1)
else:
    df_final = df_num

# ---- A xarray
ds = df_final.to_xarray()

# Garantizar que time es datetime64
ds["time"] = pd.to_datetime(ds["time"].values)

# Guardar
ds.to_netcdf(output_nc)

print("NetCDF guardado en:", output_nc)
print(ds)
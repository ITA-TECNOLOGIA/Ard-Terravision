import os
from pathlib import Path

import pandas as pd
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv("THARSIS_WATER_DATA_PATH")
data_csv = os.path.join(data_path, "water_sensors.csv")
coords_csv = os.path.join(data_path, "water_coords.csv")
output_nc = os.path.join(data_path, "tharsis_water.nc")

df = pd.read_csv(data_csv)
coords = pd.read_csv(coords_csv)
df["time"] = pd.to_datetime(df["time"])

df = df.merge(coords, on="sensor", how="left")

sensor_info = (
    df[
        [
            "sensor",
            "place",
            "X_ETRS89H29",
            "Y_ETRS89H29",
        ]
    ]
    .drop_duplicates(subset="sensor")
    .set_index("sensor")
)

ph = df.pivot(index="time", columns="sensor", values="pH")

conductivity = df.pivot(
    index="time",
    columns="sensor",
    values="conductivity",
)

temperature = df.pivot(
    index="time",
    columns="sensor",
    values="temperature",
)

ds = xr.Dataset(
    data_vars={
        "pH": (
            ("time", "sensor"),
            ph.values,
        ),
        "conductivity": (
            ("time", "sensor"),
            conductivity.values,
        ),
        "temperature": (
            ("time", "sensor"),
            temperature.values,
        ),
    },
    coords={
        "time": ph.index.values,
        "sensor": ph.columns.values,
        "x": (
            "sensor",
            sensor_info.loc[
                ph.columns,
                "X_ETRS89H29",
            ].values,
        ),
        "y": (
            "sensor",
            sensor_info.loc[
                ph.columns,
                "Y_ETRS89H29",
            ].values,
        ),
    },
)


ds["place"] = (
    "sensor",
    sensor_info.loc[
        ph.columns,
        "place",
    ].values,
)

ds.attrs["crs"] = "EPSG:25829"
ds.attrs["description"] = "Water quality monitoring datacube"

ds["pH"].attrs["units"] = "-"
ds["conductivity"].attrs["units"] = "uS/cm"
ds["temperature"].attrs["units"] = "degC"

ds["x"].attrs["units"] = "m"
ds["y"].attrs["units"] = "m"

ds.to_netcdf(output_nc)

print(ds)
print(f"\nNetCDF guardado en: {Path(output_nc).resolve()}")
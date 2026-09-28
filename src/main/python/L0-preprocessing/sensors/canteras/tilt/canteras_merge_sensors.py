import os
import xarray as xr
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("CANTERAS_TILT_DATA_PATH")
INPUT_NC = os.path.join(DATA_PATH, "canteras_tilt.nc")
OUTPUT_NC = os.path.join(DATA_PATH, "canteras_tilt_joined.nc")

# Sensor locations (longitude, latitude, elevation)
SENSOR_COORDINATES = {
    "LS-193131": (37.104236, -3.693808, 897.28),  # Base
    "LS-193132": (37.104393, -3.693153, 899.10),  # Rover I
    "LS-193195": (37.104286, -3.693017, 913.15),  # Rover II
}

TAKE_FIRST_SENSOR_VARS = {
    "sensor",
    "uuid1",
    "uuid2",
    "topic",
    "node_id",
    "reading_type",
    "msg_type",
    "msg_version",
    "msg_trigger",
    "schemaVersion",
    "ingest_ts_utc",
}

def merge_two_datasets(ds_a: xr.Dataset, ds_b: xr.Dataset) -> xr.Dataset:
    out = ds_a.copy()
    for var in ds_b.data_vars:
        if var not in out:
            out[var] = ds_b[var]
        else:
            out[var] = out[var].combine_first(ds_b[var])
    return out


def merge_sensors_for_device(ds, sensor_list):
    merged = None

    for s in sensor_list:
        ds_s = ds.sel(sensor=s).drop_vars("sensor", errors="ignore")

        for v in TAKE_FIRST_SENSOR_VARS:
            if v in ds_s:
                ds_s = ds_s.drop_vars(v)

        if merged is None:
            merged = ds_s
        else:
            merged = merge_two_datasets(merged, ds_s)

    return merged


def main():
    ds = xr.open_dataset(INPUT_NC)

    if "sensor" not in ds.dims:
        raise ValueError("The dataset has no dimension 'sensor'.")

    if "device_name" not in ds:
        raise ValueError("Variable device_name does not exist.")

    sensor_to_device = {}
    for s in ds["sensor"].values:
        dev_name = ds["device_name"].sel(sensor=s).values

        dev_name = pd.Series(dev_name).dropna()
        dev_name = dev_name[dev_name != ""]

        if len(dev_name) == 0:
            sensor_to_device[s] = "__UNKNOWN__"
        else:
            sensor_to_device[s] = str(dev_name.iloc[0])

    devices = {}
    for sensor_id, dev_name in sensor_to_device.items():
        devices.setdefault(dev_name, []).append(sensor_id)

    print("Sensors grouped by device:")
    for dev, sens in devices.items():
        print(f"  {dev}: {len(sens)} sensors -> {sens}")

    device_datasets = []
    device_labels = []

    for dev_name, sens_list in devices.items():
        merged_dev = merge_sensors_for_device(ds, sens_list)

        for dropv in ["device_name", "device_id", "device_model"]:
            if dropv in merged_dev:
                merged_dev = merged_dev.drop_vars(dropv)

        device_datasets.append(merged_dev)
        device_labels.append(dev_name)


    ds_out = xr.concat(device_datasets, dim=pd.Index(device_labels, name="device"))
    ds_out = ds_out.rename({"device": "sensor"})

    coords_lower = {k.lower(): v for k, v in SENSOR_COORDINATES.items()}
    sensor_coords = [
        coords_lower.get(str(s).lower(), (np.nan, np.nan, np.nan))
        for s in ds_out.sensor.values
    ]
    ds_out = ds_out.assign_coords(
        x=("sensor", [c[0] for c in sensor_coords]),
        y=("sensor", [c[1] for c in sensor_coords]),
        z=("sensor", [c[2] for c in sensor_coords]),
    )
    ds_out.attrs["crs"] = "EPSG:4326"
    ds_out.to_netcdf(OUTPUT_NC)
    print(f"\nOK -> saved on {OUTPUT_NC}")
    print("Final dims:", ds_out.dims)


if __name__ == "__main__":
    main()
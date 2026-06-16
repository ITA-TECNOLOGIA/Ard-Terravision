import xarray as xr
import numpy as np
import pandas as pd

INPUT_NC = "/datassd/proyectos/terravision/terravision_sensor_data/canteras/canteras_tilt.nc"
OUTPUT_NC = "/datassd/proyectos/terravision/terravision_sensor_data/canteras/canteras_tilt_joined.nc"

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
    ds_out.to_netcdf(OUTPUT_NC)
    print(f"\nOK -> saved on {OUTPUT_NC}")
    print("Final dims:", ds_out.dims)


if __name__ == "__main__":
    main()
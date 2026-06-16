import xarray as xr
import numpy as np
import pandas as pd


def _is_datetime64(arr):
    return np.issubdtype(arr.dtype, np.datetime64)


def _guess_epoch_unit(time_values):
    v = np.asarray(time_values)

    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None

    median_abs = np.median(np.abs(v))

    if median_abs < 1e11:
        return "s"
    elif median_abs < 1e14:
        return "ms"
    elif median_abs < 1e17:
        return "us"
    else:
        return "ns"


def normalize_structure(ds, sensor_dim_candidates=("sensor", "sensor_id", "device_id")):
    print("=== STRUCTURE NORMALIZATION ===")

    sensor_dim = None
    for cand in sensor_dim_candidates:
        if cand in ds.dims:
            sensor_dim = cand
            break

    if sensor_dim is None:
        print("Check sensor dimension: NOT FOUND (monosensor dataset assumed)")
    else:
        print(f"Check sensor dimension: FOUND -> '{sensor_dim}'")

    if "sensor_id" in ds.dims and "sensor" not in ds.dims:
        print("Action: renaming dimension 'sensor_id' -> 'sensor'")
        ds = ds.rename({"sensor_id": "sensor"})
        sensor_dim = "sensor"

    if "time" not in ds.coords and "time" in ds.variables:
        print("Action: 'time' exists as variable but not coordinate -> set as coordinate")
        ds = ds.set_coords("time")

    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate. Cannot proceed.")

    print(f"Check time dtype: {ds['time'].dtype}")

    if np.issubdtype(ds["time"].dtype, np.integer):
        print("Detected integer time axis. Guessing epoch units...")
        unit = _guess_epoch_unit(ds["time"].values)
        print(f"Guessed epoch unit: {unit}")

        if unit is None:
            raise ValueError("Cannot guess epoch unit (time array empty or invalid).")

        ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values, unit=unit, utc=True).tz_convert(None))
        print("Action: converted integer epoch time -> datetime64[ns]")

    if not _is_datetime64(ds["time"].values):
        print("Detected non-datetime time axis. Trying xr.decode_cf()...")
        try:
            ds = xr.decode_cf(ds)
            print("Action: xr.decode_cf() applied")
        except Exception as e:
            raise ValueError(f"xr.decode_cf() failed: {e}")

    print(f"Check time dtype after decode: {ds['time'].dtype}")

    nat_count = int(ds["time"].isnull().sum())
    print(f"Check NaT in time: {nat_count}")

    if nat_count > 0:
        ds = ds.sel(time=~ds["time"].isnull())
        print(f"Action: removed {nat_count} NaT timestamps")

    is_sorted = bool(np.all(np.diff(ds["time"].values.astype("datetime64[ns]").astype("int64")) >= 0))
    print(f"Check time sorted: {is_sorted}")

    if not is_sorted:
        ds = ds.sortby("time")
        print("Action: sorted dataset by time")

    if sensor_dim is None:
        # Monosensor: deduplicate time directly
        t = ds["time"].values
        total_init = len(t)
        _, idx = np.unique(t, return_index=True)

        duplicates = total_init - len(idx)
        print(f"Check duplicates in time: {duplicates}")

        if duplicates > 0:
            ds = ds.isel(time=np.sort(idx))
            print(f"Action: removed {duplicates} duplicate timestamps (kept first occurrence)")
    else:
        print("Check duplicates in (time,sensor): cannot be fixed safely at xarray level without stacking.")
        print("Action: no (time,sensor) deduplication applied (assumed already gridded)")

    print("=== RESULT ===")
    print(ds)

    return ds
import xarray as xr
import numpy as np
import yaml


def circular_mean_deg(da, dim):
    radians = np.deg2rad(da)
    sin_mean = np.sin(radians).mean(dim=dim, skipna=True)
    cos_mean = np.cos(radians).mean(dim=dim, skipna=True)
    angle = np.arctan2(sin_mean, cos_mean)
    deg = (np.rad2deg(angle) + 360.0) % 360.0
    return deg


def is_numeric(da):
    return np.issubdtype(da.dtype, np.number)


def resample_numeric(da, freq, method):
    r = da.resample(time=freq)

    if method == "mean":
        return r.mean(skipna=True)
    elif method == "sum":
        return r.sum(skipna=True)
    elif method == "max":
        return r.max(skipna=True)
    elif method == "min":
        return r.min(skipna=True)
    elif method == "median":
        return r.median(skipna=True)
    elif method == "first":
        return r.first()
    elif method == "last":
        return r.last()
    elif method == "circular_mean":
        return circular_mean_deg(r, dim="time")
    else:
        raise ValueError(f"Unknown numeric resample method '{method}'")


def resample_non_numeric(da, freq, method):
    r = da.resample(time=freq)

    if method == "first":
        return r.first()
    elif method == "last":
        return r.last()
    elif method == "none":
        return da
    else:
        raise ValueError(f"Unknown non-numeric resample method '{method}'")


def get_variable_semantic(cfg, var):
    return cfg.get("variables", {}).get(var, {}).get("semantic", None)


def get_resample_method(cfg, var, da):
    res_cfg = cfg.get("resample", {})
    override = res_cfg.get("override", {}) or {}

    # explicit override wins
    if var in override:
        return override[var]

    semantic = get_variable_semantic(cfg, var)

    # semantic default
    default_by_semantic = res_cfg.get("default_by_semantic", {}) or {}
    if semantic is not None and semantic in default_by_semantic:
        return default_by_semantic[semantic]

    # fallback if no semantic
    if is_numeric(da):
        return "mean"
    return "first"


def resample_dataset(ds, cfg):
    res_cfg = cfg.get("resample", {})
    if not res_cfg:
        print("=== RESAMPLE: SKIPPED (no resample config) ===")
        return ds
    freq = res_cfg.get("freq", "10min")

    print("=== RESAMPLE ===")
    print("Frequency:", freq)

    out_vars = {}

    for var in ds.data_vars:
        da = ds[var]

        if "time" not in da.dims:
            out_vars[var] = da
            print(f"Variable '{var}': no time dim -> kept unchanged")
            continue

        semantic = get_variable_semantic(cfg, var)
        method = get_resample_method(cfg, var, da)

        if is_numeric(da):
            out_vars[var] = resample_numeric(da, freq, method)
            print(f"Variable '{var}': numeric semantic={semantic} -> {method}")
        else:
            out_vars[var] = resample_non_numeric(da, freq, method)
            print(f"Variable '{var}': non-numeric semantic={semantic} -> {method}")

    ds_resampled = xr.Dataset(out_vars)

    for coord in ds.coords:
        if coord != "time":
            ds_resampled = ds_resampled.assign_coords({coord: ds.coords[coord]})
    ds_resampled.attrs = ds.attrs
    
    print("=== DONE ===")
    print(ds_resampled)

    return ds_resampled
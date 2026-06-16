import xarray as xr
import numpy as np
import yaml
import pandas as pd

def max_gap_to_limit(ds, max_gap):
    if max_gap is None:
        return None

    dt = ds["time"].diff("time").median().values
    dt = np.timedelta64(dt, "ns")

    max_gap_td = np.timedelta64(pd.to_timedelta(max_gap).to_timedelta64(), "ns")

    limit = int(max_gap_td / dt)
    if limit < 1:
        limit = 1
    return limit

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_semantic(cfg, var):
    return cfg.get("variables", {}).get(var, {}).get("semantic", None)


def is_numeric(da):
    return np.issubdtype(da.dtype, np.number)


def interpolate_linear(da, max_gap):
    return da.interpolate_na(dim="time", method="linear", max_gap=max_gap)


def run_interpolation(ds, cfg):
    print("=== INTERPOLATION ===")

    interp_cfg = cfg.get("interpolation", {})
    enabled = interp_cfg.get("enabled", True)
    write_flags = bool(interp_cfg.get("write_flags", False))

    if not enabled:
        print("Interpolation disabled.")
        return ds

    max_gap = interp_cfg.get("max_gap", None)
    limit = max_gap_to_limit(ds, max_gap)

    by_semantic = interp_cfg.get("by_semantic", {})
    override = interp_cfg.get("override", {})

    ds_out = ds.copy()

    for var in ds.data_vars:
        da = ds[var]

        if var.endswith("__was_outlier") or var.endswith("__is_interpolated"):
            continue

        if "time" not in da.dims:
            continue

        semantic = get_semantic(cfg, var)

        method = override.get(var, None)
        if method is None:
            method = by_semantic.get(semantic, None)

        if method is None:
            continue
        
        missing_before = da.isnull()
        nan_before = int(da.isnull().sum())

        if method == "linear":
            if not is_numeric(da):
                print(f"{var}: skip (linear interpolation requires numeric)")
                continue
            ds_out[var] = interpolate_linear(da, max_gap=max_gap)

        elif method in ["ffill", "forward_fill"]:
            if np.issubdtype(da.dtype, np.number):
                ds_out[var] = da.ffill(dim="time", limit=limit)
            else:
                # object / strings: fallback to pandas
                df = da.to_pandas()
                df_ff = df.ffill(limit=limit)
                ds_out[var] = xr.DataArray(df_ff, dims=da.dims, coords=da.coords)

        else:
            raise ValueError(f"Unknown interpolation method '{method}' for var '{var}'")
        
        if write_flags:
            filled_mask = missing_before & ds_out[var].notnull()
            ds_out[f"{var}__is_interpolated"] = filled_mask.fillna(False)

        nan_after = int(ds_out[var].isnull().sum())
        print(f"{var}: filled {nan_before - nan_after} (semantic={semantic}, method={method})")

    print("=== DONE ===")
    return ds_out
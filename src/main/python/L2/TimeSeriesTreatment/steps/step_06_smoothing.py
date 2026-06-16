import xarray as xr
import numpy as np
import yaml
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_numeric(da):
    return np.issubdtype(da.dtype, np.number)


def get_semantic(cfg, var):
    return cfg.get("variables", {}).get(var, {}).get("semantic", None)


def is_enabled_for_var(cfg, var):
    smooth_cfg = cfg.get("smoothing", {})
    override = smooth_cfg.get("override", {}).get(var, None)

    if override is not None and "enabled" in override:
        return bool(override["enabled"])

    semantic = get_semantic(cfg, var)
    if semantic is None:
        return False

    sem_cfg = smooth_cfg.get("by_semantic", {}).get(semantic, {})
    return bool(sem_cfg.get("enabled", True))


def get_method_for_var(cfg, var):
    smooth_cfg = cfg.get("smoothing", {})
    override = smooth_cfg.get("override", {}).get(var, None)

    if override is not None and "method" in override:
        return override["method"]

    semantic = get_semantic(cfg, var)
    sem_cfg = smooth_cfg.get("by_semantic", {}).get(semantic, {})

    return sem_cfg.get("method", "mean")


def smooth_da(da, window, min_valid, method):
    r = da.rolling(time=window, center=True)

    valid_count = da.notnull().rolling(time=window, center=True).sum().astype(int)

    if method == "mean":
        smoothed = r.mean(skipna=True)
    elif method == "median":
        smoothed = r.median(skipna=True)
    else:
        raise ValueError(f"Unknown smoothing method '{method}'")

    smoothed = smoothed.where(valid_count >= min_valid)

    out = xr.where(valid_count >= min_valid, smoothed, da)

    return out


def run_smoothing(ds, cfg):
    print("=== SMOOTHING ===")

    smooth_cfg = cfg.get("smoothing", {})
    enabled = smooth_cfg.get("enabled", True)

    if not enabled:
        print("Smoothing disabled.")
        return ds

    window = int(smooth_cfg.get("window", 6))
    min_valid = smooth_cfg.get("min_valid", None)
    if min_valid is None:
        min_valid = max(2, window // 2)

    print(f"window={window} min_valid={min_valid}")

    ds_out = ds.copy()

    for var in ds.data_vars:
        da = ds[var]
        if var.endswith("__was_outlier") or var.endswith("__is_interpolated"):
            continue

        if "time" not in da.dims:
            continue

        if not is_numeric(da):
            continue

        if not is_enabled_for_var(cfg, var):
            continue

        semantic = get_semantic(cfg, var)
        method = get_method_for_var(cfg, var)

        nan_before = int(da.isnull().sum())

        ds_out[var] = smooth_da(da, window, min_valid, method)

        nan_after = int(ds_out[var].isnull().sum())
        filled = nan_before - nan_after
        print(f"{var}: smoothed (semantic={semantic}, method={method}, filled={filled})")

    print("=== DONE ===")
    return ds_out
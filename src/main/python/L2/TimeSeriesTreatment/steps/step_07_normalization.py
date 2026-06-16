import xarray as xr
import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_numeric(da):
    return np.issubdtype(da.dtype, np.number)


def get_semantic(cfg, var):
    return cfg.get("variables", {}).get(var, {}).get("semantic", None)


def get_method_for_var(cfg, var):
    norm_cfg = cfg.get("normalization", {})
    override = norm_cfg.get("override", {}) or {}

    if var in override:
        return override[var]

    semantic = get_semantic(cfg, var)
    by_sem = norm_cfg.get("by_semantic", {}) or {}

    if semantic is None:
        return "none"

    return by_sem.get(semantic, "none")


def normalize_minmax(da):
    vmin = da.min(dim="time", skipna=True)
    vmax = da.max(dim="time", skipna=True)

    diff = vmax - vmin
    diff = xr.where(diff == 0, 1, diff)

    out = (da - vmin) / diff
    out.attrs["scalar_min"] = vmin.values.tolist()
    out.attrs["scalar_max"] = vmax.values.tolist()
    out.attrs["norm_method"] = "minmax"
    return out


def normalize_zscore(da):
    mean = da.mean(dim="time", skipna=True)
    std = da.std(dim="time", skipna=True)

    std = xr.where(std == 0, 1, std)

    out = (da - mean) / std
    out.attrs["scalar_mean"] = mean.values.tolist()
    out.attrs["scalar_std"] = std.values.tolist()
    out.attrs["norm_method"] = "zscore"
    return out


def run_normalization(ds, cfg):
    print("=== NORMALIZATION ===")

    norm_cfg = cfg.get("normalization", {})
    enabled = norm_cfg.get("enabled", True)

    if not enabled:
        print("Normalization disabled.")
        return ds

    ds_out = ds.copy()

    for var in ds.data_vars:
        da = ds[var]

        if var.endswith("__was_outlier") or var.endswith("__is_interpolated"):
            continue

        if "time" not in da.dims:
            continue

        if not is_numeric(da):
            continue

        semantic = get_semantic(cfg, var)
        method = get_method_for_var(cfg, var)

        if method in [None, "none", "skip"]:
            continue

        if method == "minmax":
            ds_out[var] = normalize_minmax(da)
        elif method == "zscore":
            ds_out[var] = normalize_zscore(da)
        else:
            raise ValueError(f"Unknown normalization method '{method}' for var '{var}'")

        print(f"{var}: normalized (semantic={semantic}, method={method})")

    print("=== DONE ===")
    return ds_out
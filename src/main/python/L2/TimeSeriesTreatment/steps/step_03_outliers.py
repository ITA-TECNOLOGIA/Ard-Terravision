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


def is_enabled_for_var(cfg, var):
    out_cfg = cfg.get("outliers", {})
    override = out_cfg.get("override", {}).get(var, None)

    if override is not None and "enabled" in override:
        return bool(override["enabled"])

    semantic = get_semantic(cfg, var)

    if semantic is None:
        return False

    sem_cfg = out_cfg.get("by_semantic", {}).get(semantic, {})
    return bool(sem_cfg.get("enabled", True))


def get_params_for_var(cfg, var):
    out_cfg = cfg.get("outliers", {})

    window = out_cfg.get("window", 12)
    threshold = out_cfg.get("threshold", 6.0)
    noise_floor = out_cfg.get("noise_floor", 0.0)

    semantic = get_semantic(cfg, var)
    sem_cfg = out_cfg.get("by_semantic", {}).get(semantic, {})

    window = sem_cfg.get("window", window)
    threshold = sem_cfg.get("threshold", threshold)
    noise_floor = sem_cfg.get("noise_floor", noise_floor)

    override = out_cfg.get("override", {}).get(var, {})
    window = override.get("window", window)
    threshold = override.get("threshold", threshold)
    noise_floor = override.get("noise_floor", noise_floor)

    return window, threshold, noise_floor


def detect_outliers_mad(da, window, threshold, noise_floor):
    rolling_median = da.rolling(time=window, center=True, min_periods=1).median()
    abs_dev = np.abs(da - rolling_median)

    mad = abs_dev.rolling(time=window, center=True, min_periods=1).median()
    limit = np.maximum(threshold * (mad / 0.6745), noise_floor)

    is_outlier = abs_dev > limit
    cleaned = da.where(~is_outlier, np.nan)

    removed = int(is_outlier.sum(skipna=True))
    return cleaned, is_outlier, removed


def run_outliers(ds, cfg):
    print("=== OUTLIER DETECTION (MAD) ===")

    out_cfg = cfg.get("outliers", {})
    write_flags = bool(out_cfg.get("write_flags", False))
    if not out_cfg.get("enabled", True):
        print("Outliers disabled.")
        return ds

    ds_out = ds.copy()

    selected = []
    for var in ds.data_vars:
        da = ds[var]

        if "time" not in da.dims:
            continue
        if not is_numeric(da):
            continue

        if is_enabled_for_var(cfg, var):
            selected.append(var)

    print(f"Variables selected: {len(selected)}")
    for v in selected:
        print("  -", v, f"(semantic={get_semantic(cfg, v)})")

    for var in selected:
        da = ds[var]

        w, t, nf = get_params_for_var(cfg, var)

        cleaned, mask_outlier, removed = detect_outliers_mad(da, w, t, nf)
        ds_out[var] = cleaned

        if write_flags:
            ds_out[f"{var}__was_outlier"] = mask_outlier.fillna(False).astype(bool)

        print(f"{var}: removed {removed} (window={w}, threshold={t}, noise_floor={nf})")

    print("=== DONE ===")
    return ds_out
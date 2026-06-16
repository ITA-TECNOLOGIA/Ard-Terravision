import xarray as xr
import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_diff_features(ds, cfg):
    diff_cfg = cfg.get("features", {}).get("diff", {})
    suffix = diff_cfg.get("suffix", "diff")
    variables = diff_cfg.get("variables", [])

    created = []

    for var in variables:
        if var not in ds.data_vars:
            continue

        da = ds[var]
        if "time" not in da.dims:
            continue

        new_name = f"{var}_{suffix}"
        dvar = da.diff("time")
        dt = ds["time"].diff("time").astype(float) / 1e9

        out = dvar / dt
        out = out.reindex(time=ds["time"])

        ds[new_name] = out
        ds[new_name].attrs["units"] = f"{da.attrs.get('units','')}/s"
        ds[new_name].attrs["feature_type"] = "diff"
        ds[new_name].attrs["source_var"] = var

        created.append(new_name)

    return ds, created


def add_magnitude_features(ds, cfg):
    mag_cfg = cfg.get("features", {}).get("magnitude", {})
    groups = mag_cfg.get("groups", [])

    created = []

    for g in groups:
        name = g.get("name", None)
        vars_ = g.get("vars", [])

        if name is None or len(vars_) < 2:
            continue

        if not all(v in ds.data_vars for v in vars_):
            continue

        # magnitude = sqrt(sum(v^2))
        arr = None
        for v in vars_:
            if arr is None:
                arr = ds[v] ** 2
            else:
                arr = arr + (ds[v] ** 2)

        ds[name] = np.sqrt(arr)
        ds[name].attrs["feature_type"] = "magnitude"
        ds[name].attrs["source_vars"] = ",".join(vars_)

        created.append(name)

    return ds, created

def add_geospatial_features(ds, cfg):
    geo_cfg = cfg.get("features", {}).get("geospatial", {})
    coord_sets = geo_cfg.get("coordinate_sets", [])

    created = []

    if "sensor" not in ds.dims and "sensor" not in ds.coords:
        print("Geospatial: no sensor dimension found -> skipping")
        return ds, created

    for cs in coord_sets:

        name = cs["name"]
        lat_var = cs["lat"]
        lon_var = cs["lon"]
        alt_var = cs["alt"]

        if not all(v in ds.data_vars for v in [lat_var, lon_var, alt_var]):
            continue

        print(f"Geospatial: processing {name}")

        lat = ds[lat_var]
        lon = ds[lon_var]
        alt = ds[alt_var]

        lat0_vals = []
        lon0_vals = []
        alt0_vals = []

        for sensor in ds["sensor"].values:

            lat_s = lat.sel(sensor=sensor).values
            lon_s = lon.sel(sensor=sensor).values
            alt_s = alt.sel(sensor=sensor).values

            valid = (
                ~np.isnan(lat_s)
                & ~np.isnan(lon_s)
                & ~np.isnan(alt_s)
            )

            idx = np.where(valid)[0]

            if len(idx) == 0:

                lat0_vals.append(np.nan)
                lon0_vals.append(np.nan)
                alt0_vals.append(np.nan)

                continue

            i0 = idx[0]

            lat0_vals.append(lat_s[i0])
            lon0_vals.append(lon_s[i0])
            alt0_vals.append(alt_s[i0])

        lat0 = xr.DataArray(
            np.array(lat0_vals),
            coords={"sensor": ds["sensor"].values},
            dims=["sensor"]
        )

        lon0 = xr.DataArray(
            np.array(lon0_vals),
            coords={"sensor": ds["sensor"].values},
            dims=["sensor"]
        )

        alt0 = xr.DataArray(
            np.array(alt0_vals),
            coords={"sensor": ds["sensor"].values},
            dims=["sensor"]
        )

        print("\nREFERENCE VALUES:")

        for sensor in ds["sensor"].values:

            print(
                f"{sensor} | "
                f"lat0={lat0.sel(sensor=sensor).item()} | "
                f"lon0={lon0.sel(sensor=sensor).item()} | "
                f"alt0={alt0.sel(sensor=sensor).item()}"
            )

        R = 111320.0

        lat0_rad = xr.apply_ufunc(np.deg2rad, lat0)

        x = (lon - lon0) * R * np.cos(lat0_rad)
        y = (lat - lat0) * R
        z = alt - alt0

        x_name = f"{name}_x_m"
        y_name = f"{name}_y_m"
        z_name = f"{name}_z_m"

        ds[x_name] = x
        ds[y_name] = y
        ds[z_name] = z

        dx = x.diff("time")
        dy = y.diff("time")
        dz = z.diff("time")

        ds[f"{name}_dx_m"] = dx
        ds[f"{name}_dy_m"] = dy
        ds[f"{name}_dz_m"] = dz

        disp = np.sqrt(dx**2 + dy**2 + dz**2)
        ds[f"{name}_disp_m"] = disp

        ds[f"{name}_cumdisp_m"] = disp.cumsum("time")

        ds[f"{name}_hdisp_m"] = np.sqrt(dx**2 + dy**2)
        ds[f"{name}_vdisp_m"] = np.abs(dz)

        created.extend([
            x_name, y_name, z_name,
            f"{name}_dx_m", f"{name}_dy_m", f"{name}_dz_m",
            f"{name}_disp_m",
            f"{name}_cumdisp_m",
            f"{name}_hdisp_m",
            f"{name}_vdisp_m"
        ])

    return ds, created

def run_feature_engineering(ds, cfg):
    print("=== FEATURE ENGINEERING ===")

    feat_cfg = cfg.get("features", {})
    enabled = feat_cfg.get("enabled", True)

    if not enabled:
        print("Feature engineering disabled.")
        return ds

    pipelines = feat_cfg.get("pipelines", [])

    ds_out = ds.copy()

    all_created = []

    for pipe in pipelines:
        if pipe == "diff":
            ds_out, created = add_diff_features(ds_out, cfg)
            all_created.extend(created)

        elif pipe == "magnitude":
            ds_out, created = add_magnitude_features(ds_out, cfg)
            all_created.extend(created)

        elif pipe == "geospatial":
            ds_out, created = add_geospatial_features(ds_out, cfg)
            all_created.extend(created)

        else:
            raise ValueError(f"Unknown feature pipeline '{pipe}'")

    print(f"-> Created {len(all_created)} features.")
    for v in all_created:
        print("   -", v)

    print("=== DONE ===")
    return ds_out
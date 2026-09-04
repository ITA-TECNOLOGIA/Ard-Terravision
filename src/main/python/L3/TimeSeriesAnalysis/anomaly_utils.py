import xarray as xr
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("AnomalyDetection")

def _compute_rolling_score(da, window=50, method="zscore", time_dim="time"):
    if method == "zscore":
        center = da.rolling({time_dim: window}, min_periods=window // 2).mean()
        spread = da.rolling({time_dim: window}, min_periods=window // 2).std()
    elif method == "mad":
        center = da.rolling({time_dim: window}, min_periods=window // 2).median()
        abs_dev = np.abs(da - center)
        spread = abs_dev.rolling({time_dim: window}, min_periods=window // 2).median()
        spread = spread * 1.4826
    else:
        raise ValueError(f"Unknown method: {method}")

    center = center.shift({time_dim: 1})
    spread = spread.shift({time_dim: 1})
    scale = np.nanmax(np.abs(da.values))
    eps = max(np.finfo(float).eps, scale * 1e-6) if np.isfinite(scale) and scale > 0 else np.finfo(float).eps
    spread = spread.where(spread > eps, eps)
    z = (da - center) / spread
    return z.clip(-50, 50)


def _detect_anomalies(z, theta=4.0):
    return np.abs(z) > theta


def _resolve_theta(theta, var, default=4.0):
    if isinstance(theta, dict):
        if var in theta:
            return theta[var]
        base_var = var[:-5] if var.endswith("_diff") else var
        if base_var in theta:
            return theta[base_var]
        return default
    return theta


def run_zscore_detection(ds, variables, entity_dim="sensor", time_dim="time",
                         window=50, theta=4.0, method="zscore"):
    ds_out = ds.copy()
    all_events = []
    has_entities = entity_dim in ds.dims

    for var in variables:
        var_theta = _resolve_theta(theta, var)

        if var not in ds:
            logger.warning(f"[zscore detection] {var} not found in dataset, skipping")
            continue

        da = ds[var]

        if has_entities and entity_dim in da.dims:
            z_list = []
            mask_list = []
            for ent in ds[entity_dim].values:
                da_ent = da.sel({entity_dim: ent})
                z_ent = _compute_rolling_score(da_ent, window=window, method=method, time_dim=time_dim)
                mask_ent = _detect_anomalies(z_ent, var_theta)

                z_list.append(z_ent)
                mask_list.append(mask_ent)

                z_anomalies = z_ent.where(mask_ent, drop=True)
                if z_anomalies.size > 0:
                    df = z_anomalies.to_dataframe(name="zscore").reset_index()
                    df["anomaly"] = True
                    df["variable"] = var
                    df[entity_dim] = str(ent)
                    all_events.append(df)

            z = xr.concat(z_list, dim=entity_dim).transpose(*da.dims)
            mask = xr.concat(mask_list, dim=entity_dim).transpose(*da.dims)
        else:
            z = _compute_rolling_score(da, window=window, method=method, time_dim=time_dim)
            mask = _detect_anomalies(z, var_theta)

            mask_vals = mask.values if hasattr(mask, "values") else np.asarray(mask)
            z_vals = z.values if hasattr(z, "values") else np.asarray(z)
            time_vals = z[time_dim].values

            df = pd.DataFrame({
                "time": time_vals[mask_vals],
                "variable": var,
                "zscore": z_vals[mask_vals],
                "anomaly": True
            })
            if not df.empty:
                all_events.append(df)

        ds_out[f"{var}_zscore"] = z
        ds_out[f"{var}_zscore_anomaly"] = mask.astype(int)
        ds_out[f"{var}_zscore_anomaly"].attrs["theta"] = var_theta

        n_var_anomalies = int(np.sum(mask.values == 1))
        logger.info(f"[zscore] {var}: theta={var_theta}, anomalies={n_var_anomalies}")

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    zscore_mask_vars = [v for v in ds_out.data_vars if v.endswith("_zscore_anomaly")]
    n_zscore_anomalies = int(sum(np.sum(ds_out[v].values == 1) for v in zscore_mask_vars))

    logger.info(f"Z-score detection: {n_zscore_anomalies} anomalies found across {len(variables)} variables")
    return ds_out, events_df


def train_and_predict_iforest(ds, features, entity_dim="sensor", time_dim="time",
                              contamination=0.005, n_estimators=250, n_jobs=-1):
    ds_out = ds.copy()
    all_events = []

    ds_out["iforest_anomaly"] = xr.full_like(ds[features[0]], 0, dtype=int)
    ds_out["iforest_score"] = xr.full_like(ds[features[0]], 0.0, dtype=float)

    has_entities = entity_dim in ds.dims or entity_dim in ds.coords
    entities = ds[entity_dim].values if has_entities else [None]

    for ent in entities:
        if has_entities:
            ds_ent = ds.sel({entity_dim: ent})
            df_features = ds_ent[features].to_dataframe()[features].dropna()
        else:
            df_features = ds[features].to_dataframe()[features].dropna()

        if len(df_features) < 15:
            logger.warning(f"Sensor/Entity {ent} discarded due to insufficient data ({len(df_features)} points).")
            continue

        model = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                                random_state=42, n_jobs=n_jobs)
        model.fit(df_features)

        scores = -model.decision_function(df_features)
        anomalies_binary = np.where(scores > 0.0, 1, 0)

        times_with_data = df_features.index
        if has_entities:
            ds_out["iforest_anomaly"].loc[{entity_dim: ent, time_dim: times_with_data}] = anomalies_binary
            ds_out["iforest_score"].loc[{entity_dim: ent, time_dim: times_with_data}] = scores
        else:
            ds_out["iforest_anomaly"].loc[{time_dim: times_with_data}] = anomalies_binary
            ds_out["iforest_score"].loc[{time_dim: times_with_data}] = scores

        df_anom = df_features[anomalies_binary == 1].reset_index()
        if not df_anom.empty:
            if has_entities:
                df_anom[entity_dim] = str(ent)
            df_anom["anomaly_score"] = scores[anomalies_binary == 1]
            df_anom["variable"] = ", ".join(features)
            all_events.append(df_anom)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return ds_out, events_df


def compute_zscore_features(ds, base_features, window=50, time_dim="time"):
    new_vars = []

    for var in base_features:
        if var not in ds.data_vars:
            logger.warning(f"zscore feature '{var}' not found in dataset, skipping")
            continue

        da = ds[var]
        rolling_mean = da.rolling({time_dim: window}, min_periods=window // 2).mean()
        rolling_std = da.rolling({time_dim: window}, min_periods=window // 2).std()
        rolling_mean = rolling_mean.shift({time_dim: 1})
        rolling_std = rolling_std.shift({time_dim: 1})
        scale = np.nanmax(np.abs(da.values))
        eps = max(np.finfo(float).eps, scale * 1e-6) if np.isfinite(scale) and scale > 0 else np.finfo(float).eps
        rolling_std = rolling_std.where(rolling_std > eps, eps)

        zscore_var = (da - rolling_mean) / rolling_std

        zscore_name = f"{var}_zscore_w{window}"
        ds[zscore_name] = zscore_var
        ds[zscore_name].attrs["feature_type"] = "rolling_zscore"
        ds[zscore_name].attrs["source_var"] = var
        ds[zscore_name].attrs["window"] = window
        new_vars.append(zscore_name)
        logger.info(f"Created rolling z-score feature: {zscore_name}")

    return ds, new_vars

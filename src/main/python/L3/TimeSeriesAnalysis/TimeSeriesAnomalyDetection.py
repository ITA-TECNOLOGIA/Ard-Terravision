# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from typing import List, Any, Optional, Dict, Union

from L3.L3_Algorithm import L3_Algorithm, L3_result
from L3.TimeSeriesAnalysis import anomaly_utils
from logger import logger

# ITA COLOR PALLETTE
#COLOR_MAIN = "#0000FF"
#COLOR_SECONDARY = "#7979FF"
#COLOR_ALERT = "#FF0000"

# TERRAVISION COLOR PALLETTE
COLOR_MAIN = "#779887"
COLOR_SECONDARY = "#ABD6BD"
COLOR_ALERT = "#3B524A"


class TimeSeriesAnomalyDetection(L3_Algorithm):
    def __init__(self,
                 features: List[str],
                 contamination: float = 0.005,
                 n_estimators: int = 200,
                 view_var: Union[str, List[str], None] = None,
                 zscore_detection: Optional[Dict] = None,
                 iforest_with_zscore: bool = False,
                 iforest_without_zscore: bool = True,
                 window: int = 50,
                 show_anomaly_table: bool = False):
        super().__init__()
        self.features = features
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.view_var = view_var
        self.zscore_detection = zscore_detection or {}
        self.iforest_with_zscore = iforest_with_zscore
        self.iforest_without_zscore = iforest_without_zscore
        self.window = window
        self.show_anomaly_table = show_anomaly_table
        self._anomaly_tables: Dict[str, Any] = {}

    def _resolve_dataset(self, l1_input, l2_datacube: Optional[xr.Dataset] = None) -> xr.Dataset:
        if l2_datacube is not None:
            logger.info("Using processed L2 datacube as input")
            return l2_datacube
        logger.info("No L2 processed dataset found; using raw L1 datacube")
        return l1_input.get_datacube()

    def _collect_valid_sensors(self, ds):
        if "sensor" not in ds.dims:
            return [None]
        sensors = []
        for s in ds["sensor"].values:
            try:
                for f in self.features:
                    if f in ds.data_vars:
                        if ds[f].sel(sensor=s).notnull().sum() > 0:
                            sensors.append(s)
                            break
            except Exception:
                continue
        return sensors

    def _short_sensor_name(self, sensor_name):
        """Return sensor name as-is for display in plot titles."""
        return str(sensor_name) if sensor_name is not None else "Global"

    # iForest debug image: sensors × methods grid

    def _add_iforest_to_figure(self, fig, panels, view_vars, sensors, total_rows, start_idx):
        valid = [(t, d) for t, d in panels if d is not None and len(d.data_vars) > 0]
        if not valid:
            return

        n_sensors = len(sensors)
        if n_sensors == 0:
            return

        idx = start_idx
        for panel_title, ds in valid:
            for view_var in view_vars:
                for sensor_name in sensors:
                    ax = fig.add_subplot(total_rows, 1, idx)
                    self._plot_iforest_cell(ax, ds, sensor_name, view_var)
                    title = f"[{self._short_sensor_name(sensor_name)}] {view_var} ({panel_title})" if sensor_name else f"{view_var} ({panel_title})"
                    ax.set_title(title, fontsize=10, fontweight="bold", color="#333333")
                    idx += 1

    def _plot_iforest_cell(self, ax, ds, sensor_name, view_var):
        da = ds[view_var] if sensor_name is None else ds[view_var].sel(sensor=sensor_name)
        ax.plot(da["time"].values, da.values, color=COLOR_MAIN, alpha=0.85,
                linewidth=1.0, label=view_var)
        if "iforest_anomaly" in ds.data_vars:
            am = ds["iforest_anomaly"]
            if sensor_name is not None:
                am = am.sel(sensor=sensor_name)
            mask = am.values == 1
            if np.any(mask):
                ax.scatter(da["time"].values[mask], da.values[mask],
                           color=COLOR_ALERT, edgecolor="#000000", linewidths=0.5,
                           s=40, zorder=5, label=f"iForest ({mask.sum()})")
        ax.legend(loc="upper left", frameon=True, fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_tick_params(rotation=20)

    # Z-score panels (appended to shared figure)

    def _add_zscore_to_figure(self, fig, ds_zscore, zscore_variables, theta, total_rows, start_idx, valid_sensors=None):

        if valid_sensors is not None:
            sensors = valid_sensors
        elif "sensor" in ds_zscore.dims:
            sensors = ds_zscore["sensor"].values
        else:
            sensors = [None]

        n_sensors = len(sensors)
        subplot_idx = start_idx

        for input_var in zscore_variables:
            base_var = input_var[:-5] if input_var.endswith("_diff") else input_var
            diff_var = input_var if input_var.endswith("_diff") else f"{input_var}_diff"
            zscore_var = f"{diff_var}_zscore"
            anom_var = f"{diff_var}_zscore_anomaly"

            if isinstance(theta, dict):
                var_theta = theta.get(input_var, theta.get(base_var, 3.5))
            else:
                var_theta = theta

            if zscore_var not in ds_zscore.data_vars:
                continue

            for s_idx, sensor_name in enumerate(sensors):
                sid = self._short_sensor_name(sensor_name)
                sensor_prefix = f"[{sid}] " if sensor_name else ""

                # Top row: original variable
                ax_top = fig.add_subplot(total_rows, 1, subplot_idx)
                subplot_idx += 1
                da_base = ds_zscore[base_var] if sensor_name is None else ds_zscore[base_var].sel(sensor=sensor_name)
                da_base = da_base.dropna("time")
                ax_top.plot(da_base["time"].values, da_base.values,
                            color=COLOR_MAIN, linewidth=1.0, alpha=0.85)

                if anom_var in ds_zscore.data_vars:
                    am = ds_zscore[anom_var] if sensor_name is None else ds_zscore[anom_var].sel(sensor=sensor_name)
                    anomalies_only = da_base.where(am == True, drop=True)
                    if len(anomalies_only) > 0:
                        ax_top.scatter(anomalies_only["time"].values, anomalies_only.values,
                                       color=COLOR_ALERT, s=15, zorder=5, label="Anomaly")

                ax_top.set_title(f"{sensor_prefix}{base_var} (original)", fontsize=11, fontweight="bold")
                ax_top.grid(True, alpha=0.3)
                ax_top.tick_params(labelsize=8)

                # Bottom row: rolling z-score
                ax_bot = fig.add_subplot(total_rows, 1, subplot_idx)
                subplot_idx += 1
                da_z = ds_zscore[zscore_var] if sensor_name is None else ds_zscore[zscore_var].sel(sensor=sensor_name)
                ax_bot.plot(da_z["time"].values, da_z.values,
                            color=COLOR_SECONDARY, linewidth=1.0, alpha=0.85)

                ax_bot.axhline(var_theta, color=COLOR_ALERT, linestyle="--", linewidth=0.8)
                ax_bot.axhline(-var_theta, color=COLOR_ALERT, linestyle="--", linewidth=0.8)

                if anom_var in ds_zscore.data_vars:
                    am = ds_zscore[anom_var] if sensor_name is None else ds_zscore[anom_var].sel(sensor=sensor_name)
                    z_anomalies_only = da_z.where(am == True, drop=True)
                    if len(z_anomalies_only) > 0:
                        ax_bot.scatter(z_anomalies_only["time"].values, z_anomalies_only.values,
                                       color=COLOR_ALERT, s=15, zorder=5)

                ax_bot.set_title(f"{sensor_prefix}{diff_var} (z-score θ=±{var_theta})", fontsize=10)
                ax_bot.grid(True, alpha=0.3)
                ax_bot.tick_params(labelsize=8)

    # Main process

    def process_data(self, l1_input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        logger.info("=== TimeSeriesAnomalyDetection: STARTING ===")

        ds = self._resolve_dataset(l1_input, l2_datacube)
        ds_working = ds.copy()

        # === Z-score spike detection ===
        zscore_cfg = self.zscore_detection
        ds_zscore = None
        n_zscore_anomalies = 0
        zscore_vars = []

        if zscore_cfg.get("enabled", False):
            zscore_vars_raw = zscore_cfg.get("variables", [])
            zscore_theta = zscore_cfg.get("theta", 3.5)
            zscore_method = zscore_cfg.get("method", "mad")
            zscore_window = zscore_cfg.get("window", self.window)

            # Smart variable resolver: auto-append _diff if not already present
            zscore_vars = []
            for v in zscore_vars_raw:
                diff_name = v if v.endswith("_diff") else f"{v}_diff"
                if diff_name in ds_working.data_vars:
                    zscore_vars.append(diff_name)
                elif v in ds_working.data_vars:
                    zscore_vars.append(v)
                else:
                    logger.warning(f"Variable '{v}' (or '{diff_name}') not found, skipping")
            zscore_cfg["resolved_variables"] = zscore_vars

            logger.info(f"Z-score detection: vars={zscore_vars}, theta={zscore_theta}, "
                         f"method={zscore_method}, window={zscore_window}")

            ds_zscore, zscore_events_df = anomaly_utils.run_zscore_detection(
                ds=ds_working,
                variables=zscore_vars,
                entity_dim="sensor",
                window=zscore_window,
                theta=zscore_theta,
                method=zscore_method
            )
            if self.show_anomaly_table and not zscore_events_df.empty:
                self._anomaly_tables["zscore"] = zscore_events_df
            mask_vars = [v for v in ds_zscore.data_vars if v.endswith("_zscore_anomaly")]
            n_zscore_anomalies = int(sum(np.sum(ds_zscore[v].values == 1) for v in mask_vars))
            logger.info(f"Z-score anomalies detected: {n_zscore_anomalies}")

        # === Compute z-score features for iForest ===
        zscore_feature_vars = zscore_cfg.get("variables", [])
        new_zscore_vars = []
        if zscore_feature_vars:
            ds_working, new_zscore_vars = anomaly_utils.compute_zscore_features(
                ds=ds_working,
                base_features=zscore_feature_vars,
                window=self.window
            )

        # === Determine view_vars ===
        view_vars = self.view_var
        if view_vars is None:
            view_vars = self.features
        if isinstance(view_vars, str):
            view_vars = [view_vars]
        view_vars = [v for v in view_vars if v in ds_working.data_vars]
        if not view_vars and self.features:
            view_vars = [f for f in self.features if f in ds_working.data_vars][:1]

        # === iForest panels ===
        iforest_panels = []

        if self.iforest_without_zscore and self.features:
            base_feats = [f for f in self.features if f in ds.data_vars]
            if base_feats:
                logger.info(f"iForest (no zscore): {base_feats}")
                ds_if, if_events_df = anomaly_utils.train_and_predict_iforest(
                    ds=ds, features=base_feats,
                    contamination=self.contamination, n_estimators=self.n_estimators
                )
                n = int(ds_if.get("iforest_anomaly", 0).sum())
                logger.info(f"iForest (no zscore) anomalies: {n}")
                iforest_panels.append(("iforest (w/o z-score)", ds_if))
                if self.show_anomaly_table and not if_events_df.empty:
                    self._anomaly_tables["iforest_no_zscore"] = if_events_df

        if self.iforest_with_zscore and self.features:
            all_feats = list(self.features) + [v for v in new_zscore_vars if v in ds_working.data_vars]
            if all_feats:
                logger.info(f"iForest (with zscore): {all_feats}")
                ds_if2, if2_events_df = anomaly_utils.train_and_predict_iforest(
                    ds=ds_working, features=all_feats,
                    contamination=self.contamination, n_estimators=self.n_estimators
                )
                n = int(ds_if2.get("iforest_anomaly", 0).sum())
                logger.info(f"iForest (with zscore) anomalies: {n}")
                iforest_panels.append(("iforest (w/ z-score)", ds_if2))
                if self.show_anomaly_table and not if2_events_df.empty:
                    self._anomaly_tables["iforest_with_zscore"] = if2_events_df

        # === Build single merged figure ===
        sensors = self._collect_valid_sensors(ds)
        n_sensors_count = len(sensors) if sensors and sensors[0] is not None else (0 if "sensor" in ds.dims else 1)

        zscore_n_rows = 0
        zscore_plot_vars = zscore_vars
        if ds_zscore is not None and zscore_cfg.get("enabled") and zscore_vars:
            zscore_view_vars = zscore_cfg.get("view_var", zscore_vars)
            if isinstance(zscore_view_vars, str):
                zscore_view_vars = [zscore_view_vars]
            zscore_plot_vars = [v for v in zscore_vars if v in zscore_view_vars
                                 or v.replace("_diff", "") in zscore_view_vars
                                 or any(v.startswith(f"{zv}_") or v.startswith(f"{zv}") for zv in zscore_view_vars)]
            if not zscore_plot_vars:
                zscore_plot_vars = zscore_vars
            zscore_n_rows = len(zscore_plot_vars) * n_sensors_count * 2

        iforest_rows = len(iforest_panels) * len(view_vars) * n_sensors_count if iforest_panels else 0

        total_rows = zscore_n_rows + iforest_rows
        if total_rows == 0 or n_sensors_count == 0:
            return [L3_result(
                debug_image=Image.new("RGB", (400, 100), (255, 255, 255)),
                algorithm_results={"error": "No valid sensors or anomaly detection methods enabled"}
            )]

        zscore_height = zscore_n_rows * 2.5 if zscore_n_rows > 0 else 0
        iforest_height = iforest_rows * 3.0 if iforest_rows > 0 else 0
        total_height = zscore_height + iforest_height

        fig = plt.figure(figsize=(14, total_height))

        # Z-score section
        if ds_zscore is not None:
            zscore_theta = zscore_cfg.get("theta", 3.5)
            self._add_zscore_to_figure(fig, ds_zscore, zscore_plot_vars, zscore_theta, total_rows, 1, valid_sensors=sensors)

        # iForest section
        if iforest_panels:
            iforest_start = zscore_n_rows + 1
            self._add_iforest_to_figure(fig, iforest_panels, view_vars, sensors, total_rows, iforest_start)

        fig.suptitle("TimeSeries Anomaly Detection", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        result_dict = {
            "n_zscore_anomalies": n_zscore_anomalies,
            "n_iforest_panels": len(iforest_panels),
            "features": self.features,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "zscore_detection_enabled": zscore_cfg.get("enabled", False),
            "view_var": view_vars,
            "zscore_view_var": zscore_cfg.get("view_var", zscore_cfg.get("variables", [])),
            "zscore_variables": zscore_cfg.get("resolved_variables", zscore_cfg.get("variables", [])),
            "theta": zscore_cfg.get("theta", 3.5),
        }
        if self.show_anomaly_table and self._anomaly_tables:
            result_dict["anomaly_tables"] = self._anomaly_tables

        try:
            from utils.output_writer import build_output_path, save_datacube, get_timestamp

            ds_to_save = ds.copy()

            if ds_zscore is not None:
                for v in ds_zscore.data_vars:
                    if v not in ds_to_save.data_vars:
                        ds_to_save[v] = ds_zscore[v]

            if iforest_panels:
                for panel_title, ds_if in iforest_panels:
                    uses_zscore = "w/ z-score" in panel_title
                    suffix_anom = "_iforest_w_zscore_anomaly" if uses_zscore else "_iforest_wo_zscore_anomaly"
                    suffix_score = "_iforest_w_zscore_score" if uses_zscore else "_iforest_wo_zscore_score"

                    if "iforest_anomaly" not in ds_if.data_vars:
                        continue

                    for f in self.features:
                        base_var = f[:-5] if f.endswith("_diff") else f
                        if base_var in ds_if.data_vars:
                            ds_to_save[f"{base_var}{suffix_anom}"] = ds_if["iforest_anomaly"]
                            ds_to_save[f"{base_var}{suffix_score}"] = ds_if["iforest_score"]

            nc_filename = f"datacube_{get_timestamp()}.nc"
            nc_path = build_output_path(self, nc_filename)
            logger.info(f"Saving processed anomaly datacube to: {nc_path}")
            save_datacube(ds_to_save, nc_path)

        except Exception as e:
            logger.error(f"Failed to automatically save processed NetCDF datacube: {e}")

        return [L3_result(
            debug_image=Image.open(buf),
            algorithm_results=result_dict
        )]

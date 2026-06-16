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
from typing import List, Any, Optional, Dict

from L3.L3_Algorithm import L3_Algorithm, L3_result
from L3.TimeSeriesAnalysis import anomaly_utils
from logger import logger

COLOR_MAIN = "#0000FF"
COLOR_SECONDARY = "#7979FF"
COLOR_ALERT = "#FF0000"


class TimeSeriesAnomalyDetection(L3_Algorithm):
    def __init__(self,
                 features: List[str],
                 contamination: float = 0.005,
                 n_estimators: int = 200,
                 view_var: Optional[str] = None,
                 zscore_detection: Optional[Dict] = None,
                 iforest_with_zscore: bool = False,
                 iforest_without_zscore: bool = True,
                 window: int = 50):
        super().__init__()
        self.features = features
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.view_var = view_var
        self.zscore_detection = zscore_detection or {}
        self.iforest_with_zscore = iforest_with_zscore
        self.iforest_without_zscore = iforest_without_zscore
        self.window = window

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

    # iForest debug image: sensors × methods grid

    def _add_iforest_to_figure(self, fig, gs, panels, view_var, sensors, row_offset):
        valid = [(t, d) for t, d in panels if d is not None and len(d.data_vars) > 0]
        if not valid:
            return 0

        n_cols = len(valid)
        n_rows = len(sensors)
        if n_rows == 0:
            return 0

        inner = gs.subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
        for col_idx, (panel_title, ds) in enumerate(valid):
            for row_idx, sensor_name in enumerate(sensors):
                ax = fig.add_subplot(inner[row_idx, col_idx])
                self._plot_iforest_cell(ax, ds, sensor_name, view_var)
                if row_idx == 0:
                    ax.set_title(panel_title, fontsize=10, fontweight="bold", color="#333333")

        return n_rows

    def _plot_iforest_cell(self, ax, ds, sensor_name, view_var):
        da = ds[view_var] if sensor_name is None else ds[view_var].sel(sensor=sensor_name)
        ax.plot(da["time"].values, da.values, color="#1f77b4", alpha=0.85,
                linewidth=1.0, label=view_var)
        if "iforest_anomaly" in ds.data_vars:
            am = ds["iforest_anomaly"]
            if sensor_name is not None:
                am = am.sel(sensor=sensor_name)
            mask = am.values == 1
            if np.any(mask):
                ax.scatter(da["time"].values[mask], da.values[mask],
                           color="#d62728", edgecolor="#000000", linewidths=0.5,
                           s=40, zorder=5, label=f"iForest ({mask.sum()})")
        short = str(sensor_name or "")
        if len(short) > 25:
            short = short[:10] + "..." + short[-10:]
        ax.set_title(short or "Global", fontsize=9, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_tick_params(rotation=20)

    # Z-score panels (appended to shared figure)

    def _add_zscore_to_figure(self, fig, gs, ds_zscore, zscore_variables, theta, row_offset):

        if "sensor" in ds_zscore.dims:
            sensors = ds_zscore["sensor"].values
        else:
            sensors = [None]

        total_rows = 0

        for input_var in zscore_variables:
            base_var = input_var[:-5] if input_var.endswith("_diff") else input_var
            diff_var = input_var if input_var.endswith("_diff") else f"{input_var}_diff"
            zscore_var = f"{diff_var}_zscore"
            anom_var = f"{diff_var}_zscore_anomaly"

            if zscore_var not in ds_zscore.data_vars:
                continue

            n_sensors = len(sensors)

            inner = gs.subgridspec(n_sensors * 2, 1, hspace=0.4)

            for s_idx, sensor_name in enumerate(sensors):
                row_top = s_idx * 2
                row_bot = s_idx * 2 + 1

                # Top row: original variable
                ax_top = fig.add_subplot(inner[row_top, 0])
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

                sn = str(sensor_name) if sensor_name is not None else ""
                if len(sn) > 25:
                    sn = sn[:10] + "..." + sn[-10:]
                lab = f"{base_var} ({sn})" if sn else base_var
                ax_top.set_title(f"{lab} — original", fontsize=11, fontweight="bold")
                ax_top.grid(True, alpha=0.3)
                ax_top.tick_params(labelsize=8)

                # Bottom row: rolling z-score
                ax_bot = fig.add_subplot(inner[row_bot, 0])
                da_z = ds_zscore[zscore_var] if sensor_name is None else ds_zscore[zscore_var].sel(sensor=sensor_name)
                ax_bot.plot(da_z["time"].values, da_z.values,
                            color=COLOR_SECONDARY, linewidth=1.0, alpha=0.85)

                ax_bot.axhline(theta, color=COLOR_ALERT, linestyle="--", linewidth=0.8)
                ax_bot.axhline(-theta, color=COLOR_ALERT, linestyle="--", linewidth=0.8)

                if anom_var in ds_zscore.data_vars:
                    am = ds_zscore[anom_var] if sensor_name is None else ds_zscore[anom_var].sel(sensor=sensor_name)
                    z_anomalies_only = da_z.where(am == True, drop=True)
                    if len(z_anomalies_only) > 0:
                        ax_bot.scatter(z_anomalies_only["time"].values, z_anomalies_only.values,
                                       color=COLOR_ALERT, s=15, zorder=5)

                ax_bot.set_title(f"{diff_var} — rolling z-score θ=±{theta}", fontsize=10)
                ax_bot.grid(True, alpha=0.3)
                ax_bot.tick_params(labelsize=8)

            total_rows += n_sensors * 2

        return total_rows

    # Main process

    def process_data(self, l1_input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        logger.info("=== TimeSeriesAnomalyDetection: STARTING ===")

        ds = self._resolve_dataset(l1_input, l2_datacube)
        ds_working = ds.copy()

        # === Z-score spike detection ===
        zscore_cfg = self.zscore_detection
        ds_zscore = None
        n_zscore_anomalies = 0

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

            ds_zscore, _ = anomaly_utils.run_zscore_detection(
                ds=ds_working,
                variables=zscore_vars,
                entity_dim="sensor",
                window=zscore_window,
                theta=zscore_theta,
                method=zscore_method
            )
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

        # === Determine view_var ===
        view_var = self.view_var
        if view_var is None:
            for f in self.features:
                if f in ds_working.data_vars:
                    view_var = f
                    break

        # === iForest panels ===
        iforest_panels = []

        if self.iforest_without_zscore and self.features:
            base_feats = [f for f in self.features if f in ds.data_vars]
            if base_feats:
                logger.info(f"iForest (no zscore): {base_feats}")
                ds_if, _ = anomaly_utils.train_and_predict_iforest(
                    ds=ds, features=base_feats,
                    contamination=self.contamination, n_estimators=self.n_estimators
                )
                n = int(ds_if.get("iforest_anomaly", 0).sum())
                logger.info(f"iForest (no zscore) anomalies: {n}")
                iforest_panels.append(("iForest\n(no z-score)", ds_if))

        if self.iforest_with_zscore and self.features:
            all_feats = list(self.features) + [v for v in new_zscore_vars if v in ds_working.data_vars]
            if all_feats:
                logger.info(f"iForest (with zscore): {all_feats}")
                ds_if2, _ = anomaly_utils.train_and_predict_iforest(
                    ds=ds_working, features=all_feats,
                    contamination=self.contamination, n_estimators=self.n_estimators
                )
                n = int(ds_if2.get("iforest_anomaly", 0).sum())
                logger.info(f"iForest (with zscore) anomalies: {n}")
                iforest_panels.append(("iForest\n(with z-score)", ds_if2))

        # === Build single merged figure ===
        sensors = self._collect_valid_sensors(ds)
        n_sensors_count = len(sensors) if sensors and sensors[0] is not None else 1

        zscore_n_rows = 0
        if ds_zscore is not None and zscore_cfg.get("enabled") and zscore_vars:
            zscore_n_rows = len(zscore_vars) * n_sensors_count * 2

        iforest_rows = n_sensors_count if iforest_panels else 0

        total_rows = zscore_n_rows + iforest_rows
        if total_rows == 0:
            return [L3_result(
                debug_image=Image.new("RGB", (400, 100), (255, 255, 255)),
                algorithm_results={"error": "No anomaly detection methods enabled"}
            )]

        # 5 height units per z-score row, 3.5 per iforest row
        zscore_height = zscore_n_rows * 2.5 if zscore_n_rows > 0 else 0
        iforest_height = iforest_rows * 3.5 if iforest_rows > 0 else 0
        total_height = zscore_height + iforest_height

        fig = plt.figure(figsize=(14, total_height))
        gs = fig.add_gridspec(min(total_rows, max(1, total_rows)), 1, hspace=0.5)

        row = 0

        # Z-score section
        if ds_zscore is not None:
            zscore_theta = zscore_cfg.get("theta", 3.5)
            z_gs = gs[row:row + zscore_n_rows]
            added = self._add_zscore_to_figure(fig, z_gs, ds_zscore, zscore_vars, zscore_theta, row)
            row += zscore_n_rows

        # iForest section
        if iforest_panels:
            if_gs = gs[row:row + iforest_rows]
            self._add_iforest_to_figure(fig, if_gs, iforest_panels, view_var, sensors, row)

        fig.suptitle("TimeSeries Anomaly Detection", fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return [L3_result(
            debug_image=Image.open(buf),
            algorithm_results={
                "n_zscore_anomalies": n_zscore_anomalies,
                "n_iforest_panels": len(iforest_panels),
                "features": self.features,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "zscore_detection_enabled": zscore_cfg.get("enabled", False),
                "view_var": view_var,
                "zscore_variables": zscore_cfg.get("resolved_variables", zscore_cfg.get("variables", [])),
                "theta": zscore_cfg.get("theta", 3.5)
            }
        )]

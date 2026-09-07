# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import os
import xarray as xr
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from typing import List, Any

from L2.L2_Algorithm import L2_Algorithm, L2_output
from L2.TimeSeriesTreatment.steps.step_01_norm_structure import normalize_structure
from L2.TimeSeriesTreatment.steps.step_02_resample import resample_dataset
from L2.TimeSeriesTreatment.steps.step_03_outliers import run_outliers
from L2.TimeSeriesTreatment.steps.step_04_interpolation import run_interpolation
from L2.TimeSeriesTreatment.steps.step_05_feature_engineering import run_feature_engineering
from L2.TimeSeriesTreatment.steps.step_06_smoothing import run_smoothing
from L2.TimeSeriesTreatment.steps.step_07_normalization import run_normalization
from logger import logger


def _deep_merge(dict_base, dict_task):
    for key, value in dict_task.items():
        if key in dict_base and isinstance(dict_base[key], dict) and isinstance(value, dict):
            _deep_merge(dict_base[key], value)
        else:
            dict_base[key] = value
    return dict_base


class TimeSeriesTreatment(L2_Algorithm):
    def __init__(self,
                 sensor_config: str,
                 task_config: str):
        super().__init__()
        self.sensor_config_path = sensor_config
        self.task_config_path = task_config

        self.cfg = None
        self.dataset_processed = None

    def _load_configs(self):
        sensor_path = self.sensor_config_path
        task_path = self.task_config_path

        module_dir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.isabs(sensor_path):
            sensor_path = os.path.normpath(os.path.join(module_dir, sensor_path))
        if not os.path.isabs(task_path):
            task_path = os.path.normpath(os.path.join(module_dir, task_path))

        if not os.path.exists(sensor_path):
            raise FileNotFoundError(f"Sensor config not found: {sensor_path}")
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task config not found: {task_path}")

        with open(sensor_path, "r") as f:
            cfg_sensor = yaml.safe_load(f)
        with open(task_path, "r") as f:
            cfg_task = yaml.safe_load(f)

        self.cfg = _deep_merge(cfg_sensor, cfg_task)

    def _create_debug_image(self, ds: xr.Dataset) -> Image.Image:
        skip_semantics = {"location", "categorical"}
        numeric_vars = [v for v in ds.data_vars 
                        if np.issubdtype(ds[v].dtype, np.number) 
                        and "time" in ds[v].dims
                        and not v.endswith("__was_outlier")
                        and not v.endswith("__is_interpolated")
                        and self.cfg.get("variables", {}).get(v, {}).get("semantic", None) not in skip_semantics]
        
        plot_vars = numeric_vars[:min(4, len(numeric_vars))]
        
        has_sensor_dim = "sensor" in ds.dims
        n_sensors = len(ds["sensor"]) if has_sensor_dim else 1
        cmap = plt.cm.tab10

        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 3 * len(plot_vars)), squeeze=False)
        fig.suptitle(f"TimeSeriesTreatment — {self.cfg.get('dataset_name', 'Dataset')} "
                     f"[{self.cfg.get('task_name', 'Task')}]",
                     fontsize=13, fontweight="bold")

        for i, var in enumerate(plot_vars):
            ax = axes[i, 0]
            da = ds[var]
            if has_sensor_dim:
                for s_idx in range(n_sensors):
                    da_s = da.isel(sensor=s_idx).dropna("time", how="all")
                    sensor_name = str(ds["sensor"].values[s_idx])
                    if len(sensor_name) > 20:
                        sensor_name = sensor_name[:8] + "..." + sensor_name[-8:]
                    ax.plot(da_s["time"].values, da_s.values, linewidth=0.7, marker='.', markersize=2,
                            color=cmap(s_idx % 10), label=sensor_name)
                if n_sensors <= 10:
                    ax.legend(fontsize=6, loc="upper right")
            else:
                da_dropped = da.dropna("time", how="all")
                ax.plot(da_dropped["time"].values, da_dropped.values, linewidth=0.8, marker='.', markersize=2, color="#1f77b4")
            ax.set_title(var, fontsize=10)
            ax.tick_params(labelsize=8)
            fig.autofmt_xdate(rotation=20, ha="right")

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def process_data(self, l1_inputs) -> L2_output:
        logger.info(f"=== TimeSeriesTreatment: STARTING ===")
        
        self._load_configs()
        
        switches = self.cfg.get("pipeline_switches", {})
        logger.info(f"Dataset: {self.cfg.get('dataset_name', 'N/A')} | "
                     f"Task: {self.cfg.get('task_name', 'N/A')} | "
                     f"Switches: {switches}")

        l1_input = l1_inputs[0]
        logger.info("Loading datacube from L1 input")
        ds = l1_input.get_datacube()

        ds = normalize_structure(ds)

        ds = resample_dataset(ds, self.cfg)

        if switches.get("outliers", True):
            ds = run_outliers(ds, self.cfg)
        else:
            logger.info("STEP 03 (Outliers): SKIPPED by task strategy.")

        if switches.get("interpolation", True):
            ds = run_interpolation(ds, self.cfg)
        else:
            logger.info("STEP 04 (Interpolation): SKIPPED by task strategy.")

        if switches.get("features", True):
            ds = run_feature_engineering(ds, self.cfg)
        else:
            logger.info("STEP 05 (Features): SKIPPED by task strategy.")

        if switches.get("smoothing", True):
            ds = run_smoothing(ds, self.cfg)
        else:
            logger.info("STEP 06 (Smoothing): SKIPPED by task strategy.")

        if switches.get("normalization", True):
            ds = run_normalization(ds, self.cfg)
        else:
            logger.info("STEP 07 (Normalization): SKIPPED by task strategy.")

        self.dataset_processed = ds

        l1_input.ts_dataset = ds

        debug_img = self._create_debug_image(ds)

        logger.info(f"=== TimeSeriesTreatment: COMPLETED ({ds.sizes}) ===")

        return L2_output(
            debug_image=debug_img,
            datacube=ds,
            processed_band_info={}
        )

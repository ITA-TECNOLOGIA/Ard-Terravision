# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from L1.L1_Input import L1_Input
import xarray as xr
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from logger import logger


class NumericalData(L1_Input):
    def __init__(self, datacube_path: str):
        logger.info(f"Initializing NumericalData with datacube: {datacube_path}")
        self.datacube_path = datacube_path
        self.datacube = self._load_datacube()
        super().__init__()
        logger.info(f"Loaded datacube with sizes: {self.datacube.sizes}")

    def _load_datacube(self):
        logger.info(f"Opening datacube from {self.datacube_path}")
        ds = xr.open_dataset(self.datacube_path)
        logger.info("Datacube opened successfully")
        return ds

    def get_datacube(self):
        logger.info("Retrieving full numerical datacube")
        return self.datacube

    def get_debug_image(self) -> Image.Image:
        logger.info("Generating debug time series image from NumericalData")

        numeric_vars = [v for v in self.datacube.data_vars
                        if np.issubdtype(self.datacube[v].dtype, np.number)
                        and "time" in self.datacube[v].dims]
        plot_vars = numeric_vars[:min(4, len(numeric_vars))]

        if not plot_vars:
            return Image.new("RGB", (400, 100), (255, 255, 255))

        has_sensor_dim = "sensor" in self.datacube.dims
        n_sensors = len(self.datacube["sensor"]) if has_sensor_dim else 1
        n_colors = min(n_sensors, 10)
        cmap = plt.cm.tab10

        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 3 * len(plot_vars)), squeeze=False)
        fig.suptitle("Numerical Data — Raw Input", fontsize=13, fontweight="bold")

        for i, var in enumerate(plot_vars):
            ax = axes[i, 0]
            da = self.datacube[var]
            if has_sensor_dim:
                for s_idx in range(n_sensors):
                    da_s = da.isel(sensor=s_idx)
                    sensor_name = str(self.datacube["sensor"].values[s_idx])
                    if len(sensor_name) > 20:
                        sensor_name = sensor_name[:8] + "..." + sensor_name[-8:]
                    ax.plot(da_s["time"].values, da_s.values, linewidth=0.7,
                            color=cmap(s_idx % 10), label=sensor_name)
                if n_sensors <= 10:
                    ax.legend(fontsize=6, loc="upper right")
            else:
                ax.plot(da["time"].values, da.values, linewidth=0.8, color="#2ca02c")
            ax.set_title(var, fontsize=10)
            ax.tick_params(labelsize=8)
            fig.autofmt_xdate(rotation=20, ha="right")

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def get_sun_angles(self, time_index: int) -> Tuple[np.ndarray, np.ndarray]:
        logger.warning("get_sun_angles not applicable for NumericalData")
        return (np.zeros((1, 1)), np.zeros((1, 1)))

    def get_view_angles(self, time_index: int) -> Tuple[np.ndarray, np.ndarray]:
        logger.warning("get_view_angles not applicable for NumericalData")
        return (np.zeros((1, 1)), np.zeros((1, 1)))

    def get_dem(self, time_index: int) -> np.ndarray:
        logger.warning("get_dem not applicable for NumericalData")
        return np.zeros((1, 1))

    def get_cloud_mask(self, time_index: int) -> np.ndarray:
        logger.warning("get_cloud_mask not applicable for NumericalData")
        return np.ones((1, 1), dtype=np.uint8)

    def get_ground_truth(self, time_index: int, band_indices: Sequence[str]) -> np.ndarray:
        logger.warning("get_ground_truth not applicable for NumericalData")
        return np.zeros((1, 1, len(band_indices)))

    def update_datacube(self, time_index: int, band_indices: Sequence[str],
                        new_values: np.ndarray) -> None:
        logger.warning("update_datacube by band indices not applicable for NumericalData "
                        "(use direct xarray operations)")

    def get_image(self, time_index: int, band_indices: list[str]):
        logger.warning("get_image not applicable for NumericalData")
        return np.zeros((1, 1, len(band_indices)))

    def get_rgb_image(self, time_index: int):
        logger.warning("get_rgb_image not applicable for NumericalData")
        return np.zeros((224, 224, 3), dtype=np.uint8)

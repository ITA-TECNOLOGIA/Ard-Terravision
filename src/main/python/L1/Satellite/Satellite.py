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
from logger import logger

class Satellite(L1_Input):
    def __init__(self,
                 datacube_path: str):  # must be .nc format
        logger.info(f"Initializing Satellite with datacube: {datacube_path}")
        self.datacube_path = datacube_path
        self.datacube = self._load_datacube()
        self.rbg_band_indices = ["B04", "B03", "B02"]
        self.cloud_band = "SCL"
        self.sun_azimuth_angles_band = "sunAzimuthAngles"
        self.sun_zenith_angles_band = "sunZenithAngles"
        self.view_azimuth_mean_band = "viewAzimuthMean"
        self.view_zenith_mean_band = "viewZenithMean"
        super().__init__()
        logger.info(f"Loaded datacube with dimensions: {self.datacube.sizes}")

    def _get_array(self, bands, time_index, array_name, dim="band"):
        logger.debug(f"Selecting array '{array_name}' for bands={bands}, time_index={time_index}")
        if not isinstance(bands, (list, tuple)):
            bands = [bands]
        selected = self.datacube[bands]
        selected_time = selected.isel(t=time_index)
        arr = selected_time.to_array(name=array_name, dim=dim).values
        logger.debug(f"Retrieved array '{array_name}' shape: {arr.shape}")
        return arr

    def get_datacube(self):
        logger.info("Retrieving full datacube")
        return self.datacube

    def get_debug_image(self):
        logger.info("Generating debug RGB image from sentinel bands")
        time_index = 0 # TODO NOTE THAT THE DEBUG IMAGE IS HARD CODED TO TIME INDEX 0
        rgb_image = self.get_rgb_image(time_index=time_index)
        img = rgb_image.astype(np.float32)
        for c in range(3):
            band = img[..., c]
            lo, hi = band.min(), band.max()
            if hi > lo:
                img[..., c] = (band - lo) / (hi - lo)
            else:
                img[..., c] = 0.0
        img_uint8 = (img * 255.0).round().astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        logger.info("Debug image generated successfully")
        return pil_img

    def get_sun_angles(self, time_index: int):
        logger.info(f"Getting sun angles at time index {time_index}")
        sun_azimuth = self._get_array(self.sun_azimuth_angles_band, time_index, "sun_azimuth_angles")
        sun_zenith = self._get_array(self.sun_zenith_angles_band, time_index, "sun_zenith_angles")
        return sun_azimuth, sun_zenith

    def get_view_angles(self, time_index: int):
        logger.info(f"Getting view angles at time index {time_index}")
        view_azimuth = self._get_array(self.view_azimuth_mean_band, time_index, "view_azimuth")
        view_zenith = self._get_array(self.view_zenith_mean_band, time_index, "view_zenith")
        return view_azimuth, view_zenith

    def get_dem(self, time_index: int):  # Digital Elevation Model (similar to depth map)
        logger.warning("DEM hardcoded; generating random DEM array")
        image_shape = self.get_image(time_index, ["B01"]).squeeze().shape
        random_array = np.random.randint(0, 100, size=image_shape)
        logger.debug(f"Generated DEM array with shape {random_array.shape}")
        return random_array

    def get_cloud_mask(self, time_index: int) -> np.ndarray:
        logger.info(f"Creating cloud mask for time index {time_index}")
        seg = self._get_array(self.cloud_band, time_index, "cloud_segmentation")
        mask = np.where(np.isin(seg, [3, 7, 8, 9]), 0, 1).astype(np.uint8)
        logger.debug(f"Cloud mask generated with shape {mask.shape}")
        return mask

    def get_ground_truth(self, time_index: int, band_indices: list[str]): # TODO NOTE GORUND TRUTH IS HARD CODED!!!
        logger.warning("Ground truth timestamp is hardcoded to index 18")
        rgb = self._get_array(band_indices, 18, "ground_truth_rgb")
        logger.debug(f"Retrieved ground truth array shape {rgb.shape}")
        return rgb

    def get_image(self, time_index: int, band_indices: list[str]):
        logger.debug(f"Fetching image bands {band_indices} at time {time_index}")
        return self._get_array(band_indices, time_index, "custom_image")

    def get_rgb_image(self, time_index: int):
        logger.info(f"Fetching RGB image at time {time_index} for bands {self.rbg_band_indices}")
        img = self._get_array(self.rbg_band_indices, time_index, "rgb_image")
        img = img.transpose(1, 2, 0).copy()
        logger.debug(f"RGB image array shape {img.shape}")
        return img

    def update_datacube(self, time_index: int, band_indices: list[str], new_values: np.ndarray):
        logger.info(f"Updating datacube at time {time_index} for bands {band_indices}")
        for band in band_indices:
            if band not in self.datacube:
                logger.error(f"Band '{band}' not found in datacube")
                raise ValueError(f"Band '{band}' not found in the datacube.")
        sample = self.datacube[band_indices[0]]
        if "t" not in sample.dims:
            logger.error(f"Band '{band_indices[0]}' has no time dimension")
            raise ValueError(f"Band '{band_indices[0]}' does not have a time ('t') dimension.")
        tsize = sample.sizes['t']
        if time_index < 0 or time_index >= tsize:
            logger.error(f"time_index {time_index} out of bounds (0, {tsize-1})")
            raise ValueError(f"time_index {time_index} is out of bounds for time dimension of size {tsize}.")
        spatial = [dim for dim in sample.dims if dim != 't']
        expected_shape = tuple(sample.shape[sample.get_axis_num(d)] for d in spatial)
        if new_values.shape[2] != len(band_indices) or new_values.shape[:2] != expected_shape:
            logger.error(f"new_values shape {new_values.shape} does not match expected {expected_shape} and band count {len(band_indices)}")
            raise ValueError(f"new_values shape {new_values.shape} does not match expected shape {expected_shape} and band count {len(band_indices)}")
        for i, band in enumerate(band_indices):
            data = self.datacube[band].values
            data[time_index, :, :] = new_values[:, :, i]
            self.datacube[band].values = data
        logger.info("Datacube update completed successfully")

    def _normalize_image(self, image):
        logger.debug("Normalizing image to uint8 range")
        min_val = np.nanmin(image)
        max_val = np.nanmax(image)
        if np.isnan(min_val) or np.isnan(max_val) or (max_val - min_val) == 0:
            logger.warning("Image has no variation; returning zeros array")
            return np.zeros_like(image, dtype=np.uint8)
        norm = (image - min_val) / (max_val - min_val)
        arr = (np.nan_to_num(norm) * 255).astype(np.uint8)
        logger.debug("Image normalization complete")
        return arr

    def _otsu_threshold(self, image):
        logger.debug("Computing Otsu threshold")
        flat = image.ravel()
        hist, _ = np.histogram(flat, bins=256, range=(0, 255))
        cum_hist = np.cumsum(hist)
        cum_int = np.cumsum(hist * np.arange(256))
        total = float(flat.size)
        best_t, max_var = 0, 0.0
        for t in range(256):
            w0 = cum_hist[t]
            w1 = total - w0
            if w0 == 0 or w1 == 0:
                continue
            sum0 = cum_int[t]
            sum1 = cum_int[-1] - sum0
            m0, m1 = sum0 / w0, sum1 / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > max_var:
                max_var, best_t = var, t
        logger.debug(f"Otsu threshold determined: {best_t}")
        return best_t

    def _load_datacube(self):
        logger.info(f"Opening datacube from {self.datacube_path}")
        ds = xr.open_dataset(self.datacube_path)
        logger.info("Datacube opened successfully")
        return ds

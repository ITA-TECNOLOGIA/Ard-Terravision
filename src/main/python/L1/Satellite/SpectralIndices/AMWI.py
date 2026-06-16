# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------
import os
from dotenv import load_dotenv
import geopandas as gpd
from logger import logger
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import openeo
from PIL import Image
from shapely.geometry import mapping
import xarray as xr
from typing import List, Optional, Sequence

from L1.L1_Input import L1_Input

load_dotenv()

class AMWI(L1_Input):
    def __init__(self,
                 use_openeo: bool = True,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 shapefile: Optional[str] = None,
                 datacube_path: Optional[str] = None,
                 time_indices: Optional[List[int]] = None,
                 debug_time_index: int = 7
                 ):

        self.spectral_index = self.__class__.__name__
        self.start_date = start_date
        self.end_date = end_date

        super().__init__()

        if use_openeo:
            if start_date is None:
                raise ValueError("start_date is required when using openEO")
            if end_date is None:
                raise ValueError("end_date is required when using openEO")

            logger.info(f"Downloading {self.spectral_index} using openEO for {shapefile} between {start_date} and {end_date}")
            if shapefile is None:
                raise ValueError("shapefile is required when using openEO")

            self.datacube = self._download_datacube(shapefile, start_date, end_date)
            self.datacube_path = None
        else:
            logger.info(f"Reading Spectral Index datacube from: {datacube_path}")
            self.datacube_path = datacube_path
            self.datacube = self._read_datacube(start_date=start_date, end_date=end_date,
                                                time_indices=time_indices)

        self.time_indices = self._resolve_time_indices(time_indices)
        self.debug_time_index = debug_time_index

        logger.info(f"Time indices: {self.time_indices}, debug_time_index: {self.debug_time_index}")
        logger.info(f"Loaded datacube with dimensions: {self.datacube.sizes}")

    def _download_workflow_spectral_index(self, connection, filename, shape, start_date, end_date):
        
        job_options = {
            "executor-memory": "12G",
            "executor-memoryOverhead": "4G",
            "executor-cores": "1",
            "driver-memory": "8G",
            "driver-memoryOverhead": "2G",
            "max-executors": "50"
        }

        bands = ["B02", "B04"]
        # Define datacube
        datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=shape,
        temporal_extent=[start_date, end_date],
        bands=bands
        )
        # Compute AMWI
        amwi = (datacube.band("B04") - datacube.band("B02")) / (datacube.band("B04") + datacube.band("B02"))

        job = amwi.execute_batch(
            filename,
            title=f"Download field {self.spectral_index} data of {shape} for the period {start_date} to {end_date}",
            job_options=job_options,
            out_format="netcdf"
        )

    def _download_datacube(self, shapefile_name, start_date, end_date):
        # Get paths from .env
        OPENEO_DOWNLOADS_PATH = os.getenv("OPENEO_DOWNLOADS_PATH")
        OPENEO_CLIENT_ID = os.getenv("OPENEO_CLIENT_ID")
        OPENEO_CLIENT_SECRET = os.getenv("OPENEO_CLIENT_SECRET")

        SHAPEFILES_PATH = os.getenv("SHAPEFILES_PATH")

        # Read the shapefile
        shape = gpd.read_file(os.path.join(SHAPEFILES_PATH, shapefile_name))
        shape = shape['geometry'][0]
        
        # Define output path
        output_dir = OPENEO_DOWNLOADS_PATH
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Define the filename
        shapefile_name = shapefile_name.replace(".shp", "")
        filename = os.path.join(output_dir, f"s2_l2a_{self.spectral_index}_{shapefile_name}_{start_date}_{end_date}.nc")

        # define the connection
        connection = openeo.connect("openeofed.dataspace.copernicus.eu")
        # connection = openeo.connect("openeo.dataspace.copernicus.eu")
        connection.authenticate_oidc_client_credentials(
                            client_id=OPENEO_CLIENT_ID,
                            client_secret=OPENEO_CLIENT_SECRET
                        )

        self._download_workflow_spectral_index(connection, filename, shape, start_date, end_date)
        logger.info(f"Downloaded {self.spectral_index} datacube to {filename}")

        # Read the datacube
        ds = xr.open_dataset(filename)
        ds = ds.rename_vars({"var": self.spectral_index})
        return ds

    def _resolve_time_indices(self, time_indices: Optional[List[int]]) -> List[int]:
        if time_indices is not None and len(time_indices) > 0:
            return list(time_indices)
        n_times = self.datacube.sizes.get('t', 0)
        return list(range(n_times))

    def _read_datacube(self, start_date, end_date, time_indices=None):
        logger.info(f"Opening datacube from {self.datacube_path}")
        ds_full = xr.open_dataset(self.datacube_path)
        logger.info("Datacube opened successfully")

        if time_indices is not None and len(time_indices) > 0:
            logger.info(f"Using provided time indices: {time_indices}")
            resolved_indices = time_indices
        elif start_date is not None and end_date is not None:
            logger.info(f"Using time indices between {start_date} and {end_date}")
            full_time = ds_full['t'].values
            start_index = np.where(full_time >= np.datetime64(start_date, 'ns'))[0][0]
            end_index = np.where(full_time <= np.datetime64(end_date, 'ns'))[0][-1]
            resolved_indices = list(range(start_index, end_index + 1))
        else:
            logger.info("No dates provided. Using all time indices from input.")
            resolved_indices = list(range(ds_full.sizes['t']))

        datacube_subset = ds_full.isel(t=resolved_indices)
        ds = (datacube_subset['B04'] - datacube_subset['B02']) / (datacube_subset['B04'] + datacube_subset['B02']).to_dataset(name="AMWI")

        print(f"Processed {self.spectral_index} for time index {resolved_indices}")
        return ds

    def _get_array(self, time_index, array_name, dim="band"):
        logger.debug(f"Selecting array '{array_name}' for time_index={time_index}")
        selected_time = self.datacube.isel(t=time_index)
        arr = selected_time.to_array(name=array_name, dim=dim).values
        logger.debug(f"Retrieved array '{array_name}' shape: {arr.shape}")
        return arr

    def get_datacube(self):
        logger.info("Retrieving full datacube")
        return self.datacube
    
    def get_datacube_subset(self, time_indices: list[int]):
        logger.debug(f"Subsetting datacube for time_indices={time_indices}")
        # Select coordinates (time)
        if time_indices is not None:
            datacube_subset = self.datacube.isel(t=time_indices)
        logger.debug(f"Retrieved datacube subset shape: {datacube_subset.coords}")
        return datacube_subset

    def get_debug_image(self):
        logger.info(f"Generating debug image for {self.spectral_index}")
        time_index = self.debug_time_index
        amwi_debug = self.datacube[self.spectral_index].isel(t=time_index).values

        # Configure the color palette
        norm = colors.Normalize(vmin=-1, vmax=1)
        rgba = cm.get_cmap("YlGnBu_r")(norm(amwi_debug))
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        amwi_debug_img = Image.fromarray(rgb, mode="RGB")
        logger.info("Debug image generated successfully")

        return amwi_debug_img

    def get_rgb_image(self, time_index: int):
        logger.info(f"Fetching RGB image at time {time_index} for bands {self.spectral_index}")
        img = self._get_array(time_index=time_index, array_name="rgb_image")
        img = img.transpose(1, 2, 0).copy()
        logger.debug(f"RGB image array shape {img.shape}")
        return img


    def get_sun_angles(self, time_index: int):
        raise NotImplementedError(f"get_sun_angles not implemented for {self.spectral_index}")

    def get_view_angles(self, time_index: int):
        raise NotImplementedError(f"get_view_angles not implemented for {self.spectral_index}")

    def get_dem(self, time_index: int):  # Digital Elevation Model (similar to depth map)
        raise NotImplementedError(f"get_dem not implemented for {self.spectral_index}")

    def get_cloud_mask(self, time_index: int) -> np.ndarray:
        raise NotImplementedError(f"get_cloud_mask not implemented for {self.spectral_index}")

    def get_ground_truth(self, time_index: int, band_indices: list[str]): # TODO NOTE GORUND TRUTH IS HARD CODED!!!
        raise NotImplementedError(f"get_ground_truth not implemented for {self.spectral_index}")

    def update_datacube(self, time_index: int, band_indices: Sequence[str], new_values: np.ndarray) -> None:
        raise NotImplementedError("update_datacube is deprecated and will be removed in a future version")

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


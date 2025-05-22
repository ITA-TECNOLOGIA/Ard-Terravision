# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from L2.L2_Algorithm import L2_Algorithm, L2_result
from typing import List
from logger import logger
import os
from osgeo import gdal, ogr
import json
import shapely
import re
import numpy as np
from PIL import Image
import subprocess

from L2.Orthorectification.utils.ortho_tools import make_ortho, unpack_rpc_parameters
from L2.Orthorectification.utils.scaling_tools import gaussian_rescale
from L2.Orthorectification.utils.io_tools import save_raster_as_geotiff

class Orthorectification(L2_Algorithm):
    def __init__(self,
                 time_indices: list[int],
                 band_names: list[int]): 
        self.time_indices = time_indices
        self.band_names = band_names

    def process_data(self, input):
        results: List[L2_result] = []

        for time_index in self.time_indices:
            dem = input.get_dem(time_index)
            for band_name in self.band_names:
                print(f"Processing orthorectification correction for time index {time_index} and band {band_name}")
                logger.warning(f"THIS IS JUST AN EXAMPLE WITH THE ORIGINAL DATA, NOT OUR DATA, MUST BE INTEGRATED WITH THE REAL DATA")

                # Paths
                shapefile_path = "/home/anavarroa/pro24_0061_terravision_he/Pipeline/tasks/ortho/images/3V050905M0000880351A520001100172M_001659557/3v050905m0000880351a520001100172m_001659557.shp"
                image_path = "/home/anavarroa/pro24_0061_terravision_he/Pipeline/tasks/ortho/images/3V050905M0000880351A520001100172M_001659557/3v050905m0000880351a520001100172m_001659557.tif"
                output_dir = "/home/anavarroa/pro24_0061_terravision_he/Pipeline/tasks/ortho/outputs"
                output_dem_path = os.path.join(output_dir, 'elevation.dem')
                pre_ortho_output_path = os.path.join(output_dir, "pre_ortho_scene.tif")
                output_ortho_path = os.path.join("./ortho_scene.tif")

                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)

                # Open and parse shapefile
                logger.debug("Opening shapefile to read image footprint...")
                file = ogr.Open(shapefile_path)
                if file is None:
                    raise RuntimeError("OGR could not open the shapefile. Check file path and dependencies.")
                shape = file.GetLayer(0)
                feature = shape.GetFeature(0)
                image_footprint = json.loads(feature.ExportToJson())
                image_footprint = shapely.geometry.Polygon(image_footprint["geometry"]["coordinates"][0])
                logger.debug(f"Image Footprint: {image_footprint}")

                # Extract lat/lon bounds
                lons = image_footprint.exterior.coords.xy[0]
                lats = image_footprint.exterior.coords.xy[1]

                # Load GDAL image and RPC
                logger.debug("Loading dataset and reading metadata...")
                gdal.AllRegister()
                orbview_dataset = gdal.Open(image_path)
                rpc_dict = orbview_dataset.GetMetadata_Dict("RPC")
                rpc_dict_repaired = {key: re.sub("[^0-9\-.\sE]", "", value) for key, value in rpc_dict.items()}
                rpc_dict_repaired["MIN_LONG"] = min(lons)
                rpc_dict_repaired["MAX_LONG"] = max(lons)
                rpc_dict_repaired["MIN_LAT"] = min(lats)
                rpc_dict_repaired["MAX_LAT"] = max(lats)

                logger.debug("RPC Dictionary after cleaning:")
                logger.debug(rpc_dict_repaired)

                # Unpack RPC
                logger.debug("Unpacking RPC parameters...")
                rpcs = unpack_rpc_parameters(rpc_dict_repaired)
                logger.debug(f"Unpacked RPC Parameters: {rpcs}")

                # Read and rescale image
                logger.debug("Reading image data...")
                pixel_data = orbview_dataset.ReadAsArray()
                logger.debug(f"Maximum pixel value in the image: {pixel_data.max()}")
                logger.debug("Rescaling image data...")
                single_channel = pixel_data.mean(axis=0)
                rescaled_img = gaussian_rescale(single_channel, bitness=11).astype(np.uint8)

                # Generate DEM
                logger.debug("Generating DEM using subprocess...")
                #subprocess.run(['createdem', '11', '49', '2', '2', '--data-source', 'NASA', '--output', output_dem_path], check=True)
                logger.debug(f"Generated DEM saved at: {output_dem_path}")

                # Load DEM
                logger.debug("Opening the generated DEM file...")
                dem_ds = gdal.Open(output_dem_path)
                dem_geo_t = dem_ds.GetGeoTransform()
                raw_dem_data = dem_ds.ReadAsArray()

                # Save pre-orthorectified image
                logger.debug("Saving pre-orthorectified image...")
                estimated_gsd = 0.5
                ul_lon_est = rpc_dict_repaired["MIN_LONG"]
                ul_lat_est = rpc_dict_repaired["MAX_LAT"]
                save_raster_as_geotiff(rescaled_img, ul_lon_est, ul_lat_est, estimated_gsd, pre_ortho_output_path)
                logger.debug(f"Pre-orthorectified image saved at: {pre_ortho_output_path}")


                # Orthorectification
                logger.debug("Creating orthorectified image...")
                img_data_warped, gsd, ul_lon, ul_lat = make_ortho(
                    min(lons), max(lons), min(lats), max(lats),
                    5000, rescaled_img, rpcs, raw_dem_data, dem_geo_t
                )

                # Convert to scalars if needed
                gsd = gsd.item() if isinstance(gsd, np.ndarray) else gsd
                ul_lon = ul_lon.item() if isinstance(ul_lon, np.ndarray) else ul_lon
                ul_lat = ul_lat.item() if isinstance(ul_lat, np.ndarray) else ul_lat

                # Rotate/flip and save final ortho image
                logger.debug("Rotating and flipping orthorectified image to match pre-ortho orientation...")
                img_data_corrected = np.fliplr(np.rot90(img_data_warped, 2))
                save_raster_as_geotiff(img_data_corrected.astype(np.uint8), ul_lon, ul_lat, gsd, output_ortho_path)
                logger.debug(f"Corrected orthorectified image saved at: {output_ortho_path}")

                # Read the orthorectified image
                logger.debug("Reading the orthorectified image and convert to numpy...")
                ortho_img = Image.open(output_ortho_path)
                ortho_img = np.array(ortho_img)

                results.append(L2_result(
                    debug_image=ortho_img,
                    algorithm_results=None
                ))

        return results
# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import List, Optional
import xarray as xr

from L3.L3_Algorithm import L3_Algorithm, L3_result
from logger import logger

class NDVI(L3_Algorithm):
    def __init__(self,
                 time_indices: list[int]):
        self.time_indices = time_indices
        self.bands = ['B04', 'B08']

    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        print(f"Processing NDVI for time index {self.time_indices}")
        data_source = l2_datacube if l2_datacube is not None else input

        if not self.time_indices:
            logger.info("No time indices provided. Using all time indices from input.")
            self.time_indices = np.arange(data_source.datacube.sizes['t'])

        datacube_subset = data_source.get_datacube_subset(bands=self.bands, time_indices=self.time_indices)
        ndvi = (datacube_subset['B08'] - datacube_subset['B04']) / (datacube_subset['B08'] + datacube_subset['B04'])
        ndvi.name = self.__class__.__name__
        ndvi_debug = ndvi.isel(t=0).values
        norm = colors.Normalize(vmin=-1, vmax=1)
        rgba = cm.get_cmap("RdYlGn")(norm(ndvi_debug))
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        ndvi_debug_img = Image.fromarray(rgb, mode="RGB")
        result = [L3_result(
            debug_image=ndvi_debug_img,
            algorithm_results=ndvi,
            time_indices=list(self.time_indices),
            result_type="datacube"
        )]
        return result

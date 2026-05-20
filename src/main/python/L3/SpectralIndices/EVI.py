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

class EVI(L3_Algorithm):
    def __init__(self,
                 time_indices: list[int]): 
        self.time_indices = time_indices
        self.bands = ['B02', 'B04', 'B08']

    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        data_source = l2_datacube if l2_datacube is not None else input
        print(f"Processing EVI for time index {self.time_indices}")
        if not self.time_indices: # Empty array
            logger.info("No time indices provided. Using all time indices from input.")
            self.time_indices = np.arange(data_source.datacube.sizes['t']) # Use all time indices from input.

        datacube_subset = data_source.get_datacube_subset(bands=self.bands, time_indices=self.time_indices)
        evi = 2.5 * (datacube_subset['B08'] - datacube_subset['B04']) / (datacube_subset['B08'] + 6*datacube_subset['B04'] - 7.5*datacube_subset['B02'] + 1)
        evi.name = self.__class__.__name__
        evi_debug = evi.isel(t=0).values
        # Configure the color palette
        norm = colors.Normalize(vmin=-1, vmax=1)
        rgba = cm.get_cmap("RdYlGn")(norm(evi_debug))
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        evi_debug_img = Image.fromarray(rgb, mode="RGB")
        result = [L3_result(debug_image=evi_debug_img, algorithm_results=evi, time_indices=list(self.time_indices), result_type="datacube")]
        return result

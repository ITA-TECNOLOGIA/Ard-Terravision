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

from L3.L3_Algorithm import L3_Algorithm, L3_result
from logger import logger

class BSI(L3_Algorithm):
    def __init__(self,
                 time_indices: list[int]): 
        self.time_indices = time_indices
        self.bands = ['B02', 'B04', 'B08', 'B11']

    def process_data(self, input):
        print(f"Processing BSI for time index {self.time_indices}")
        if self.time_indices is not None: # Empty array
            logger.info("No time indices provided. Using all time indices from input.")
            self.time_indices = np.arange(input.datacube.sizes['t']) # Use all time indices from input.

        datacube_subset = input.get_datacube_subset(bands=self.bands, time_indices=self.time_indices)
        b02 = datacube_subset['B02']
        b04 = datacube_subset['B04']
        b08 = datacube_subset['B08']
        b11 = datacube_subset['B11']
        bsi = ((b11 + b04) - (b02 + b08)) / ((b11 + b04) + (b02 + b08))
        bsi.name = self.__class__.__name__
        bsi_debug = bsi.isel(t=53).values
        # Configure the color palette
        norm = colors.Normalize(vmin=-1, vmax=1)
        rgba = cm.get_cmap("YlOrBr")(norm(bsi_debug))
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        bsi_debug_img = Image.fromarray(rgb, mode="RGB")
        result = [L3_result(debug_image=bsi_debug_img, algorithm_results=bsi)]
        return result

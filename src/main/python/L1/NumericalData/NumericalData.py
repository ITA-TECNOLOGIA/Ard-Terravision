# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from L1.L1_Input import L1_Input
import numpy as np

class NumericalData(L1_Input):
    def __init__(self):
        super().__init__()

    def get_datacube(self):
        raise NotImplementedError("get_datacube not implemented for NumericalData")

    def get_sun_angles(self, time_index: int):
        raise NotImplementedError("get_sun_angles not implemented for NumericalData")

    def get_view_angles(self, time_index: int):
        raise NotImplementedError("get_view_angles not implemented for NumericalData")

    def get_DEM(self, time_index: int): # Digital Elevation Model (similar to depth map)
        raise NotImplementedError("get_DEM not implemented for NumericalData")

    def get_cloud_mask(self, time_index: int):
        raise NotImplementedError("get_cloud_mask not implemented for NumericalData")

    def get_ground_truth(self, time_index: int, band_indices: list[str]):
        raise NotImplementedError("get_ground_truth not implemented for NumericalData")

    def get_image(self, time_index: int, band_indices: list[str]):
        return self._get_array(band_indices, time_index, "custom image")

    def update_datacube(self, time_index: int, band_indices: list[str], new_values: np.ndarray):
        raise NotImplementedError("update_datacube not implemented for NumericalData")


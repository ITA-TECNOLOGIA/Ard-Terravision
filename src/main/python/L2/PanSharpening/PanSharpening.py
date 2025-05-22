# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from L2.L2_Algorithm import L2_Algorithm

class PanSharpening(L2_Algorithm):
    def __init__(self,
                 time_indices: list[int],
                 band_names: list[int]): 
        self.time_indices = time_indices
        self.band_names = band_names

    def process_data(self, input):
        for time_index in self.time_indices:
            for band_name in self.band_names:
                print(f"Processing pan sharpening for time index {time_index} and band {band_name}")
                # TODO Update datacube input

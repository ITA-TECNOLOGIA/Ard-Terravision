# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from L3.L3_Algorithm import L3_Algorithm, L3_result

class ChangeDetection(L3_Algorithm):
    def __init__(self,
                 time_indices: list[int],
                 band_names: list[int]): 
        self.time_indices = time_indices
        self.band_names = band_names

    def process_data(self, input):
        for time_index in self.time_indices:
            for band_name in self.band_names:
                print(f"Processing change detection for time index {time_index} and band {band_name}")
                # TODO Update datacube input??? Return info???? 
                return [L3_result(debug_image=None, algorithm_results=None)]
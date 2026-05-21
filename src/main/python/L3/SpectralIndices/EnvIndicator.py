# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

from L3.L3_Algorithm import L3_Algorithm, L3_result
from L3.SpectralIndices.AMWI import AMWI
from L3.SpectralIndices.BSI import BSI
from L3.SpectralIndices.EVI import EVI
from L3.SpectralIndices.NDCI import NDCI
from L3.SpectralIndices.NDDI import NDDI
from L3.SpectralIndices.NDTI import NDTI
from L3.SpectralIndices.NDVI import NDVI
from L3.SpectralIndices.NDWI import NDWI

from logger import logger

class EnvIndicator(L3_Algorithm):
    def __init__(self,
                 time_indices: list[int]): 
        self.time_indices = time_indices
        self.bands = None
        self.spectral_indices = [AMWI, BSI, EVI, NDCI, NDDI, NDTI, NDVI, NDWI]
        # self.spectral_indices = [NDVI, NDWI]

    def _pixelwise_pca(self, datacube_list: list[xr.DataArray]):
        # Concatenate into a single datacube
        # Alignment
        spec_list_aligned = xr.align(*datacube_list, join="exact") #join="inner" if slight missmatch
        # Create feature dimenison
        feature_names = [da.name for i, da in enumerate(spec_list_aligned)] #this should be equal to self.spectral_indices
        X = xr.concat(spec_list_aligned, dim="feature")
        X = X.assign_coords(feature=feature_names)
        X = X.transpose("t", "feature", "y", "x")

        # For each pixel, we have the time and feature values. Only features are used for PCA
        X_pix = X.stack(pix=("y", "x")).transpose("t", "pix", "feature")

        out_maps = []
        evr_list = []

        # Loop through the days
        for tval in X_pix["t"].values:
            a = 1.0
            b = 0.0

            A = X_pix.sel(t=tval).values  # (pix, feature)

            # Mask invalid pixels (NaNs etc.)
            valid_px = np.isfinite(A).all(axis=1)
            A_valid = A[valid_px]

            # Standardize features across pixels for each day
            A_valid = StandardScaler().fit_transform(A_valid)

            pca = PCA(n_components=min(2, A_valid.shape[1]))
            Z = pca.fit_transform(A_valid)  # (pix_valid, pc)

            # combine PC score maps
            eigenvalues = pca.explained_variance_
            rsei = eigenvalues[0] * Z[:, 0] + (eigenvalues[1] * Z[:, 1] if Z.shape[1] > 1 else 0.0)
            # Normalize the indicator
            mrsei = (rsei - np.nanmin(rsei)) / (np.nanmax(rsei) - np.nanmin(rsei))

            env_indicator = np.full((X_pix.sizes["pix"],), np.nan, dtype=np.float32)
            env_indicator[valid_px] = mrsei.astype(np.float32)

            env_indicator_da = (
                xr.DataArray(env_indicator, coords={"pix": X_pix["pix"]}, dims=("pix",))
                .unstack("pix")
                .rename("environmental_indicator")
            )
            out_maps.append(env_indicator_da)

            evr_list.append(
                xr.DataArray(pca.explained_variance_ratio_, dims=("pc",)).assign_coords(pc=np.arange(Z.shape[1]))
            )

        env_datacube = xr.concat(out_maps, dim="t").assign_coords(t=X_pix["t"])
        evr = xr.concat(evr_list, dim="t").assign_coords(t=X_pix["t"]).rename("explained_variance_ratio")

        return env_datacube, evr


    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        print(f"Processing Environmental Indicator for time index {self.time_indices}")
        data_source = l2_datacube if l2_datacube is not None else input

        if not self.time_indices:
            logger.info("No time indices provided. Using all time indices from input.")
            self.time_indices = np.arange(data_source.datacube.sizes['t'])

        spec_list = []
        for spectral_index in self.spectral_indices:
            spec = spectral_index(time_indices=self.time_indices).process_data(data_source)
            spec_list.append(spec[0].algorithm_results)

        env_datacube, evr = self._pixelwise_pca(spec_list)

        env_indicator_debug = env_datacube.isel(t=0).values
        norm = colors.Normalize(vmin=0, vmax=1)
        rgba = cm.get_cmap("RdYlGn")(norm(env_indicator_debug))
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        env_indicator_debug_img = Image.fromarray(rgb, mode="RGB")

        result = [L3_result(
            debug_image=env_indicator_debug_img,
            algorithm_results=env_datacube,
            time_indices=list(self.time_indices),
            result_type="datacube"
        )]

        return result

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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

from L3.L3_Algorithm import L3_Algorithm, L3_result
from logger import logger


class EnvIndicator(L3_Algorithm):
    def __init__(self):
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1

    def _pixelwise_pca(self, fused_datacube: xr.Dataset):
        feature_names = list(fused_datacube.data_vars)
        X = xr.concat([fused_datacube[var] for var in feature_names], dim="feature")
        X = X.assign_coords(feature=feature_names)
        X = X.transpose("t", "feature", "y", "x")

        X_pix = X.stack(pix=("y", "x")).transpose("t", "pix", "feature")

        out_maps = []
        evr_list = []

        for tval in X_pix["t"].values:
            A = X_pix.sel(t=tval).values

            valid_px = np.isfinite(A).all(axis=1)
            A_valid = A[valid_px]

            A_valid = StandardScaler().fit_transform(A_valid)

            pca = PCA(n_components=min(2, A_valid.shape[1]))
            Z = pca.fit_transform(A_valid)

            eigenvalues = pca.explained_variance_
            rsei = eigenvalues[0] * Z[:, 0] + (eigenvalues[1] * Z[:, 1] if Z.shape[1] > 1 else 0.0)
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
        if l2_datacube is None:
            raise ValueError("EnvIndicator requires L2 fused datacube from SpectralIndexFusion. "
                             "Please add SpectralIndexFusion to L2 algorithms.")

        print(f"Processing Environmental Indicator for time index {self.time_indices}")

        if not self.time_indices:
            logger.info("No time indices provided. Using all time indices from input.")
            self.time_indices = list(range(l2_datacube.sizes['t']))

        if self.time_indices:
            fused_subset = l2_datacube.isel(t=self.time_indices)
        else:
            fused_subset = l2_datacube

        env_datacube, evr = self._pixelwise_pca(fused_subset)

        env_indicator_debug = env_datacube.isel(t=0).values
        norm = colors.Normalize(vmin=0, vmax=1)
        rgba = plt.get_cmap("RdYlGn")(norm(env_indicator_debug))
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        env_indicator_debug_img = Image.fromarray(rgb, mode="RGB")

        result = [L3_result(
            debug_image=env_indicator_debug_img,
            algorithm_results=env_datacube,
            time_indices=list(self.time_indices),
            result_type="datacube"
        )]

        return result
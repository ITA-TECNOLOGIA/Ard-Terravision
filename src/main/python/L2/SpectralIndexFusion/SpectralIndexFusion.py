# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

import numpy as np
import xarray as xr
from typing import Any, List, Optional
from PIL import Image

from L2.L2_Algorithm import L2_Algorithm, L2_output
from logger import logger


class SpectralIndexFusion(L2_Algorithm):
    def __init__(self):
        self._fused_datacube = None

    def _resolve_common_time_indices(self, l1_inputs: List[Any]) -> Optional[List[int]]:
        all_time_indices = []
        for inp in l1_inputs:
            ti = getattr(inp, 'time_indices', None)
            all_time_indices.append(set(ti) if ti else set())

        non_empty = [ti for ti in all_time_indices if ti]
        if not non_empty:
            return None

        if len(non_empty) == 1 and non_empty[0]:
            return sorted(non_empty[0])
        elif len(non_empty) > 1:
            if any(not ti for ti in all_time_indices):
                raise ValueError(
                    "Inconsistent time_indices across L1 inputs. "
                    "Some have time_indices, some don't."
                )

            common = non_empty[0]
            for ti in non_empty[1:]:
                common &= ti

            if not common:
                raise ValueError(
                    f"No overlapping time indices across L1 inputs. "
                    f"Got: {[sorted(ti) for ti in all_time_indices]}"
                )

            if any(ti != common for ti in non_empty):
                logger.warning(
                    f"L1 inputs have different time_indices. Using intersection: "
                    f"{sorted(common)}. Original: {[sorted(ti) for ti in all_time_indices]}"
                )

            return sorted(common)

        return None

    def process_data(self, l1_inputs: List[Any]) -> Optional[L2_output]:
        if not l1_inputs:
            raise ValueError("No L1 inputs provided for fusion")

        logger.info(f"Fusing {len(l1_inputs)} L1 inputs")

        common_time_indices = self._resolve_common_time_indices(l1_inputs)
        if common_time_indices is not None:
            logger.info(f"Using common time indices: {common_time_indices}")

        datacubes = []
        for l1_instance in l1_inputs:
            datacube = l1_instance.get_datacube()
            if common_time_indices is not None and len(common_time_indices) > 0:
                datacube = datacube.isel(t=common_time_indices)
            datacubes.append(datacube)

        aligned_datacubes = xr.align(*datacubes, join="exact")
        fused_dataset = xr.merge(aligned_datacubes, compat="equals")

        self._fused_datacube = fused_dataset

        debug_image = self._generate_debug_image()

        processed_band_info = {
            "spectral_indices": list(fused_dataset.data_vars),
            "num_indices": len(fused_dataset.data_vars),
            "fused_shape": dict(fused_dataset.dims)
        }

        logger.info(f"Spectral index fusion completed. Fused dataset shape: {fused_dataset.dims}")

        return L2_output(
            datacube=fused_dataset,
            debug_image=debug_image,
            processed_band_info=processed_band_info
        )

    def _generate_debug_image(self) -> Image.Image:
        if self._fused_datacube is None:
            return Image.new("RGB", (256, 256), (128, 128, 128))

        first_var = list(self._fused_datacube.data_vars)[0]
        data = self._fused_datacube[first_var]

        if "t" in data.dims and data.sizes["t"] > 0:
            data = data.isel(t=0)

        arr = data.values

        if arr.ndim == 2:
            arr = np.nan_to_num(arr, nan=0.0)
            vmin, vmax = np.min(arr), np.max(arr)
            if vmax - vmin > 0:
                arr = (arr - vmin) / (vmax - vmin)
            else:
                arr = np.zeros_like(arr)

            arr = (arr * 255).astype(np.uint8)
            arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr, mode="RGB")

        return Image.new("RGB", (256, 256), (128, 128, 128))

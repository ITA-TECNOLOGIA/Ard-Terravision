# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from __future__ import annotations

import os
import numpy as np
import torch
from typing import List, Optional
import xarray as xr
from dotenv import load_dotenv

from L3.L3_Algorithm import L3_Algorithm, L3_result
from L3.LulcClassification.lulc_core import (
    DEFAULT_BAND_NAMES,
    build_model,
    load_checkpoint,
    infer_multiband_frame,
    create_visualization_pil,
)
from utils.coords_utils import extract_spatial_coords

load_dotenv()

DEVICE = os.getenv("DEVICE", "cuda:0")

def get_multiband_frame_from_input(input_obj, time_index: int, band_names: list[str], fallback_input=None) -> np.ndarray:
    """
    Uses the same style as ObjectDetectionDetrex:
      - input.get_image(time_index, band_name) for each band
    Also supports:
      - input.get_image(time_index, band_names) returning a multiband array
      - input.get_multiband(time_index, band_names)
    Returns (B,H,W) float32.
    """
    # If input_obj is an xarray Dataset (from L2), try direct access
    if isinstance(input_obj, xr.Dataset):
        try:
            arrays = [input_obj[b].isel(t=time_index).values for b in band_names]
            return np.stack(arrays, axis=0).astype(np.float32)
        except KeyError:
            # Fall back to original input if time_index not found in L2 datacube
            if fallback_input is not None:
                input_obj = fallback_input
            else:
                raise

    # If L1 supports a direct multiband call
    if hasattr(input_obj, "get_multiband") and callable(getattr(input_obj, "get_multiband")):
        arr = np.asarray(input_obj.get_multiband(time_index, band_names))
        if arr.ndim == 3 and arr.shape[0] == len(band_names):
            return arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == len(band_names):
            return arr.transpose(2, 0, 1).astype(np.float32)
        raise ValueError(f"get_multiband returned unexpected shape {arr.shape}")

    # If get_image accepts a list of bands
    try:
        arr = np.asarray(input_obj.get_image(time_index, band_names))
        if arr.ndim == 3 and arr.shape[0] == len(band_names):
            return np.nan_to_num(arr, nan=0.0).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == len(band_names):
            arr = arr.transpose(2, 0, 1)
            return np.nan_to_num(arr, nan=0.0).astype(np.float32)
        # If it didn't throw but returned weird, fall through to per-band
    except Exception:
        pass

    # Default: stack single-band get_image calls
    bands = []
    for b in band_names:
        frame = np.asarray(input_obj.get_image(time_index, b))

        # normalize to (H,W)
        if frame.ndim == 3 and frame.shape[0] == 1:
            frame = frame[0]
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        elif frame.ndim != 2:
            raise ValueError(f"get_image(time_index={time_index}, band={b}) returned shape {frame.shape}, expected (H,W)")

        frame = np.nan_to_num(frame, nan=0.0).astype(np.float32)
        bands.append(frame)

    return np.stack(bands, axis=0)  # (B,H,W)


class LulcClassification(L3_Algorithm):
    def __init__(
        self,
        model_path: str = os.getenv("LUCL_MODEL"),
        net: str = "segformer",
        band_names: Optional[list[str]] = None,
        return_debug_image: bool = True,
    ):
        super().__init__()
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1
        self.net = net
        self.band_names = band_names or list(DEFAULT_BAND_NAMES)
        self.return_debug_image = return_debug_image

        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.model, _ = build_model(self.net, device=self.device)
        self.model = load_checkpoint(self.model, model_path, self.device)

    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        masks_2d: List[xr.DataArray] = []

        data_source = l2_datacube if l2_datacube is not None else input
        coord_source = data_source

        first_debug_img = None

        for time_index in self.time_indices:
            print(f"Processing LULC classification for time index {time_index} and bands {self.band_names}")

            img_3d = get_multiband_frame_from_input(data_source, time_index, self.band_names, fallback_input=input)
            mask = infer_multiband_frame(self.model, self.device, img_3d)

            if first_debug_img is None and self.return_debug_image:
                rgb_vis = np.stack([img_3d[2], img_3d[1], img_3d[0]], axis=-1) / 10000.0
                first_debug_img = create_visualization_pil(rgb_vis, mask, time_idx=time_index, net=self.net)

            spatial_coords = extract_spatial_coords(coord_source, mask.shape[0], mask.shape[1])
            mask_da = xr.DataArray(
                mask,
                dims=("y", "x"),
                coords={"y": spatial_coords["y"], "x": spatial_coords["x"]},
                attrs={
                    "result_type": "mask",
                    "time_index": time_index,
                    "band_names": self.band_names,
                    "net": self.net,
                },
            )

            masks_2d.append(mask_da)

        lulc_datacube = xr.concat(masks_2d, dim="t")
        lulc_datacube = lulc_datacube.assign_coords(t=self.time_indices)

        return [
            L3_result(
                debug_image=first_debug_img,
                algorithm_results=lulc_datacube,
                time_indices=list(self.time_indices),
                result_type="datacube",
            )
        ]

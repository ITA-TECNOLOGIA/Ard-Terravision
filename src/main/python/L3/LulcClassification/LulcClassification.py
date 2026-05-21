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
from typing import List, Optional, Dict, Any
from pathlib import Path
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

load_dotenv()

DEVICE = os.getenv("DEVICE", "cuda:0")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

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
            arrays = [input_obj[b].sel(t=time_index).values for b in band_names]
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
        time_indices: list[int],
        model_path: str = os.getenv("LUCL_MODEL"),
        net: str = "segformer",
        band_names: Optional[list[str]] = None,
        return_debug_image: bool = True,
        write_mask_netcdf: bool = False,
        mask_labels: Optional[list[str]] = None,   # e.g. ["lulc"]
        var_name: str = "mask_data",               # avoid var="mask" when dim="mask"
        compress: bool = True,
    ):
        super().__init__()
        self.time_indices = time_indices
        self.net = net
        self.band_names = band_names or list(DEFAULT_BAND_NAMES)
        self.return_debug_image = return_debug_image

        # output settings
        self.write_mask_netcdf = write_mask_netcdf
        self.out_nc_path = OUTPUT_DIR
        self.mask_labels = mask_labels or ["lulc"]
        self.var_name = var_name
        self.compress = compress

        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.model, _ = build_model(self.net, device=self.device)
        self.model = load_checkpoint(self.model, model_path, self.device)

    def _coords_from_input(self, input_obj, time_indices: list[int], H: int, W: int):
        """
        Try to extract coords from input_obj (preferred: xarray ds),
        else fall back to integer pixel coords and time indices.
        """
        # Preferred: input_obj.ds is an xarray Dataset with coords
        if hasattr(input_obj, "ds"):
            ds = getattr(input_obj, "ds")
            # pick dimension/coord names flexibly
            time_name = "time" if "time" in ds.coords else ("t" if "t" in ds.coords else None)
            y_name = "y" if "y" in ds.coords else ("lat" if "lat" in ds.coords else None)
            x_name = "x" if "x" in ds.coords else ("lon" if "lon" in ds.coords else None)

            time_coord = ds[time_name].values[time_indices] if time_name else np.array(time_indices)
            y_coord = ds[y_name].values if y_name else np.arange(H)
            x_coord = ds[x_name].values if x_name else np.arange(W)

            # keep output dims named exactly y/x as you requested
            return time_coord, y_coord, x_coord

        # Fallback
        return np.array(time_indices), np.arange(H), np.arange(W)
    
    def _write_mask_cube(self, input_obj, masks_3d: np.ndarray, time_indices: list[int]):
        """
        Write a new NetCDF cube with dims (time, y, x, mask).
        masks_3d: (T, H, W)
        """
        if not self.out_nc_path:
            raise ValueError("write_mask_netcdf=True but out_nc_path was not provided")

        T, H, W = masks_3d.shape
        K = len(self.mask_labels)
        if K != 1:
            raise ValueError(f"This implementation expects exactly 1 mask label; got {K}")

        time_coord, y_coord, x_coord = self._coords_from_input(input_obj, time_indices, H, W)

        data_4d = masks_3d[..., None]  # (T,H,W,1)

        ds_out = xr.Dataset(
            data_vars={
                self.var_name: xr.DataArray(
                    data_4d,
                    dims=("time", "y", "x", "mask"),
                    coords={
                        "time": time_coord,
                        "y": y_coord,
                        "x": x_coord,
                        "mask": self.mask_labels,
                    },
                    attrs={
                        "long_name": "LULC mask output",
                        "source_model": self.net,
                        "band_names": ",".join(self.band_names),
                    },
                )
            },
            attrs={"creator": "L3.LulcClassification"},
        )

        out_path = Path(self.out_nc_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        encoding = None
        if self.compress:
            encoding = {self.var_name: {"zlib": True, "complevel": 4}}


        ds_out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

        return str(out_path)

    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        out: List[L3_result] = []
        masks_for_cube = []
        time_indices_done = []

        data_source = l2_datacube if l2_datacube is not None else input

        for time_index in self.time_indices:
            print(f"Processing LULC classification for time index {time_index} and bands {self.band_names}")

            img_3d = get_multiband_frame_from_input(data_source, time_index, self.band_names, fallback_input=input)  # (B,H,W)
            mask = infer_multiband_frame(self.model, self.device, img_3d)               # (H,W)

            masks_for_cube.append(mask)
            time_indices_done.append(time_index)


            debug_img = None
            if self.return_debug_image:
                rgb_vis = np.stack([img_3d[2], img_3d[1], img_3d[0]], axis=-1) / 10000.0
                debug_img = create_visualization_pil(rgb_vis, mask, time_idx=time_index, net=self.net)

            out.append(
                L3_result(
                    debug_image=debug_img,
                    algorithm_results={
                        "time_index": time_index,
                        "band_names": self.band_names,
                        "mask": mask,
                    },
                    time_indices=[time_index],
                    result_type="mask",
                )
            )
        if self.write_mask_netcdf:
            masks_3d = np.stack(masks_for_cube, axis=0).astype(np.uint8)  # (T,H,W)
            saved_path = self._write_mask_cube(input, masks_3d, time_indices_done)
            for r in out:
                r.algorithm_results["mask_cube_path"] = saved_path
        return out

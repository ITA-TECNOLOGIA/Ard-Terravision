# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import cv2
import pytorch_lightning as pl
from dotenv import load_dotenv
import os
import xarray as xr

from L2.L2_Algorithm import L2_Algorithm, L2_output
from L2.CloudMasking.satellite_cloud_removal_dip.src import LitDIP
from logger import logger
load_dotenv()

@dataclass
class CloudMaskingResult:
    image_with_clouds_resized: np.ndarray
    ground_truth_resized: np.ndarray
    cloud_mask_resized: np.ndarray
    inpainted_result: np.ndarray
    inpainted_result_resized: np.ndarray

class CloudMasking(L2_Algorithm):
    def __init__(self,
                gt_time_index: int | None = None,
                band_names: List[str] = [],
                rgb_band_names: List[List[str]] = []):
        
        self.gt_time_index = gt_time_index  
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1
        self.band_names = band_names
        self.rgb_band_names = rgb_band_names
        # TODO assert rgb_band_names are not repeated
        # TODO assert there is more than one and always 3
        DEVICE = os.getenv("DEVICE", "cpu")  # e.g. "cuda:0"
        if DEVICE.startswith("cuda"):
            self.accelerator = "gpu"
            self.devices     = [int(DEVICE.split(":", 1)[1])]
        else:
            self.accelerator = "cpu"
            self.devices     = 1

    def _find_best_gt_time_index(self, input, exclude_indices: List[int]) -> int:
        best_idx = None
        max_clean_score = -1.0

        try:
            num_times = input.datacube.sizes.get('t', 100)
        except:
            num_times = 100 

        for t in range(num_times):
            if t in exclude_indices:
                continue
            
            try:
                mask = input.get_cloud_mask(t)
                if mask is None:
                    continue

                clean_percentage = np.mean(mask)
                
                print(f"[CloudMasking] Evaluating index {t}: {clean_percentage*100:.2f}% clean (white)")

                if clean_percentage > max_clean_score:
                    max_clean_score = clean_percentage
                    best_idx = t
                
                if max_clean_score >= 0.999:
                    break
            except Exception:
                break

        if best_idx is None:
            raise ValueError("No valid image found.")

        print(f"[CloudMasking] SELECTED index {best_idx} as GT ({max_clean_score*100:.2f}% clean surface)")
        return best_idx

    def process_data(self, l1_inputs) -> L2_output:
        input = l1_inputs[0] if isinstance(l1_inputs, list) else l1_inputs
        processed_bands: Dict[str, np.ndarray] = {}
        processed_band_names: List[str] = []
        debug_images: List[Image.Image] = []

        original_datacube = input.get_datacube()

        if not self.time_indices:
            logger.warning(
                "CloudMasking has no time_indices injected. "
                "Falling back to all time indices from datacube."
            )
            self.time_indices = list(range(original_datacube.sizes.get('t', 0)))

        if self.gt_time_index is None:
            self.gt_time_index = self._find_best_gt_time_index(
                input,
                exclude_indices=self.time_indices
            )

        for time_index in self.time_indices:
            print("Only processing rgb bands")
            for rgb_band_name in self.rgb_band_names:
                cloud_mask = input.get_cloud_mask(time_index).squeeze(0)
                if cloud_mask.dtype == np.bool_:
                    cloud_mask = cloud_mask.astype(np.uint8)

                ground_truth = input.get_ground_truth(self.gt_time_index, rgb_band_name)
                ground_truth = np.nan_to_num(ground_truth)
                ground_truth = np.transpose(ground_truth, (1, 2, 0))
                image_with_clouds = input.get_image(time_index, rgb_band_name)
                image_with_clouds = np.nan_to_num(image_with_clouds)
                image_with_clouds = np.transpose(image_with_clouds, (1, 2, 0))

                original_size = image_with_clouds.shape[:2]

                image_with_clouds_resized = cv2.resize(image_with_clouds, (256, 256)).astype(np.float32)
                ground_truth_resized      = cv2.resize(ground_truth,      (256, 256)).astype(np.float32)
                cloud_mask_resized        = cv2.resize(cloud_mask,      (256, 256), interpolation=cv2.INTER_NEAREST)

                eps    = 1e-6
                max_ic = image_with_clouds_resized.max()
                max_gt = ground_truth_resized.max()
                image_with_clouds_resized = image_with_clouds_resized / (max_ic + eps)
                ground_truth_resized      = ground_truth_resized      / (max_gt + eps)

                ones_mask_resized = np.ones(cloud_mask_resized.shape, dtype=np.uint8)

                model = LitDIP()
                model.set_target([image_with_clouds_resized, ground_truth_resized])
                model.set_mask([cloud_mask_resized, ones_mask_resized])
                trainer = pl.Trainer(
                    max_epochs   = 4,
                    accelerator  = self.accelerator,
                    devices      = self.devices
                )
                trainer.fit(model)
                result, _ = model.output()

                result_resized = cv2.resize(result, (original_size[1], original_size[0]))

                mask_vis = (cloud_mask_resized * 255).astype(np.uint8)
                mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

                result_norm = result / (result.max() + eps)
                result_resized_norm = cv2.resize(result_norm, (256, 256)).astype(np.float32)

                top_row = np.hstack([image_with_clouds_resized, ground_truth_resized])
                bottom_row = np.hstack([mask_vis, result_resized_norm])
                composite = np.vstack([top_row, bottom_row])

                debug_img = Image.fromarray((composite * 255).astype(np.uint8))
                debug_images.append(debug_img)

                for i, band in enumerate(rgb_band_name):
                    processed_bands[f"{band}_cm"] = result_resized[:, :, i]
                    processed_band_names.append(f"{band}_cm")

        combined_debug = Image.new('RGB', (debug_images[0].width * len(debug_images), debug_images[0].height))
        for i, img in enumerate(debug_images):
            combined_debug.paste(img, (i * img.width, 0))

        new_datacube = original_datacube.copy(deep=False)
        for band_name, band_data in processed_bands.items():
            data_arrays = []
            for t_idx in self.time_indices:
                da = xr.DataArray(
                    band_data[np.newaxis, :, :],
                    dims=["t", "y", "x"],
                    coords={"t": [t_idx]}
                )
                data_arrays.append(da)
            if data_arrays:
                combined = xr.concat(data_arrays, dim="t")
                new_datacube[band_name] = combined

        processed_band_info: Dict[str, Any] = {
            "algorithm": "CloudMasking",
            "processed_band_names": processed_band_names,
            "time_indices": self.time_indices,
            "gt_time_index": self.gt_time_index,
        }

        return L2_output(
            datacube=new_datacube,
            debug_image=combined_debug,
            processed_band_info=processed_band_info
        )

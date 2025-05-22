# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List
from PIL import Image
import numpy as np
import cv2
import pytorch_lightning as pl
from dotenv import load_dotenv
import os

from L2.L2_Algorithm import L2_Algorithm, L2_result
from L2.CloudMasking.satellite_cloud_removal_dip.src import LitDIP
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
                 time_indices: List[int],
                 band_names: List[str] = [],
                 rgb_band_names: List[List[str]] = []):  # TODO actually is bgr
        self.time_indices = time_indices
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

    def process_data(self, input) -> List[L2_result]:
        results: List[L2_result] = []

        for time_index in self.time_indices:
            print("Only processing rgb bands") # TODO The algorithm should work with any band
            for rgb_band_name in self.rgb_band_names:
                # Get the cloud mask and convert from boolean to uint8 if needed.
                cloud_mask = input.get_cloud_mask(time_index).squeeze(0)
                if cloud_mask.dtype == np.bool_:
                    cloud_mask = cloud_mask.astype(np.uint8)

                # Retrieve and transpose ground truth and input image.
                ground_truth = input.get_ground_truth(time_index, rgb_band_name)
                ground_truth = np.transpose(ground_truth, (1, 2, 0))
                image_with_clouds = input.get_image(time_index, rgb_band_name)
                image_with_clouds = np.transpose(image_with_clouds, (1, 2, 0))

                # Save the original size for later resizing.
                original_size = image_with_clouds.shape[:2]

                # Resize the image, ground truth, and cloud mask to 256x256.
                image_with_clouds_resized = cv2.resize(image_with_clouds, (256, 256)).astype(np.float32)
                ground_truth_resized      = cv2.resize(ground_truth,      (256, 256)).astype(np.float32)
                cloud_mask_resized        = cv2.resize(cloud_mask,      (256, 256), interpolation=cv2.INTER_NEAREST)

                # Normalize by per-image max (add eps to avoid div-by-zero)
                eps    = 1e-6
                max_ic = image_with_clouds_resized.max()
                max_gt = ground_truth_resized.max()
                image_with_clouds_resized = image_with_clouds_resized / (max_ic + eps)
                ground_truth_resized      = ground_truth_resized      / (max_gt + eps)

                # Create a ones mask for the LitDIP model
                ones_mask_resized = np.ones(cloud_mask_resized.shape, dtype=np.uint8)

                # Setup and run the model
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

                # Resize the model output back to the original input size.
                result_resized = cv2.resize(result, (original_size[1], original_size[0]))

                # ---- DEBUG: create composite image ----
                mask_vis = (cloud_mask_resized * 255).astype(np.uint8)
                mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

                # Composite: top row [input | ground truth], bottom row [mask | result]
                result_norm = result / (result.max() + eps)
                result_resized_norm = cv2.resize(result_norm, (256, 256)).astype(np.float32)

                top_row = np.hstack([image_with_clouds_resized, ground_truth_resized])
                bottom_row = np.hstack([mask_vis, result_resized_norm])
                composite = np.vstack([top_row, bottom_row])

                # Convert numpy composite to PIL Image
                debug_img = Image.fromarray((composite * 255).astype(np.uint8))
                # ---------------------------------------

                # Build CloudMaskingResult
                cm_result = CloudMaskingResult(
                    image_with_clouds_resized=image_with_clouds_resized,
                    ground_truth_resized=ground_truth_resized,
                    cloud_mask_resized=cloud_mask_resized,
                    inpainted_result=result,
                    inpainted_result_resized=result_resized
                )

                # Append L2_result
                results.append(L2_result(
                    debug_image=debug_img,
                    algorithm_results=cm_result
                ))

                # Update the input with the inpainted image
                input.update_datacube(
                    time_index,
                    rgb_band_name,
                    result_resized
                )

        return results
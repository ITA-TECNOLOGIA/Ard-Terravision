import torch
from PIL import Image
import torchvision.transforms as tfs
import numpy as np
from typing import List, Any, Dict, Optional
import os
from dotenv import load_dotenv
import xarray as xr

# Assuming L3 imports are correct
from L3.L3_Algorithm import L3_Algorithm, L3_result
from L3.ChangeDetection.MineNetCD.upernet import UperNetForSemanticSegmentation

load_dotenv()

# =========================================================================
# HELPER FUNCTION FOR NORMALIZATION AND CONVERSION
# =========================================================================
def normalize_and_convert_to_pil(image_np: np.ndarray) -> Image.Image:
    """
    Performs min-max normalization on a NumPy array (H, W, C) to the [0, 1] range,
    then scales and converts it to a standard 8-bit PIL Image ([0, 255]).
    """
    # Ensure array is float for accurate division
    image_np = image_np.astype(np.float32)

    # 1. Min-Max Normalization to [0, 1]
    min_val = image_np.min()
    max_val = image_np.max()

    # Handle the case where max_val == min_val (constant image)
    if max_val == min_val:
        image_normalized = np.zeros_like(image_np, dtype=np.float32)
    else:
        image_normalized = (image_np - min_val) / (max_val - min_val)

    # 2. Scale back to [0, 255] and convert to unsigned 8-bit integer (np.uint8)
    image_uint8 = (image_normalized * 255).astype(np.uint8)

    # 3. Convert to PIL Image
    image_pil = Image.fromarray(image_uint8)

    return image_pil
# =========================================================================

def upscale_by_factor(img: Image.Image, alpha: float) -> Image.Image:
    w, h = img.size
    new_w = int(round(w * alpha))
    new_h = int(round(h * alpha))
    return img.resize((new_w, new_h), Image.LANCZOS)

def crop_by_percent(
    img: Image.Image,
    top: float = 0.3,
    bottom: float = 0.3,
    left: float = 0.2,
    right: float = 0.2,
) -> Image.Image:
    w, h = img.size
    left_px   = int(round(w * left))
    right_px  = int(round(w * (1 - right)))
    top_px    = int(round(h * top))
    bottom_px = int(round(h * (1 - bottom)))

    if left_px >= right_px or top_px >= bottom_px:
        raise ValueError("Too aggressive crop.")

    return img.crop((left_px, top_px, right_px, bottom_px))

class ChangeDetectionMineNetCD(L3_Algorithm):
    def __init__(self,
                 args_list: List[Dict[str, Any]]):
        self.checkpoint_dir = os.getenv("CHANGEDET_CHECKPOINT_DIR")

        if self.checkpoint_dir is None:
            raise ValueError("CHANGEDET_CHECKPOINT_DIR not defined in .env")

        super().__init__()
        if not isinstance(args_list, list) or not all(isinstance(d, dict) for d in args_list):
            raise ValueError("`args_list` must be a list of dicts")
        self.args_list = args_list

        # --- transformation ---
        # NOTE: ADE_MEAN/STD values are likely for 0-255 images, but your model 
        # may require these specific float values after ToTensor() converts to [0, 1].
        ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
        self.transform = tfs.Compose([
            tfs.ToTensor(), # Converts PIL Image to Tensor and scales [0, 255] to [0.0, 1.0]
            tfs.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ])

        # --- load model ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = UperNetForSemanticSegmentation.from_pretrained(self.checkpoint_dir, ignore_mismatched_sizes=True)
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        results: List[L3_result] = []
        data_source = l2_datacube if l2_datacube is not None else input
        getter = data_source.get_rgb_image if hasattr(data_source, 'get_rgb_image') else input.get_rgb_image

        for kwargs in self.args_list:
            time_index_A = kwargs.get("time_index_A")
            time_index_B = kwargs.get("time_index_B")

            if time_index_A is None or time_index_B is None:
                raise ValueError("Each dictionary in `args_list` must contain 'time_index_A' and 'time_index_B'")

            # Get images from the input data cube (expected H, W, C, float values)
            imageA_np = getter(time_index=time_index_A)
            imageB_np = getter(time_index=time_index_B)

            orig_h, orig_w = imageA_np.shape[:2]

            # Replace NaNs with 0
            imageA_np = np.nan_to_num(imageA_np)
            imageB_np = np.nan_to_num(imageB_np)

            UPSCALE_ALPHA = 3.25

            imageA_pil = normalize_and_convert_to_pil(imageA_np)
            imageB_pil = normalize_and_convert_to_pil(imageB_np)

            imageA_pil = crop_by_percent(imageA_pil, top=0, bottom=0, left=0, right=0)
            imageB_pil = crop_by_percent(imageB_pil, top=0, bottom=0, left=0, right=0)

            imageA_original = imageA_pil.copy()
            imageB_original = imageB_pil.copy()

            imageA_pil = upscale_by_factor(imageA_pil, UPSCALE_ALPHA)
            imageB_pil = upscale_by_factor(imageB_pil, UPSCALE_ALPHA)

            # Apply transformations
            # The tfs.ToTensor() step handles the conversion from PIL Image to Tensor and 
            # scales the [0, 255] PIL image data to [0.0, 1.0] Tensor data before Normalize.
            imageA_transformed = self.transform(imageA_pil).unsqueeze(0)
            imageB_transformed = self.transform(imageB_pil).unsqueeze(0)

            # Concatenate images to create the input tensor for the change detection model
            pixel_values = torch.cat([imageA_transformed, imageB_transformed], dim=0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                # Apply softmax and argmax to get the predicted change map
                pred = torch.argmax(torch.nn.functional.softmax(outputs.logits, dim=1), dim=1)

            # Post-process prediction
            # Scale the prediction map indices (e.g., 0, 1, 2) to a displayable range (0, 255)
            # This makes the change map visible as a grayscale image.
            pred_map_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            pred_map_pil = Image.fromarray(pred_map_np)
            pred_map_pil = pred_map_pil.resize(imageA_original.size, Image.NEAREST)
            pred_map_pil = pred_map_pil.resize((orig_w, orig_h), Image.NEAREST)

            # Create combined debug image
            imageA_display = imageA_pil.resize((orig_w, orig_h), Image.LANCZOS)
            imageB_display = imageB_pil.resize((orig_w, orig_h), Image.LANCZOS)
            w, h = imageA_original.size

            debug_image = Image.new("RGB", (w * 3, h))

            debug_image.paste(imageA_original, (0, 0))
            debug_image.paste(imageB_original, (w, 0))
            debug_image.paste(pred_map_pil.convert("RGB"), (w * 2, 0))

            results.append(L3_result(
                debug_image=debug_image,
                algorithm_results={"change_map": pred_map_np, "time_index_A": time_index_A, "time_index_B": time_index_B},
                time_indices=[time_index_A, time_index_B],
                result_type="change_map"
            ))

        return results
import os
import json
from datetime import datetime
from typing import Any, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

if TYPE_CHECKING:
    from L3.L3_Algorithm import L3_Algorithm, L3_result
    from L4.L4_Algorithm import L4_Algorithm


def get_base_output_dir() -> str:
    return os.getenv("OUTPUT_DIR", "./outputs")


def get_layer_from_instance(obj: Any) -> str:
    module = type(obj).__module__ or ""
    if "L4" in module:
        return "L4"
    elif "L3" in module:
        return "L3"
    return "Unknown"


def get_algorithm_name(obj: Any) -> str:
    return type(obj).__name__


def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_output_path(
    algorithm_obj: Any,
    filename: str,
    subdir: Optional[str] = None
) -> str:
    base = get_base_output_dir()
    layer = get_layer_from_instance(algorithm_obj)
    algo_name = get_algorithm_name(algorithm_obj)

    if subdir:
        path = os.path.join(base, layer, algo_name, subdir)
    else:
        path = os.path.join(base, layer, algo_name)

    ensure_output_dir(path)
    return os.path.join(path, filename)


def save_datacube(data: Union[xr.DataArray, xr.Dataset], path: str) -> None:
    if isinstance(data, xr.DataArray):
        data.to_netcdf(path)
    elif isinstance(data, xr.Dataset):
        data.to_netcdf(path)


def save_image(image: Image.Image, path: str) -> None:
    image.save(path, format="PNG")


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_text(content: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def save_detection_mask(
    detections: List[Any],
    height: int,
    width: int,
    num_classes: int,
    path: str
) -> None:
    mask = np.zeros((height, width, num_classes), dtype=np.float32)

    for det in detections:
        bbox = det.bbox
        class_id = det.class_id

        if isinstance(bbox, dict):
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("width", 0)
            h = bbox.get("height", 0)
        else:
            x, y, w, h = bbox

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)

        try:
            class_id_int = int(class_id) if not isinstance(class_id, int) else class_id
            if 0 <= class_id_int < num_classes:
                mask[y1:y2, x1:x2, class_id_int] = 1.0
        except (ValueError, TypeError):
            pass

    da = xr.DataArray(
        mask,
        dims=["y", "x", "class_id"],
        coords={
            "y": np.arange(height),
            "x": np.arange(width),
            "class_id": np.arange(num_classes)
        }
    )
    da.to_netcdf(path)


def extract_detection_metadata(result: Any, algo_name: str) -> dict:
    from L3.L3_Algorithm import L3_result
    algo_results = result.algorithm_results

    metadata = {
        "algorithm": algo_name,
        "result_type": result.result_type,
        "time_indices": result.time_indices,
        "timestamp": get_timestamp()
    }

    if hasattr(algo_results, "detections"):
        detections = algo_results.detections
        metadata["num_detections"] = len(detections)

        det_list = []
        for det in detections:
            det_dict = {
                "bbox": det.bbox,
                "class_id": det.class_id,
            }
            if hasattr(det, "confidence"):
                det_dict["confidence"] = det.confidence
            det_list.append(det_dict)

        metadata["detections"] = det_list

        if hasattr(algo_results, "kwargs"):
            metadata["kwargs"] = algo_results.kwargs

    return metadata


def save_l3_result(result: Any, algorithm_obj: Any, time_index: Optional[int] = None) -> None:
    from L3.L3_Algorithm import L3_result
    algo_name = get_algorithm_name(algorithm_obj)
    timestamp = get_timestamp()

    algo_results = result.algorithm_results

    if isinstance(algo_results, (xr.DataArray, xr.Dataset)):
        filename = f"{timestamp}.nc"
        path = build_output_path(algorithm_obj, filename)
        save_datacube(algo_results, path)

    if hasattr(algo_results, "detections"):
        idx_str = f"time_{time_index}" if time_index is not None else timestamp

        json_path = build_output_path(algorithm_obj, f"{idx_str}_detections.json")
        metadata = extract_detection_metadata(result, algo_name)
        save_json(metadata, json_path)

        if hasattr(algo_results, "kwargs"):
            h, w = 0, 0
            try:
                if result.debug_image is not None:
                    h, w = result.debug_image.size[1], result.debug_image.size[0]
                elif hasattr(algo_results, "detections") and algo_results.detections:
                    sample_bbox = algo_results.detections[0].bbox
                    if isinstance(sample_bbox, dict):
                        w = sample_bbox.get("x", 0) + sample_bbox.get("width", 100)
                        h = sample_bbox.get("y", 0) + sample_bbox.get("height", 100)
            except:
                h, w = 256, 256

            if h > 0 and w > 0:
                mask_path = build_output_path(algorithm_obj, f"{idx_str}_mask.nc")
                save_detection_mask(
                    algo_results.detections,
                    h, w,
                    num_classes=100,
                    path=mask_path
                )

    if result.debug_image is not None:
        image_to_save = result.visual_output if result.visual_output is not None else result.debug_image
        debug_filename = f"debug_{timestamp}.png"
        if time_index is not None:
            debug_filename = f"debug_time_{time_index}_{timestamp}.png"
        debug_path = build_output_path(algorithm_obj, debug_filename)
        save_image(image_to_save, debug_path)


def save_l4_result(results: Union[List[str], Any], algorithm_obj: Any) -> None:
    algo_name = get_algorithm_name(algorithm_obj)
    timestamp = get_timestamp()

    if isinstance(results, list):
        for i, result in enumerate(results):
            filename = f"{timestamp}_{i}.txt"
            path = build_output_path(algorithm_obj, filename)
            save_text(str(result), path)
    else:
        filename = f"{timestamp}.txt"
        path = build_output_path(algorithm_obj, filename)
        save_text(str(results), path)


def save_algorithm_output(results: Any, algorithm_obj: Any, time_index: Optional[int] = None) -> None:
    from L3.L3_Algorithm import L3_result
    layer = get_layer_from_instance(algorithm_obj)

    if layer == "L3":
        if isinstance(results, list):
            for result in results:
                if isinstance(result, L3_result):
                    save_l3_result(result, algorithm_obj, time_index)
        elif isinstance(results, L3_result):
            save_l3_result(results, algorithm_obj, time_index)
    elif layer == "L4":
        save_l4_result(results, algorithm_obj)
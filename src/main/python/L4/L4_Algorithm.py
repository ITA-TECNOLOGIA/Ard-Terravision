# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import List, Any, Optional
import os

import numpy as np
import cv2

from utils.output_writer import save_algorithm_output


class L4_Algorithm(ABC):
    """
    Abstract base class for Layer 4 algorithms focused on information fusion,
    large model interaction, reasoning, or generative AI over multi-source outputs.
    """

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__()
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "./outputs")

    def save_results(self, results: Any) -> None:
        save_algorithm_output(results, self)

    @abstractmethod
    def process_data(self, input, l3_results: List[Any], target_time_index: Optional[int] = None):
        """
        Process L1 input combined with L3 results to generate final outputs.

        Args:
            input (L1_Input): The L1 input instance for accessing raw data.
            l3_results (List[Any]): List of all L3 results from different algorithms.
            target_time_index (int, optional): Specific time index to query.
                                              If None, processes all available time indices.
        """
        pass

    @staticmethod
    def _normalize_bbox(bbox) -> tuple:
        if isinstance(bbox, dict):
            return (int(bbox["x"]), int(bbox["y"]),
                    int(bbox["x"] + bbox["width"]), int(bbox["y"] + bbox["height"]))
        return tuple(map(int, bbox))

    def _overlay_result_on_image(self, rgb_image: np.ndarray, l3_result: Any, time_index: int) -> np.ndarray:
        result_type = l3_result.result_type
        algo_results = l3_result.algorithm_results

        overlay = rgb_image.copy()

        if result_type == "mask" and algo_results is not None:
            mask = algo_results.get("mask")
            if mask is not None:
                mask_normalized = (mask / mask.max() * 255).astype(np.uint8) if mask.max() > 0 else mask.astype(np.uint8)
                cmap = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
                overlay = (overlay / overlay.max() * 255).astype(np.uint8) if overlay.max() > 0 else overlay.astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.7, cmap, 0.3, 0)

        elif result_type == "detections" and algo_results is not None:
            detections = algo_results.get("detections", [])
            for det in detections:
                bbox = det.bbox
                confidence = det.confidence
                x1, y1, x2, y2 = self._normalize_bbox(bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2f}"
                cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif result_type == "change_map" and algo_results is not None:
            change_map = algo_results.get("change_map")
            if change_map is not None:
                change_normalized = (change_map / change_map.max() * 255).astype(np.uint8) if change_map.max() > 0 else change_map.astype(np.uint8)
                cmap = cv2.applyColorMap(change_normalized, cv2.COLORMAP_RED)
                overlay = (overlay / overlay.max() * 255).astype(np.uint8) if overlay.max() > 0 else overlay.astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.7, cmap, 0.3, 0)

        return overlay

    def _format_l3_context(self, l3_results: List[Any], time_index: int) -> str:
        context_parts = []
        for result in l3_results:
            result_type = result.result_type
            algo_results = result.algorithm_results
            algo_name = algo_results.__class__.__name__ if hasattr(algo_results, '__class__') else str(type(algo_results))

            if result_type == "mask" and algo_results is not None:
                mask = algo_results.get("mask")
                if mask is not None:
                    unique_classes = np.unique(mask)
                    context_parts.append(f"Land use classification detected {len(unique_classes)} classes: {list(unique_classes)}")

            elif result_type == "detections" and algo_results is not None:
                if hasattr(algo_results, 'detections'):
                    detections = algo_results.detections
                else:
                    detections = algo_results.get("detections", [])
                if detections:
                    context_parts.append(f"Object detection found {len(detections)} objects")

            elif result_type == "change_map" and algo_results is not None:
                change_map = algo_results.get("change_map")
                if change_map is not None:
                    change_pixels = np.sum(change_map > 0)
                    context_parts.append(f"Change detection identified {change_pixels} changed pixels")

            elif result_type == "datacube" and algo_results is not None:
                context_parts.append("Environmental indicator datacube available for analysis")

        return "; ".join(context_parts) if context_parts else "No analysis results available"

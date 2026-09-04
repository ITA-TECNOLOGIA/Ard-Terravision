# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Any, Optional
from dataclasses import dataclass, field
import xarray as xr
import os

from utils.output_writer import save_algorithm_output


@dataclass
class L3_result:
    debug_image: Image.Image
    algorithm_results: Any
    time_indices: List[int] = field(default_factory=list)
    result_type: str = ""
    visual_output: Optional[Image.Image] = None


class L3_Algorithm(ABC):
    """
    Abstract base class for Layer 3 algorithms that process L1_Input instances.
    Optionally receives L2 processed datacube if L2 algorithms were run.
    """

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__()
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "./outputs")

    def save_results(self, results: List['L3_result'], time_index: Optional[int] = None) -> None:
        for result in results:
            save_algorithm_output(result, self, time_index)

    @abstractmethod
    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        """
        Core method to process Layer 1 input data and perform L3 operations,
        such as change detection, lulc classification, or object detection.

        Args:
            input (L1_Input): An instance of a class implementing L1_Input,
                              containing the data to be processed.
            l2_datacube (xr.Dataset, optional): Processed datacube from L2 algorithms.
                                               If provided, use this instead of raw L1 data.

        Returns:
            List[L3_result]: List of results, each containing debug image,
                             algorithm results, time indices covered, and result type.
        """
        pass


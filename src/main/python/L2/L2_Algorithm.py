# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from PIL import Image
from dataclasses import dataclass
from typing import List, Any, Dict, Optional
import xarray as xr

@dataclass
class L2_output:
    datacube: xr.Dataset
    debug_image: Image.Image
    processed_band_info: Dict[str, Any]

class L2_Algorithm(ABC):
    """
    Abstract base class for Level 2 algorithms that process L1_Input instances.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process_data(self, input) -> Optional[L2_output]:
        """
        Core method to process Level 1 input data and perform L2 operations,
        such as cloud masking, atmospheric correction, or enhancement.

        Args:
            input (L1_Input): An instance of a class implementing L1_Input,
                              containing the data to be processed.

        Returns:
            L2_output: Contains the processed datacube, debug image for
                       visualization, and metadata about processed bands.
                       Returns None if no processing is needed.
        """
        pass

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

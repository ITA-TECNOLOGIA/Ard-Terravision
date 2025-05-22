# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from abc import ABC, abstractmethod

class L4_Algorithm(ABC):
    """
    Abstract base class for Layer 4 algorithms focused on information fusion,
    large model interaction, reasoning, or generative AI over multi-source outputs.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process_data(self, input):
        pass

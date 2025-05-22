# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Any, Tuple, Sequence
import numpy as np

class L1_Input(ABC):
    """
    Abstract base class for Layer‑1 input data handlers.
    Subclasses must implement all core data operations for loading,
    inspecting, and modifying the underlying datacube.
    """
    @abstractmethod
    def get_debug_image(self) -> np.ndarray:
        """
        Generate and return a debug image representing the current state
        of the data (e.g., an overview or mosaic).

        Returns:
            A 2D or 3D NumPy array (HxW or HxWxC) suitable for display.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sun_angles(self, time_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve per-pixel sun azimuth and zenith angles for the specified timestep.

        Args:
            time_index: Integer index identifying the time/spectrum slice.

        Returns:
            A tuple of two NumPy arrays:
              - azimuth angles (HxW)
              - zenith angles  (HxW)
        """
        raise NotImplementedError

    @abstractmethod
    def get_view_angles(self, time_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve per-pixel view azimuth and zenith angles for the specified timestep.

        Args:
            time_index: Integer index identifying the time/spectrum slice.

        Returns:
            A tuple of two NumPy arrays:
              - view azimuth angles (HxW)
              - view zenith angles   (HxW)
        """
        raise NotImplementedError

    @abstractmethod
    def get_dem(self, time_index: int) -> np.ndarray:
        """
        Obtain the Digital Elevation Model (DEM) for the given timestep.

        Args:
            time_index: Integer index identifying the time slice.

        Returns:
            A 2D NumPy array (HxW) of elevation values.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cloud_mask(self, time_index: int) -> np.ndarray:
        """
        Generate a binary cloud mask for the given timestep.

        Args:
            time_index: Integer index identifying the time slice.

        Returns:
            A 2D NumPy array (HxW) of dtype uint8 or bool, where
            True/1 indicates clear sky and False/0 indicates cloud.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self, time_index: int, band_indices: Sequence[str]) -> np.ndarray:
        """
        Retrieve a ground-truth/reference image for supervised tasks.

        Args:
            time_index: Integer index for the time slice.
            band_indices: Sequence of band names to include (e.g. ['B01','B02',...]).

        Returns:
            A NumPy array (HxWxlen(band_indices)) representing the requested bands.
        """
        raise NotImplementedError

    @abstractmethod
    def update_datacube(
        self,
        time_index: int,
        band_indices: Sequence[str],
        new_values: np.ndarray
    ) -> None:
        """
        Modify the internal datacube at the specified time and bands.

        Args:
            time_index: Integer index for the time slice.
            band_indices: Sequence of band names to update.
            new_values: NumPy array (HxWxlen(band_indices)) of replacement values.

        Raises:
            ValueError if indices are out of range or shapes mismatch.
        """
        raise NotImplementedError

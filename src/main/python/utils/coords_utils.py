import numpy as np
import xarray as xr
from typing import Any


def extract_spatial_coords(obj: Any, H: int, W: int) -> dict:
    """
    Extract spatial y/x coordinates from an L1 input object or xarray Dataset.

    Returns dict with keys 'y' and 'x', each a 1D np.ndarray.
    Falls back to np.arange(H), np.arange(W) if no coordinates are found
    or if the source coordinates don't match the requested shape.
    """

    ds = _resolve_dataset(obj)
    if ds is None:
        return {"y": np.arange(H), "x": np.arange(W)}

    y_name = _first_present(ds, "y", "lat")
    x_name = _first_present(ds, "x", "lon")

    y_coord = ds[y_name].values if y_name else None
    x_coord = ds[x_name].values if x_name else None

    if y_coord is not None and len(y_coord) != H:
        y_coord = None
    if x_coord is not None and len(x_coord) != W:
        x_coord = None

    return {
        "y": y_coord if y_coord is not None else np.arange(H),
        "x": x_coord if x_coord is not None else np.arange(W),
    }


def _resolve_dataset(obj: Any):
    """
    Extract an xarray Dataset from an object.
    Handles: raw xr.Dataset, Satellite (obj.datacube), Airborne (obj.ds).
    """
    if isinstance(obj, xr.Dataset):
        return obj

    if hasattr(obj, "ds") and isinstance(obj.ds, xr.Dataset):
        return obj.ds

    if hasattr(obj, "datacube") and isinstance(obj.datacube, xr.Dataset):
        return obj.datacube

    return None


def _first_present(ds: xr.Dataset, *candidates: str):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None

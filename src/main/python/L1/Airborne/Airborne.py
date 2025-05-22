# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import os
import glob
from typing import List, Optional

import numpy as np
import spectral as spy
import xarray as xr
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from tqdm import tqdm

from L1.L1_Input import L1_Input
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from logger import logger

class Airborne(L1_Input):
    def __init__(
        self,
        path: str,
        hslvl3_project: str,
        time_indices: Optional[List[int]] = None
    ):
        """
        Initialize the Airborne loader.

        :param path: Base path to the data folder
        :param hslvl3_project: Project name prefix for hyperspectral files
        :param time_indices: Optional list of `line` identifiers to filter which files to load
        """
        super().__init__()
        self.path = path
        self.hslvl3_project = hslvl3_project
        self.time_indices = time_indices
        self.ds = xr.Dataset()

        # Load hyperspectral, then images, lidar and thermal (all without padding)
        self._load_hslvl3()
        self._load_images()
        self._load_lidar()
        self._load_thermal()

        logger.info(f"✅ Final dataset contains {len(self.ds.data_vars)} bands")

    def _load_hslvl3(self) -> None:
        base_folder = os.path.join(self.path, "hyperspectral", "hslvl3")
        if not os.path.isdir(base_folder):
            raise FileNotFoundError(f"{base_folder} does not exist")

        hdr_files = sorted(
            fn for fn in os.listdir(base_folder)
            if fn.lower().endswith(".hdr")
        )

        for fn in tqdm(hdr_files, desc="Loading HSLVL3"):
            name, _ = os.path.splitext(fn)
            parts = name.split("_")
            if len(parts) < 7:
                tqdm.write(f"Skipping unexpected name: {fn}")
                continue

            project, line, hypex, sensor, serial, fov, *desc = parts
            line = int(line)
            if project != self.hslvl3_project:
                tqdm.write(f"  • skip project {project}")
                continue
            if self.time_indices and line not in self.time_indices:
                tqdm.write(f"  • skip line {line}")
                continue

            descriptor = name  # full descriptor
            hdr_path = os.path.join(base_folder, fn)

            # load the raw array (rows × cols × bands)
            img = spy.open_image(hdr_path)
            arr = img.load().astype(np.float32)
            arr = np.squeeze(arr)  # maybe drop band dim if single

            # shape and coords
            rows, cols = arr.shape[:2]
            coords = {"y": np.arange(rows), "x": np.arange(cols)}

            if arr.ndim == 3:
                # one DataArray per band
                for b in range(arr.shape[2]):
                    band_name = f"{descriptor}_B{b+1:03d}"
                    da = xr.DataArray(
                        arr[..., b],
                        dims=("y", "x"),
                        coords=coords,
                        attrs={
                            "project": project,
                            "line": line,
                            "hypex": hypex,
                            "sensor": sensor,
                            "serial": serial,
                            "fov": fov
                        }
                    )
                    self.ds[band_name] = da

            else:
                da = xr.DataArray(
                    arr,
                    dims=("y", "x"),
                    coords=coords,
                    attrs={
                        "project": project,
                        "line": line,
                        "hypex": hypex,
                        "sensor": sensor,
                        "serial": serial,
                        "fov": fov
                    }
                )
                self.ds[descriptor] = da

        logger.info(f"✅ Loaded {len(self.ds.data_vars)} hyperspectral bands")
    
    def _load_images(self) -> None:
        """
        Load all tiled orthophoto .tif files, merge into a single mosaic,
        and generate an overview image with tile boundaries and filenames.
        Also stores the overview PNG as a NumPy array in self._debug_image.
        """
        folder = os.path.join(self.path, "images", "tiled_ortho")
        if not os.path.isdir(folder):
            tqdm.write("Skipping images (folder not found)")
            return

        tile_paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
        tile_paths = tile_paths[:10]  # TODO remove this limit
        if not tile_paths:
            tqdm.write("No image tiles found in tiled_ortho")
            return

        # Reference CRS from first tile
        with rasterio.open(tile_paths[0]) as src0:
            ref_crs = src0.crs

        # Partition into matching‐CRS and mismatched
        good_tiles, bad_tiles = [], []
        for p in tile_paths:
            with rasterio.open(p) as src:
                (good_tiles if src.crs == ref_crs else bad_tiles).append(p)

        # Reassign CRS on the bad ones, then treat them as good
        if bad_tiles:
            tqdm.write("Reassigning CRS on mismatched tiles:")
            for p in bad_tiles:
                tqdm.write(f"  • {p}")
                with rasterio.open(p, 'r+') as dst:
                    dst.crs = ref_crs
            good_tiles += bad_tiles

        # Open all tiles for merging
        src_files = [rasterio.open(p) for p in good_tiles]
        orig_res_x, orig_res_y = src_files[0].res
        factor = 100
        coarse_res = (orig_res_x * factor, orig_res_y * factor)

        # Merge into mosaic
        mosaic, transform = merge(
            src_files,
            res=coarse_res,
            resampling=Resampling.average
        )
        # Close inputs now that we've merged
        for src in src_files:
            src.close()

        bands, rows, cols = mosaic.shape

        # Create xarray DataArray
        coords = {"y": np.arange(rows), "x": np.arange(cols)}
        da = xr.DataArray(
            mosaic,
            dims=("band", "y", "x"),
            coords=coords,
            attrs={"transform": transform, "crs": ref_crs.to_string()}
        )
        self.ds["MOSAIC"] = da
        tqdm.write(f"Loaded mosaic with shape {mosaic.shape}")

        # ——— Overview generation ———
        fig, ax = plt.subplots(figsize=(12, 12))

        extent = (
            transform[2],
            transform[2] + cols * transform[0],
            transform[5] + rows * transform[4],
            transform[5]
        )

        if bands >= 3:
            img = np.transpose(mosaic[:3], (1, 2, 0)).astype(np.float32)
            img_norm = np.empty_like(img)
            for c in range(3):
                ch = img[:, :, c]
                lo, hi = ch.min(), ch.max()
                img_norm[:, :, c] = (ch - lo) / (hi - lo)
            ax.imshow(img_norm, extent=extent)
        else:
            band = mosaic[0].astype(np.float32)
            lo, hi = band.min(), band.max()
            ax.imshow((band - lo) / (hi - lo), extent=extent, cmap='gray')

        # Draw tile boundaries & labels
        for p in good_tiles:
            with rasterio.open(p) as src:
                b = src.bounds
            w, h = b.right - b.left, b.top - b.bottom

            rect = Rectangle(
                (b.left, b.bottom), w, h,
                edgecolor='red', facecolor='none', linewidth=1
            )
            ax.add_patch(rect)

            cx, cy = b.left + w/2, b.bottom + h/2
            name = os.path.splitext(os.path.basename(p))[0]
            ax.text(cx, cy, name,
                    color='red', ha='center', va='center', fontsize=15)

        ax.set_title("Mosaic with Tile Boundaries and Labels")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        plt.tight_layout()

        # Save and cache
        overview_path = os.path.join(self.path, "mosaic_overview.png")
        fig.savefig(overview_path, dpi=150)
        plt.close(fig)
        tqdm.write(f"Mosaic overview saved to {overview_path}")

        # Read it back into memory once and for all
        self._debug_image = plt.imread(overview_path)


    def get_debug_image(self) -> np.ndarray:
        """
        Returns the stored debug image (the mosaic overview) as a NumPy array.
        If it's not yet generated, calls _load_images() to produce it.
        """
        if not hasattr(self, "_debug_image"):
            tqdm.write("Debug image not yet generated; creating now...")
            self._load_images()

        return self._debug_image

    def _load_lidar(self) -> None:
        for folder, prefix in [
            (os.path.join(self.path, "lidar", "dsm"), "DSM"),
            (os.path.join(self.path, "lidar", "dtm"), "DTM")
        ]:
            if not os.path.isdir(folder):
                tqdm.write(f"Skipping {prefix} (folder not found)")
                continue

            for tif in sorted(glob.glob(os.path.join(folder, "*.tif"))):
                base = os.path.splitext(os.path.basename(tif))[0].upper()
                band_name = f"{prefix}_{base}"
                with rasterio.open(tif) as src:
                    arr = src.read(1).astype(np.float32)
                rows, cols = arr.shape
                coords = {f"{prefix}_y": np.arange(rows),
                          f"{prefix}_x": np.arange(cols)}
                da = xr.DataArray(
                    arr,
                    dims=(f"{prefix}_y", f"{prefix}_x"),
                    coords=coords,
                    attrs={"source": prefix, "file": base}
                )
                self.ds[band_name] = da
                tqdm.write(f"Loaded {band_name} ({rows}×{cols})")

    def _load_thermal(self) -> None:
        folder = os.path.join(self.path, "thermal")
        if not os.path.isdir(folder):
            tqdm.write("Skipping thermal (folder not found)")
            return

        for tif in sorted(glob.glob(os.path.join(folder, "*.tif"))):
            base = os.path.splitext(os.path.basename(tif))[0].upper()
            prefix = "REL_TEMP" if "REL" in base else "THERMAL"
            band_name = f"{prefix}_{base}"
            with rasterio.open(tif) as src:
                arr = src.read(1).astype(np.float32)
            rows, cols = arr.shape
            coords = {f"{prefix}_y": np.arange(rows),
                      f"{prefix}_x": np.arange(cols)}
            da = xr.DataArray(
                arr,
                dims=(f"{prefix}_y", f"{prefix}_x"),
                coords=coords,
                attrs={"source": prefix, "file": base}
            )
            self.ds[band_name] = da
            tqdm.write(f"Loaded {band_name} ({rows}×{cols})")

    def get_datacube(self) -> xr.Dataset:
        return self.ds

    def get_sun_angles(self, time_index: int):
        raise NotImplementedError("get_sun_angles not implemented for Airborne")

    def get_view_angles(self, time_index: int):
        raise NotImplementedError("get_view_angles not implemented for Airborne")

    def get_dem(self, time_index: int):
        raise NotImplementedError("get_DEM not implemented for Airborne")

    def get_cloud_mask(self, time_index: int):
        raise NotImplementedError("get_cloud_mask not implemented for Airborne")

    def get_ground_truth(self, time_index: int, band_indices: List[str]):
        raise NotImplementedError("get_ground_truth not implemented for Airborne")

    def get_rgb_image(
        self,
        airborne_tile: str,
        max_side: Optional[int] = 1000 # NOTE THAT BY DEFAULT DOES RESIZING
    ) -> np.ndarray:
        """
        Return an RGB image array for the specified airborne tile name
        (without file extension) by reading bands 1, 2 and 3 of its .tif.
        If max_side is provided, the longer edge of the output will be
        resized to max_side pixels (preserving aspect ratio).
        """
        folder = os.path.join(self.path, "images", "tiled_ortho")
        # try the exact .tif
        tile_file = os.path.join(folder, f"{airborne_tile}.tif")
        if not os.path.exists(tile_file):
            # fallback: case-insensitive or other extension
            candidates = glob.glob(os.path.join(folder, f"{airborne_tile}.*"))
            tif_candidates = [
                f for f in candidates
                if os.path.splitext(f)[1].lower() == ".tif"
            ]
            if tif_candidates:
                tile_file = tif_candidates[0]
            else:
                raise FileNotFoundError(
                    f"Tile '{airborne_tile}' not found in {folder}"
                )

        with rasterio.open(tile_file) as src:
            if src.count < 3:
                raise ValueError(
                    f"Tile '{airborne_tile}' has {src.count} bands; "
                    "at least 3 required for RGB."
                )

            # decide whether to resample
            if max_side is not None:
                # original dimensions
                orig_h, orig_w = src.height, src.width
                # only downscale if the longer side exceeds max_side
                scale = min(1.0, max_side / max(orig_h, orig_w))
                if scale < 1.0:
                    new_h = int(orig_h * scale)
                    new_w = int(orig_w * scale)
                    arr = src.read(
                        [1, 2, 3],
                        out_shape=(3, new_h, new_w),
                        resampling=Resampling.bilinear
                    ).astype(np.float32)
                else:
                    # no need to resize
                    arr = src.read([1, 2, 3]).astype(np.float32)
            else:
                arr = src.read([1, 2, 3]).astype(np.float32)

        # transpose to (height, width, 3)
        rgb = np.transpose(arr, (1, 2, 0))
        return rgb

    def update_datacube(
        self,
        time_index: int,
        band_indices: List[str],
        new_values: np.ndarray
    ):
        raise NotImplementedError("update_datacube not implemented for Airborne")

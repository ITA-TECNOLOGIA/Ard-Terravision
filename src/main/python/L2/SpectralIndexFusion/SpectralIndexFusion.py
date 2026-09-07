# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
from typing import Any, List, Optional
from PIL import Image

from pyproj import CRS, Transformer
from L2.L2_Algorithm import L2_Algorithm, L2_output
from logger import logger

_VALID_FREQS = {"monthly": "MS", "yearly": "YS"}
_VALID_STATS = {"mean", "median", "min", "max", "sum"}


class SpectralIndexFusion(L2_Algorithm):
    def __init__(
        self,
        temporal_aggregation: Optional[str] = None,
        agg_statistic: str = "mean",
        temporal_alignment_tolerance: Optional[str] = None,
        spatial_alignment: str = "upsample",
    ):
        self._fused_datacube = None
        self.time_indices: List[int] = []

        if temporal_aggregation is not None and temporal_aggregation not in _VALID_FREQS:
            raise ValueError(
                f"temporal_aggregation must be one of {sorted(_VALID_FREQS)} or None, "
                f"got '{temporal_aggregation}'"
            )
        if agg_statistic not in _VALID_STATS:
            raise ValueError(
                f"agg_statistic must be one of {sorted(_VALID_STATS)}, "
                f"got '{agg_statistic}'"
            )
        if temporal_alignment_tolerance is not None and not temporal_alignment_tolerance:
            raise ValueError(
                "temporal_alignment_tolerance must be a non-empty string or None"
            )
        if spatial_alignment not in ("upsample", "none"):
            raise ValueError(
                f"spatial_alignment must be 'upsample' or 'none', "
                f"got '{spatial_alignment}'"
            )

        self.temporal_aggregation = temporal_aggregation
        self.agg_statistic = agg_statistic
        self.temporal_alignment_tolerance = temporal_alignment_tolerance
        self.spatial_alignment = spatial_alignment

    def _resolve_common_time_indices(self, l1_inputs: List[Any]) -> Optional[List[int]]:
        all_time_indices = []
        for inp in l1_inputs:
            ti = getattr(inp, 'time_indices', None)
            all_time_indices.append(set(ti) if ti else set())

        non_empty = [ti for ti in all_time_indices if ti]
        if not non_empty:
            return None

        if len(non_empty) == 1 and non_empty[0]:
            return sorted(non_empty[0])
        elif len(non_empty) > 1:
            if any(not ti for ti in all_time_indices):
                raise ValueError(
                    "Inconsistent time_indices across L1 inputs. "
                    "Some have time_indices, some don't."
                )

            common = non_empty[0]
            for ti in non_empty[1:]:
                common &= ti

            if not common:
                raise ValueError(
                    f"No overlapping time indices across L1 inputs. "
                    f"Got: {[sorted(ti) for ti in all_time_indices]}"
                )

            if any(ti != common for ti in non_empty):
                logger.warning(
                    f"L1 inputs have different time_indices. Using intersection: "
                    f"{sorted(common)}. Original: {[sorted(ti) for ti in all_time_indices]}"
                )

            return sorted(common)

        return None

    def _apply_temporal_aggregation(self, fused_dataset: xr.Dataset) -> xr.Dataset:
        freq = _VALID_FREQS[self.temporal_aggregation]
        logger.info(
            f"Applying temporal aggregation: freq={self.temporal_aggregation} ({freq}), "
            f"statistic={self.agg_statistic}"
        )

        resampler = fused_dataset.resample({"t": freq})
        aggregated = getattr(resampler, self.agg_statistic)(skipna=True)

        logger.info(
            f"Temporal aggregation complete. Timesteps: "
            f"{fused_dataset.sizes['t']} -> {aggregated.sizes['t']}"
        )
        return aggregated

    def _trim_to_overlapping_dates(self, datacubes: List[xr.Dataset]) -> List[xr.Dataset]:
        starts = [ds["t"].values[0] for ds in datacubes if "t" in ds.dims and ds.sizes.get("t", 0) > 0]
        ends = [ds["t"].values[-1] for ds in datacubes if "t" in ds.dims and ds.sizes.get("t", 0) > 0]
        if not starts or not ends:
            return datacubes
        start = max(starts)
        end = min(ends)
        if start >= end:
            raise ValueError(
                f"No overlapping date range across datasets. "
                f"Starts: {starts}, Ends: {ends}"
            )
        logger.info(f"Trimming to overlapping date range: {start} to {end}")
        return [ds.sel(t=slice(start, end)) for ds in datacubes]

    def _align_temporal(self, datacubes: List[xr.Dataset], tolerance: str) -> List[xr.Dataset]:
        logger.info(f"Aligning {len(datacubes)} datasets temporally (tolerance={tolerance})")
        tol = pd.Timedelta(tolerance)

        ref_idx = min(range(len(datacubes)), key=lambda i: datacubes[i].sizes.get("t", 0))
        ref_times = datacubes[ref_idx]["t"].values
        ref_var = list(datacubes[ref_idx].data_vars)[0] if datacubes[ref_idx].data_vars else f"dataset_{ref_idx}"
        logger.info(f"  Reference: {ref_var} with {len(ref_times)} timesteps (sparsest)")

        aligned = []
        for i, ds in enumerate(datacubes):
            if i == ref_idx:
                aligned.append(ds)
                continue
            crs_var = ds.get("crs", None)
            crs_attrs = crs_var.attrs.copy() if crs_var is not None else None
            ds_aligned = ds.reindex(t=ref_times, method="nearest", tolerance=tol)
            if crs_attrs is not None and "crs" not in ds_aligned:
                ds_aligned["crs"] = xr.DataArray(b"", attrs=crs_attrs)
            n_matched = ds_aligned.sizes.get("t", 0)
            var_name = list(ds.data_vars)[0] if ds.data_vars else f"dataset_{i}"
            logger.info(f"  {var_name}: reindexed to reference grid ({n_matched} timesteps)")
            aligned.append(ds_aligned)

        return aligned

    def _filter_incomplete_timesteps(self, datacubes: List[xr.Dataset]) -> List[xr.Dataset]:
        if len(datacubes) <= 1:
            return datacubes
        valid_masks = []
        labels = []
        for ds in datacubes:
            spatial_dims = [d for d in ds.dims if d != "t"]
            if not spatial_dims or "t" not in ds.dims:
                valid_masks.append(None)
                continue
            per_timestep_valid = None
            for var in ds.data_vars:
                data = ds[var]
                has_data = data.notnull().any(dim=spatial_dims).values
                if per_timestep_valid is None:
                    per_timestep_valid = has_data
                else:
                    per_timestep_valid |= has_data
            if per_timestep_valid is not None:
                valid_masks.append(per_timestep_valid)
            else:
                valid_masks.append(None)
            labels.append(list(ds.data_vars)[0] if ds.data_vars else "unknown")

        masks_to_check = [m for m in valid_masks if m is not None]
        if not masks_to_check:
            return datacubes

        all_valid = masks_to_check[0].copy()
        for mask in masks_to_check[1:]:
            all_valid &= mask

        valid_indices = np.where(all_valid)[0]
        n_total = len(all_valid)
        n_dropped = n_total - len(valid_indices)

        if n_dropped > 0:
            dropped_indices = np.where(~all_valid)[0]
            times = datacubes[0]["t"].values
            logger.warning(
                f"Dropping {n_dropped}/{n_total} timesteps with incomplete data "
                f"(missing in at least one dataset)."
            )
            for idx in dropped_indices:
                date_str = str(times[idx])[:10]
                missing = []
                for j, mask in enumerate(valid_masks):
                    if mask is not None and not mask[idx]:
                        missing.append(labels[j] if j < len(labels) else f"dataset_{j}")
                logger.warning(f"  Dropped {date_str}: missing in {', '.join(missing)}")

        if len(valid_indices) == 0:
            logger.warning("All timesteps have incomplete data. Returning without filtering.")
            return datacubes

        return [ds.isel(t=valid_indices) for ds in datacubes]

    def _align_spatial(self, datacubes: List[xr.Dataset]) -> List[xr.Dataset]:
        if len(datacubes) <= 1:
            return datacubes
        if self.spatial_alignment == "none":
            return datacubes

        grid_sizes = [(ds.sizes.get("x", 0) * ds.sizes.get("y", 0)) for ds in datacubes]
        finest_idx = int(np.argmax(grid_sizes))
        target_x = datacubes[finest_idx]["x"].values
        target_y = datacubes[finest_idx]["y"].values

        target_crs = self._extract_crs(datacubes[finest_idx])

        aligned = []
        for i, ds in enumerate(datacubes):
            if i == finest_idx:
                aligned.append(ds)
                continue
            src_x = ds.sizes.get("x", 0)
            src_y = ds.sizes.get("y", 0)
            var_name = list(ds.data_vars)[0] if ds.data_vars else f"dataset_{i}"

            if (src_x == len(target_x) and src_y == len(target_y)
                    and np.array_equal(ds["x"].values, target_x)
                    and np.array_equal(ds["y"].values, target_y)):
                logger.info(f"  {var_name}: already at target grid ({src_x}x{src_y}), skipping interpolation")
                aligned.append(ds)
                continue

            src_crs = self._extract_crs(ds)

            if src_crs is not None and target_crs is None:
                target_crs = self._infer_utm_crs(src_crs, ds)
                if target_crs is not None:
                    logger.info(f"Inferred target CRS as {target_crs} from {var_name} coordinates")

            if src_crs is not None and target_crs is not None and src_crs != target_crs:
                logger.info(f"CRS mismatch for {var_name}: {src_crs} -> {target_crs}, transforming coordinates")
                ds = self._reproject_coords(ds, src_crs, target_crs)

            logger.info(f"Upsampling {var_name} from {src_x}x{src_y} to {len(target_x)}x{len(target_y)} (nearest-neighbor)")
            ds_upsampled = ds.interp(x=target_x, y=target_y, method="nearest")
            aligned.append(ds_upsampled)

        return aligned

    @staticmethod
    def _infer_utm_crs(src_crs: CRS, ds: xr.Dataset):
        if src_crs.to_epsg() != 4326:
            return None
        x_vals = ds["x"].values
        y_vals = ds["y"].values
        if len(x_vals) == 0 or len(y_vals) == 0:
            return None
        center_lon = float(np.mean(x_vals))
        center_lat = float(np.mean(y_vals))
        zone = int((center_lon + 180) / 6) + 1
        if center_lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone
        try:
            inferred = CRS.from_epsg(epsg)
            logger.info(f"Inferring UTM zone {zone} {'N' if center_lat >= 0 else 'S'} from lon={center_lon:.2f}, lat={center_lat:.2f} -> EPSG:{epsg}")
            return inferred
        except Exception:
            return None

    @staticmethod
    def _extract_crs(ds: xr.Dataset):
        crs_var = ds.get("crs", None)
        if crs_var is not None:
            wkt = crs_var.attrs.get("spatial_ref", "") or crs_var.attrs.get("crs_wkt", "")
            if wkt:
                try:
                    return CRS.from_wkt(wkt)
                except Exception:
                    pass
            epsg = crs_var.attrs.get("epsg_code", None)
            if epsg:
                try:
                    return CRS.from_epsg(int(epsg))
                except Exception:
                    pass
        x_vals = ds["x"].values
        if len(x_vals) > 0 and abs(x_vals[0]) < 360:
            return CRS.from_epsg(4326)
        return None

    @staticmethod
    def _reproject_coords(ds: xr.Dataset, src_crs: CRS, target_crs: CRS) -> xr.Dataset:
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        x_orig = ds["x"].values
        y_orig = ds["y"].values
        xx, yy = np.meshgrid(x_orig, y_orig)
        x_proj, y_proj = transformer.transform(xx, yy)
        new_x = x_proj[0, :]
        new_y = y_proj[:, 0]
        ds = ds.assign_coords(x=new_x, y=new_y)
        return ds

    def process_data(self, l1_inputs: List[Any]) -> Optional[L2_output]:
        if not l1_inputs:
            raise ValueError("No L1 inputs provided for fusion")

        logger.info(f"Fusing {len(l1_inputs)} L1 inputs")

        datacubes = [l1_instance.get_datacube() for l1_instance in l1_inputs]

        if self.temporal_alignment_tolerance is not None:
            logger.info("Using multi-sensor alignment path "
                        f"(temporal_alignment_tolerance={self.temporal_alignment_tolerance}, "
                        f"spatial_alignment={self.spatial_alignment})")
            datacubes = self._trim_to_overlapping_dates(datacubes)
            datacubes = self._align_temporal(datacubes, self.temporal_alignment_tolerance)
            datacubes = self._filter_incomplete_timesteps(datacubes)
            datacubes = self._align_spatial(datacubes)
            fused_dataset = xr.merge(datacubes, join="inner")
        else:
            common_time_indices = self._resolve_common_time_indices(l1_inputs)
            if common_time_indices is not None:
                logger.info(f"Using common time indices: {common_time_indices}")
                datacubes = [
                    ds.isel(t=common_time_indices) if common_time_indices else ds
                    for ds in datacubes
                ]
            aligned_datacubes = xr.align(*datacubes, join="exact")
            fused_dataset = xr.merge(aligned_datacubes, compat="equals")

        original_num_timesteps = fused_dataset.sizes.get("t", 0)
        if self.temporal_aggregation is not None:
            fused_dataset = self._apply_temporal_aggregation(fused_dataset)

        self._fused_datacube = fused_dataset

        debug_image = self._generate_debug_image()

        processed_band_info = {
            "spectral_indices": list(fused_dataset.data_vars),
            "num_indices": len(fused_dataset.data_vars),
            "fused_shape": dict(fused_dataset.dims),
            "temporal_aggregation": self.temporal_aggregation,
            "agg_statistic": self.agg_statistic if self.temporal_aggregation else None,
            "temporal_alignment_tolerance": self.temporal_alignment_tolerance,
            "spatial_alignment": self.spatial_alignment,
            "original_num_timesteps": original_num_timesteps,
            "aggregated_num_timesteps": fused_dataset.sizes.get("t", 0),
        }

        logger.info(f"Spectral index fusion completed. Fused dataset shape: {fused_dataset.dims}")

        return L2_output(
            datacube=fused_dataset,
            debug_image=debug_image,
            processed_band_info=processed_band_info
        )

    def _generate_debug_image(self) -> Image.Image:
        if self._fused_datacube is None:
            return Image.new("RGB", (256, 256), (128, 128, 128))

        first_var = list(self._fused_datacube.data_vars)[0]
        data = self._fused_datacube[first_var]

        if "t" in data.dims and data.sizes["t"] > 0:
            data = data.isel(t=0)

        arr = data.values

        if arr.ndim == 2:
            arr = np.nan_to_num(arr, nan=0.0)
            vmin, vmax = np.min(arr), np.max(arr)
            if vmax - vmin > 0:
                arr = (arr - vmin) / (vmax - vmin)
            else:
                arr = np.zeros_like(arr)

            arr = (arr * 255).astype(np.uint8)
            arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr, mode="RGB")

        return Image.new("RGB", (256, 256), (128, 128, 128))

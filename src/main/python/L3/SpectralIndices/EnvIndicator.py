# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

import os
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import mapping
from pyproj import CRS, Transformer

from L3.L3_Algorithm import L3_Algorithm, L3_result
from logger import logger


class EnvIndicator(L3_Algorithm):
    def __init__(
        self,
        mask_water_bodies: bool = False,
        water_shapefile: str = "",
        similarity_threshold: float = 0.7,
    ):
        self.time_indices: List[int] = []
        self.mask_water_bodies = mask_water_bodies
        self.water_shapefile = water_shapefile
        self.similarity_threshold = similarity_threshold

    def _build_water_mask(self, fused_datacube: xr.Dataset) -> Optional[np.ndarray]:
        if not self.water_shapefile:
            return None

        shapefile_path = self.water_shapefile
        if not os.path.isabs(shapefile_path):
            shapefile_path = os.path.join(
                os.getenv("WATER_SHAPEFILES_PATH", "data/water_shapefiles"),
                shapefile_path,
            )

        if not os.path.isfile(shapefile_path):
            logger.warning(f"Water shapefile not found at {shapefile_path}.")
            return None

        gdf = gpd.read_file(shapefile_path)
        if gdf.empty:
            logger.warning(f"Water shapefile {shapefile_path} contains no geometries.")
            return None

        logger.info(f"Loaded {len(gdf)} water geometry/geometries from {os.path.basename(shapefile_path)}")

        y_coord = fused_datacube["y"].values
        x_coord = fused_datacube["x"].values
        H, W = len(y_coord), len(x_coord)

        target_crs = self._extract_crs(fused_datacube)
        geom_crs = gdf.crs

        if target_crs is None and geom_crs is not None and geom_crs.to_epsg() == 4326:
            target_crs = self._infer_utm_from_geoms(gdf, fused_datacube)
            if target_crs is not None:
                logger.info(f"Inferred target CRS as {target_crs} from water shapefile coordinates")

        if geom_crs is not None and target_crs is not None and geom_crs != target_crs:
            logger.info(f"Reprojecting water geometries from {geom_crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)

        res_y = abs(y_coord[1] - y_coord[0]) if H > 1 else 1.0
        res_x = abs(x_coord[1] - x_coord[0]) if W > 1 else 1.0

        x_min = x_coord.min() - res_x / 2
        x_max = x_coord.max() + res_x / 2
        y_min = y_coord.min() - res_y / 2
        y_max = y_coord.max() + res_y / 2

        transform = from_bounds(x_min, y_min, x_max, y_max, W, H)

        geometries = [(mapping(geom), 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]

        mask = rasterize(
            geometries,
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

        water_pct = np.sum(mask) / mask.size * 100
        logger.info(f"Water mask: {np.sum(mask)} pixels ({water_pct:.1f}%) masked as water")

        return mask.astype(bool)

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
    def _infer_utm_from_geoms(gdf: gpd.GeoDataFrame, ds: xr.Dataset):
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            return None
        geom_bounds = gdf.total_bounds
        center_lon = (geom_bounds[0] + geom_bounds[2]) / 2
        center_lat = (geom_bounds[1] + geom_bounds[3]) / 2
        if not (-180 <= center_lon <= 180 and -80 <= center_lat <= 80):
            return None
        zone = int((center_lon + 180) / 6) + 1
        epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
        try:
            inferred = CRS.from_epsg(epsg)
            cx = float(np.mean(ds["x"].values))
            cy = float(np.mean(ds["y"].values))
            transformer = Transformer.from_crs(inferred, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(cx, cy)
            dist = abs(lon - center_lon) + abs(lat - center_lat)
            if dist < 5.0:
                logger.info(f"Inferred UTM zone {zone} {'N' if center_lat >= 0 else 'S'} from shapefile (lon={center_lon:.2f}, lat={center_lat:.2f}) -> EPSG:{epsg} (validation: lon={lon:.2f}, lat={lat:.2f})")
                return inferred
        except Exception:
            pass
        return None

    @staticmethod
    def _apply_water_mask(fused_datacube: xr.Dataset, water_mask: np.ndarray) -> xr.Dataset:
        masked = fused_datacube.copy(deep=True)
        for var in masked.data_vars:
            masked[var] = masked[var].where(~water_mask)
        return masked

    @staticmethod
    def align_pca_to_reference(
        reference_loadings: np.ndarray,
        current_loadings: np.ndarray,
        current_scores: np.ndarray,
    ):
        """
        Match current PCA components to reference components and align signs.

        Parameters
        ----------
        reference_loadings: (n_components, n_features)
        current_loadings:   (n_components, n_features)
        current_scores:     (n_samples, n_components)

        Returns
        -------
        aligned_loadings, aligned_scores, similarities, order
        """
        signed_similarity = reference_loadings @ current_loadings.T
        matching_similarity = np.abs(signed_similarity)

        reference_idx, current_idx = linear_sum_assignment(-matching_similarity)
        order = current_idx[np.argsort(reference_idx)]

        aligned_loadings = current_loadings[order].copy()
        aligned_scores = current_scores[:, order].copy()

        similarities = np.empty(len(order))
        for component in range(len(order)):
            similarity = np.dot(
                reference_loadings[component],
                aligned_loadings[component],
            )
            if similarity < 0:
                aligned_loadings[component] *= -1
                aligned_scores[:, component] *= -1
                similarity *= -1
            similarities[component] = similarity

        return aligned_loadings, aligned_scores, similarities, order

    def _datewise_pca(self, fused_datacube: xr.Dataset):
        feature_names = list(fused_datacube.data_vars)
        X = xr.concat([fused_datacube[var] for var in feature_names], dim="feature")
        X = X.assign_coords(feature=feature_names)
        X = X.transpose("t", "feature", "y", "x")

        X_pix = X.stack(pix=("y", "x")).transpose("t", "pix", "feature")

        n_components = min(2, len(feature_names))

        # ---- Build reference PCA on pooled, globally-scaled data ----
        A_all = X_pix.values
        valid_all = np.isfinite(A_all).all(axis=2)
        A_pooled = A_all[valid_all]

        if A_pooled.shape[0] == 0:
            logger.warning("No valid pixels across all dates; returning empty results.")
            n_pix = X_pix.sizes["pix"]
            n_t = X_pix.sizes["t"]

            env_datacube = xr.DataArray(
                np.full((n_t, n_pix), np.nan, dtype=np.float32),
                coords={"t": X_pix["t"], "pix": X_pix["pix"]}, dims=("t", "pix"),
            ).unstack("pix").rename("environmental_indicator")

            evr = xr.DataArray(
                np.full((n_t, n_components), np.nan, dtype=np.float32),
                coords={"t": X_pix["t"], "pc": np.arange(n_components)}, dims=("t", "pc"),
            ).rename("explained_variance_ratio")

            components = xr.DataArray(
                np.full((n_t, n_components, len(feature_names)), np.nan, dtype=np.float32),
                coords={"t": X_pix["t"], "pc": np.arange(n_components), "feature": feature_names},
                dims=("t", "pc", "feature"),
            ).rename("pca_components")

            similarities = xr.DataArray(
                np.full((n_t, n_components), np.nan, dtype=np.float32),
                coords={"t": X_pix["t"], "pc": np.arange(n_components)}, dims=("t", "pc"),
            ).rename("pca_similarity")

            return env_datacube, evr, components, similarities

        global_scaler = StandardScaler().fit(A_pooled)
        A_pooled_scaled = global_scaler.transform(A_pooled)
        reference_pca = PCA(n_components=n_components).fit(A_pooled_scaled)
        reference_loadings = reference_pca.components_

        out_maps = []
        evr_list = []
        comp_list = []
        sim_list = []

        for tval in X_pix["t"].values:
            A = X_pix.sel(t=tval).values

            valid_px = np.isfinite(A).all(axis=1)
            A_valid = A[valid_px]

            if A_valid.shape[0] == 0:
                logger.warning(f"Time step {tval}: no valid pixels across all indices, skipping.")
                env_indicator_da = (
                    xr.DataArray(
                        np.full((X_pix.sizes["pix"],), np.nan, dtype=np.float32),
                        coords={"pix": X_pix["pix"]}, dims=("pix",)
                    )
                    .unstack("pix")
                    .rename("environmental_indicator")
                )
                out_maps.append(env_indicator_da)
                evr_list.append(
                    xr.DataArray([np.nan] * n_components, dims=("pc",)).assign_coords(pc=np.arange(n_components))
                )
                comp_list.append(
                    xr.DataArray(
                        np.full((n_components, len(feature_names)), np.nan, dtype=np.float32),
                        dims=("pc", "feature"),
                        coords={"feature": feature_names}
                    )
                )
                sim_list.append(
                    xr.DataArray([np.nan] * n_components, dims=("pc",)).assign_coords(pc=np.arange(n_components))
                )
                continue

            for feat_idx in range(A_valid.shape[1]):
                col = A_valid[:, feat_idx]
                lo, hi = np.percentile(col, [1, 99])
                A_valid[:, feat_idx] = np.clip(col, lo, hi)

            A_valid = global_scaler.transform(A_valid)

            pca = PCA(n_components=min(2, A_valid.shape[1]))
            Z = pca.fit_transform(A_valid)
            eigenvalues = pca.explained_variance_

            aligned_loadings, aligned_scores, sims, order = self.align_pca_to_reference(
                reference_loadings, pca.components_, Z
            )

            aligned_eigenvalues = eigenvalues[order]
            aligned_evr = pca.explained_variance_ratio_[order]

            for pc_idx, sim in enumerate(sims):
                if sim < self.similarity_threshold:
                    logger.warning(
                        f"Time step {tval}: PC{pc_idx + 1} similarity to reference is {sim:.3f} "
                        f"(below threshold {self.similarity_threshold}). "
                        f"Component may have rotated, not just flipped."
                    )

            rsei = aligned_eigenvalues[0] * aligned_scores[:, 0] + (
                aligned_eigenvalues[1] * aligned_scores[:, 1] if aligned_scores.shape[1] > 1 else 0.0
            )
            denom = np.nanmax(rsei) - np.nanmin(rsei)
            mrsei = (rsei - np.nanmin(rsei)) / denom if denom > 0 else np.zeros_like(rsei)

            env_indicator = np.full((X_pix.sizes["pix"],), np.nan, dtype=np.float32)
            env_indicator[valid_px] = mrsei.astype(np.float32)

            env_indicator_da = (
                xr.DataArray(env_indicator, coords={"pix": X_pix["pix"]}, dims=("pix",))
                .unstack("pix")
                .rename("environmental_indicator")
            )
            out_maps.append(env_indicator_da)

            evr_list.append(
                xr.DataArray(aligned_evr, dims=("pc",)).assign_coords(pc=np.arange(len(aligned_evr)))
            )

            comp_list.append(
                xr.DataArray(aligned_loadings, dims=("pc", "feature"),
                             coords={"feature": feature_names})
            )

            sim_list.append(
                xr.DataArray(sims, dims=("pc",)).assign_coords(pc=np.arange(len(sims)))
            )

        env_datacube = xr.concat(out_maps, dim="t").assign_coords(t=X_pix["t"])
        evr = xr.concat(evr_list, dim="t").assign_coords(t=X_pix["t"]).rename("explained_variance_ratio")
        components = xr.concat(comp_list, dim="t").assign_coords(t=X_pix["t"]).rename("pca_components")
        similarities = xr.concat(sim_list, dim="t").assign_coords(t=X_pix["t"]).rename("pca_similarity")

        return env_datacube, evr, components, similarities

    def _build_debug_image(self, result_ds: xr.Dataset) -> Image.Image:
        n_t = result_ds.sizes["t"]

        debug_t = getattr(self, "debug_time_index", 0) or 0
        if debug_t >= n_t:
            debug_t = 0

        feature_names = list(result_ds["pca_components"]["feature"].values)

        fig, ax = plt.subplots(figsize=(8, 8))
        env_data = result_ds["environmental_indicator"].isel(t=debug_t).values
        ax.imshow(env_data, cmap="RdYlGn", vmin=0, vmax=1)
        ax.axis("off")

        t_val = result_ds["t"].values[debug_t]
        try:
            date_str = np.datetime_as_string(t_val, unit="D")
        except Exception:
            date_str = str(t_val)

        evr_vals = result_ds["explained_variance_ratio"].isel(t=debug_t).values
        comp_vals = result_ds["pca_components"].isel(t=debug_t).values
        sim_vals = result_ds["pca_similarity"].isel(t=debug_t).values if "pca_similarity" in result_ds.data_vars else None

        lines = [f"Date: {date_str}"]
        for pc_idx in range(len(evr_vals)):
            if np.isnan(evr_vals[pc_idx]):
                lines.append(f"PC{pc_idx + 1}: N/A")
                continue
            evr_pct = evr_vals[pc_idx] * 100
            loadings = comp_vals[pc_idx]
            top_idx = np.argsort(np.abs(loadings))[::-1][:3]
            loadings_str = ", ".join(
                f"{feature_names[j]} {loadings[j]:+.2f}" for j in top_idx
            )
            sim_str = ""
            if sim_vals is not None and not np.isnan(sim_vals[pc_idx]):
                sim_str = f" [sim={sim_vals[pc_idx]:.2f}]"
            lines.append(f"PC{pc_idx + 1} ({evr_pct:.1f}%): {loadings_str}{sim_str}")

        ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba_buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba_arr = rgba_buf.reshape(h, w, 4)
        rgb_arr = rgba_arr[:, :, :3]
        plt.close(fig)

        return Image.fromarray(rgb_arr, mode="RGB")

    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        if l2_datacube is None:
            raise ValueError("EnvIndicator requires L2 fused datacube from SpectralIndexFusion. "
                             "Please add SpectralIndexFusion to L2 algorithms.")

        print(f"Processing Environmental Indicator for time index {self.time_indices}")

        if not self.time_indices:
            logger.info("No time indices provided. Using all time indices from input.")
            self.time_indices = list(range(l2_datacube.sizes['t']))

        if self.time_indices:
            fused_subset = l2_datacube.isel(t=self.time_indices)
        else:
            fused_subset = l2_datacube

        if self.mask_water_bodies:
            water_mask = self._build_water_mask(fused_subset)
            if water_mask is not None:
                fused_subset = self._apply_water_mask(fused_subset, water_mask)
                logger.info("Water bodies masked using water shapefile.")
            else:
                logger.warning(
                    "mask_water_bodies=True but no water shapefile found. "
                    "Proceeding without water masking."
                )

        env_datacube, evr, components, similarities = self._datewise_pca(fused_subset)

        result_ds = xr.Dataset({
            "environmental_indicator": env_datacube,
            "explained_variance_ratio": evr,
            "pca_components": components,
            "pca_similarity": similarities,
        })

        debug_img = self._build_debug_image(result_ds)

        result = [L3_result(
            debug_image=debug_img,
            algorithm_results=result_ds,
            time_indices=list(self.time_indices),
            result_type="datacube"
        )]

        return result

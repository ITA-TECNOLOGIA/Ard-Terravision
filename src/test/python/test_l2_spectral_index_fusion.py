# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: July 2025
# All rights reserved
# --------------------------------------------------------------------------------

import unittest
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from L2.SpectralIndexFusion.SpectralIndexFusion import SpectralIndexFusion
from L2.L2_Algorithm import L2_output


class _MockL1Input:
    """Minimal L1 input mock for unit testing SpectralIndexFusion in isolation."""

    def __init__(self, datacube: xr.Dataset, time_indices=None):
        self._datacube = datacube
        self.time_indices = time_indices or []

    def get_datacube(self):
        return self._datacube


def _build_synthetic_dataset(
    n_timesteps=24,
    freq="15D",
    start="2023-01-01",
    var_names=None,
    shape=(4, 4),
    seed=42,
):
    """Build a synthetic xarray Dataset with datetime t coordinate."""
    if var_names is None:
        var_names = ["NDVI", "NDWI"]

    dates = pd.date_range(start, periods=n_timesteps, freq=freq)
    rng = np.random.default_rng(seed)

    data_vars = {}
    for name in var_names:
        data = rng.random((n_timesteps, *shape)).astype(np.float32)
        data_vars[name] = (["t", "y", "x"], data)

    ds = xr.Dataset(data_vars, coords={"t": dates})
    return ds


class TestSpectralIndexFusionInit(unittest.TestCase):
    """Test constructor validation."""

    def test_default_no_aggregation(self):
        fusion = SpectralIndexFusion()
        self.assertIsNone(fusion.temporal_aggregation)
        self.assertEqual(fusion.agg_statistic, "mean")

    def test_monthly_mean(self):
        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="mean")
        self.assertEqual(fusion.temporal_aggregation, "monthly")
        self.assertEqual(fusion.agg_statistic, "mean")

    def test_yearly_median(self):
        fusion = SpectralIndexFusion(temporal_aggregation="yearly", agg_statistic="median")
        self.assertEqual(fusion.temporal_aggregation, "yearly")
        self.assertEqual(fusion.agg_statistic, "median")

    def test_invalid_freq_raises(self):
        with self.assertRaises(ValueError) as ctx:
            SpectralIndexFusion(temporal_aggregation="weekly")
        self.assertIn("temporal_aggregation", str(ctx.exception))

    def test_invalid_stat_raises(self):
        with self.assertRaises(ValueError) as ctx:
            SpectralIndexFusion(agg_statistic="var")
        self.assertIn("agg_statistic", str(ctx.exception))

    def test_none_freq_ignores_stat(self):
        fusion = SpectralIndexFusion(temporal_aggregation=None, agg_statistic="median")
        self.assertIsNone(fusion.temporal_aggregation)
        self.assertEqual(fusion.agg_statistic, "median")


class TestSpectralIndexFusionNoAggregation(unittest.TestCase):
    """Test that default behavior (no aggregation) is unchanged."""

    def test_no_aggregation_preserves_timesteps(self):
        ds = _build_synthetic_dataset(n_timesteps=24)
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion()
        result = fusion.process_data(l1_inputs)

        self.assertIsInstance(result, L2_output)
        self.assertEqual(result.datacube.sizes["t"], 24)
        self.assertIsNone(result.processed_band_info["temporal_aggregation"])
        self.assertIsNone(result.processed_band_info["agg_statistic"])
        self.assertEqual(result.processed_band_info["original_num_timesteps"], 24)
        self.assertEqual(result.processed_band_info["aggregated_num_timesteps"], 24)

    def test_all_indices_present(self):
        var_names = ["NDVI", "NDWI", "EVI", "BSI"]
        ds = _build_synthetic_dataset(n_timesteps=10, var_names=var_names)
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion()
        result = fusion.process_data(l1_inputs)

        for name in var_names:
            self.assertIn(name, result.datacube.data_vars)

    def test_dims_preserved(self):
        ds = _build_synthetic_dataset(n_timesteps=5, shape=(8, 6))
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion()
        result = fusion.process_data(l1_inputs)

        self.assertIn("t", result.datacube.dims)
        self.assertIn("y", result.datacube.dims)
        self.assertIn("x", result.datacube.dims)
        self.assertEqual(result.datacube.sizes["y"], 8)
        self.assertEqual(result.datacube.sizes["x"], 6)


class TestSpectralIndexFusionMonthlyAggregation(unittest.TestCase):
    """Test monthly temporal aggregation."""

    def test_monthly_reduces_timesteps(self):
        ds = _build_synthetic_dataset(n_timesteps=48, freq="7D")
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        self.assertLess(result.datacube.sizes["t"], 48)
        self.assertGreater(result.datacube.sizes["t"], 0)

    def test_monthly_correct_count(self):
        dates = pd.date_range("2023-01-01", periods=24, freq="15D")
        data = np.random.rand(24, 2, 2).astype(np.float32)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        self.assertEqual(result.datacube.sizes["t"], 12)

    def test_monthly_mean_correctness(self):
        dates = pd.to_datetime([
            "2023-01-05", "2023-01-20",
            "2023-02-05", "2023-02-20",
        ])
        vals = np.array([2.0, 4.0, 6.0, 10.0], dtype=np.float32)
        data = vals.reshape(4, 1, 1)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        jan_mean = result.datacube["NDVI"].isel(t=0).values
        feb_mean = result.datacube["NDVI"].isel(t=1).values

        self.assertAlmostEqual(jan_mean[0, 0], 3.0, places=5)
        self.assertAlmostEqual(feb_mean[0, 0], 8.0, places=5)

    def test_monthly_median_correctness(self):
        dates = pd.to_datetime([
            "2023-01-05", "2023-01-15", "2023-01-25",
        ])
        vals = np.array([1.0, 5.0, 9.0], dtype=np.float32)
        data = vals.reshape(3, 1, 1)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="median")
        result = fusion.process_data(l1_inputs)

        jan_median = result.datacube["NDVI"].isel(t=0).values
        self.assertAlmostEqual(jan_median[0, 0], 5.0, places=5)

    def test_monthly_min_correctness(self):
        dates = pd.to_datetime([
            "2023-01-05", "2023-01-15", "2023-01-25",
        ])
        vals = np.array([1.0, 5.0, 9.0], dtype=np.float32)
        data = vals.reshape(3, 1, 1)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="min")
        result = fusion.process_data(l1_inputs)

        jan_min = result.datacube["NDVI"].isel(t=0).values
        self.assertAlmostEqual(jan_min[0, 0], 1.0, places=5)

    def test_monthly_max_correctness(self):
        dates = pd.to_datetime([
            "2023-01-05", "2023-01-15", "2023-01-25",
        ])
        vals = np.array([1.0, 5.0, 9.0], dtype=np.float32)
        data = vals.reshape(3, 1, 1)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="max")
        result = fusion.process_data(l1_inputs)

        jan_max = result.datacube["NDVI"].isel(t=0).values
        self.assertAlmostEqual(jan_max[0, 0], 9.0, places=5)

    def test_monthly_metadata(self):
        ds = _build_synthetic_dataset(n_timesteps=24, freq="15D")
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        info = result.processed_band_info
        self.assertEqual(info["temporal_aggregation"], "monthly")
        self.assertEqual(info["agg_statistic"], "mean")
        self.assertEqual(info["original_num_timesteps"], 24)
        self.assertEqual(info["aggregated_num_timesteps"], 12)


class TestSpectralIndexFusionYearlyAggregation(unittest.TestCase):
    """Test yearly temporal aggregation."""

    def test_yearly_correct_count(self):
        dates = pd.date_range("2022-01-01", periods=48, freq="15D")
        data = np.random.rand(48, 2, 2).astype(np.float32)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="yearly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        self.assertEqual(result.datacube.sizes["t"], 2)

    def test_yearly_mean_correctness(self):
        dates = pd.to_datetime([
            "2022-01-15", "2022-07-15",
            "2023-01-15", "2023-07-15",
        ])
        vals = np.array([2.0, 4.0, 6.0, 10.0], dtype=np.float32)
        data = vals.reshape(4, 1, 1)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="yearly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        y1_mean = result.datacube["NDVI"].isel(t=0).values
        y2_mean = result.datacube["NDVI"].isel(t=1).values

        self.assertAlmostEqual(y1_mean[0, 0], 3.0, places=5)
        self.assertAlmostEqual(y2_mean[0, 0], 8.0, places=5)

    def test_yearly_metadata(self):
        ds = _build_synthetic_dataset(n_timesteps=48, freq="15D")
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="yearly", agg_statistic="median")
        result = fusion.process_data(l1_inputs)

        info = result.processed_band_info
        self.assertEqual(info["temporal_aggregation"], "yearly")
        self.assertEqual(info["agg_statistic"], "median")
        self.assertEqual(info["original_num_timesteps"], 48)


class TestSpectralIndexFusionMultiInput(unittest.TestCase):
    """Test aggregation with multiple L1 inputs (fusion + aggregation)."""

    def test_multi_input_monthly_aggregation(self):
        dates = pd.date_range("2023-01-01", periods=24, freq="15D")
        rng = np.random.default_rng(0)

        ds1 = xr.Dataset(
            {"NDVI": (["t", "y", "x"], rng.random((24, 3, 3)).astype(np.float32))},
            coords={"t": dates},
        )
        ds2 = xr.Dataset(
            {"NDWI": (["t", "y", "x"], rng.random((24, 3, 3)).astype(np.float32))},
            coords={"t": dates},
        )

        l1_inputs = [
            _MockL1Input(ds1, time_indices=[]),
            _MockL1Input(ds2, time_indices=[]),
        ]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        self.assertIn("NDVI", result.datacube.data_vars)
        self.assertIn("NDWI", result.datacube.data_vars)
        self.assertEqual(result.datacube.sizes["t"], 12)
        self.assertEqual(result.datacube.sizes["y"], 3)
        self.assertEqual(result.datacube.sizes["x"], 3)


class TestSpectralIndexFusionSkipNA(unittest.TestCase):
    """Test that NaN values are handled correctly (skipna=True)."""

    def test_nan_skipped_in_aggregation(self):
        dates = pd.to_datetime([
            "2023-01-05", "2023-01-15", "2023-01-25",
        ])
        vals = np.array([1.0, np.nan, 9.0], dtype=np.float32)
        data = vals.reshape(3, 1, 1)
        ds = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data)},
            coords={"t": dates},
        )
        l1_inputs = [_MockL1Input(ds, time_indices=[])]

        fusion = SpectralIndexFusion(temporal_aggregation="monthly", agg_statistic="mean")
        result = fusion.process_data(l1_inputs)

        jan_mean = result.datacube["NDVI"].isel(t=0).values
        self.assertAlmostEqual(jan_mean[0, 0], 5.0, places=5)


class TestSpectralIndexFusionTemporalAlignment(unittest.TestCase):
    """Test direct temporal alignment via temporal_alignment_tolerance."""

    def test_tolerance_drops_distant_timesteps(self):
        """When the sparse dataset (reference) has a date with no nearby
        observation in the dense dataset, that timestep should be dropped."""
        # Sparse dataset (S2-like, becomes reference): 3 dates
        dates_sparse = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
        data_sparse = np.random.rand(3, 2, 2).astype(np.float32)
        ds_sparse = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data_sparse)},
            coords={"t": dates_sparse},
        )

        # Dense dataset (S3-like): daily data, but with a gap around Feb 1
        # Has data on Jan 1 (0d from sparse), Jan 30 (2d from Feb 1),
        # but NOT within 3 days of Mar 1 (gap from Feb 5 to Mar 10)
        dates_dense = pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-03",
            "2023-01-30", "2023-01-31",
            "2023-03-10", "2023-03-11",
        ])
        data_dense = np.random.rand(7, 2, 2).astype(np.float32)
        ds_dense = xr.Dataset(
            {"LST": (["t", "y", "x"], data_dense)},
            coords={"t": dates_dense},
        )

        l1_inputs = [
            _MockL1Input(ds_sparse, time_indices=[]),
            _MockL1Input(ds_dense, time_indices=[]),
        ]

        fusion = SpectralIndexFusion(
            temporal_alignment_tolerance="3D", spatial_alignment="none"
        )
        result = fusion.process_data(l1_inputs)

        t_values = pd.to_datetime(result.datacube["t"].values)

        # Jan 1: dense has data on Jan 1 (0d) -> survives
        # Feb 1: nearest dense is Jan 31 (1d) -> survives
        # Mar 1: nearest dense is Mar 10 (9d) -> dropped (>3d tolerance)
        self.assertEqual(len(t_values), 2)
        self.assertIn(pd.Timestamp("2023-01-01"), set(t_values))
        self.assertIn(pd.Timestamp("2023-02-01"), set(t_values))
        self.assertNotIn(pd.Timestamp("2023-03-01"), set(t_values))

    def test_tolerance_keeps_nearby_timesteps(self):
        """Timesteps where both datasets have data within the tolerance window
        should be preserved."""
        # Sparse (reference): 5 dates
        dates_sparse = pd.date_range("2023-01-01", periods=5, freq="5D")
        data_sparse = np.random.rand(5, 2, 2).astype(np.float32)
        ds_sparse = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data_sparse)},
            coords={"t": dates_sparse},
        )

        # Dense: daily data covering all sparse dates within 1-2 days
        dates_dense = pd.date_range("2023-01-02", periods=30, freq="1D")
        data_dense = np.random.rand(30, 2, 2).astype(np.float32)
        ds_dense = xr.Dataset(
            {"LST": (["t", "y", "x"], data_dense)},
            coords={"t": dates_dense},
        )

        l1_inputs = [
            _MockL1Input(ds_sparse, time_indices=[]),
            _MockL1Input(ds_dense, time_indices=[]),
        ]

        fusion = SpectralIndexFusion(
            temporal_alignment_tolerance="3D", spatial_alignment="none"
        )
        result = fusion.process_data(l1_inputs)

        self.assertGreater(result.datacube.sizes["t"], 0)
        self.assertIn("NDVI", result.datacube.data_vars)
        self.assertIn("LST", result.datacube.data_vars)

    def test_output_matches_sparsest_dataset(self):
        """Output should have at most as many timesteps as the sparsest dataset,
        not the densest. No duplication of sparse data."""
        # Dense dataset: 30 daily observations
        dates_dense = pd.date_range("2023-01-01", periods=30, freq="1D")
        data_dense = np.random.rand(30, 2, 2).astype(np.float32)
        ds_dense = xr.Dataset(
            {"LST": (["t", "y", "x"], data_dense)},
            coords={"t": dates_dense},
        )

        # Sparse dataset: only 3 observations
        dates_sparse = pd.to_datetime(["2023-01-05", "2023-01-15", "2023-01-25"])
        data_sparse = np.random.rand(3, 2, 2).astype(np.float32)
        ds_sparse = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data_sparse)},
            coords={"t": dates_sparse},
        )

        l1_inputs = [
            _MockL1Input(ds_dense, time_indices=[]),
            _MockL1Input(ds_sparse, time_indices=[]),
        ]

        fusion = SpectralIndexFusion(
            temporal_alignment_tolerance="3D", spatial_alignment="none"
        )
        result = fusion.process_data(l1_inputs)

        n_output = result.datacube.sizes["t"]
        # Sparse has 3 dates, dense has 30. Output should be <= 3 (sparse is reference).
        self.assertLessEqual(
            n_output, 3,
            f"Expected at most 3 timesteps (sparsest dataset), got {n_output}. "
            f"Data should not be duplicated."
        )
        # All output dates should be the sparse dataset's actual dates
        t_output = pd.to_datetime(result.datacube["t"].values)
        t_sparse = set(dates_sparse)
        for t in t_output:
            self.assertIn(
                t, t_sparse,
                f"Output timestep {t.date()} is not a sparse dataset date. "
                f"Only the sparsest dataset's dates should appear in output."
            )


class TestSpectralIndexFusionSpatialAlignment(unittest.TestCase):
    """Test that _align_spatial skips redundant interpolation when grids match."""

    def test_matching_grid_skips_interpolation(self):
        """Datasets already at the target grid should not be re-interpolated.
        Verifies that identical coordinate arrays are preserved exactly."""
        dates = pd.date_range("2023-01-01", periods=5, freq="5D")

        # Two datasets with the SAME spatial grid
        x_coords = np.linspace(0, 100, 10)
        y_coords = np.linspace(0, 100, 8)

        data_a = np.random.rand(5, 8, 10).astype(np.float32)
        data_b = np.random.rand(5, 8, 10).astype(np.float32)

        ds_a = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data_a)},
            coords={"t": dates, "x": x_coords, "y": y_coords},
        )
        ds_b = xr.Dataset(
            {"BSI": (["t", "y", "x"], data_b)},
            coords={"t": dates, "x": x_coords, "y": y_coords},
        )

        l1_inputs = [
            _MockL1Input(ds_a, time_indices=[]),
            _MockL1Input(ds_b, time_indices=[]),
        ]

        fusion = SpectralIndexFusion(spatial_alignment="upsample")
        result = fusion.process_data(l1_inputs)

        # Both datasets should retain their exact coordinate arrays
        # (not copies from interpolation)
        np.testing.assert_array_equal(
            result.datacube["x"].values, x_coords
        )
        np.testing.assert_array_equal(
            result.datacube["y"].values, y_coords
        )
        self.assertIn("NDVI", result.datacube.data_vars)
        self.assertIn("BSI", result.datacube.data_vars)

    def test_different_grid_still_interpolated(self):
        """Datasets with different spatial grids should still be interpolated
        to the finest grid."""
        # Sparse (reference): 3 dates
        dates_sparse = pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-10"])

        # Fine grid (10x8)
        x_fine = np.linspace(0, 100, 10)
        y_fine = np.linspace(0, 100, 8)
        data_fine = np.random.rand(3, 8, 10).astype(np.float32)
        ds_fine = xr.Dataset(
            {"NDVI": (["t", "y", "x"], data_fine)},
            coords={"t": dates_sparse, "x": x_fine, "y": y_fine},
        )

        # Coarse grid (4x3) with same dates
        x_coarse = np.linspace(0, 100, 4)
        y_coarse = np.linspace(0, 100, 3)
        data_coarse = np.random.rand(3, 3, 4).astype(np.float32)
        ds_coarse = xr.Dataset(
            {"LST": (["t", "y", "x"], data_coarse)},
            coords={"t": dates_sparse, "x": x_coarse, "y": y_coarse},
        )

        l1_inputs = [
            _MockL1Input(ds_fine, time_indices=[]),
            _MockL1Input(ds_coarse, time_indices=[]),
        ]

        fusion = SpectralIndexFusion(
            temporal_alignment_tolerance="3D", spatial_alignment="upsample"
        )
        result = fusion.process_data(l1_inputs)

        # Output should be on the fine grid
        self.assertEqual(result.datacube.sizes["x"], 10)
        self.assertEqual(result.datacube.sizes["y"], 8)
        np.testing.assert_array_equal(
            result.datacube["x"].values, x_fine
        )
        np.testing.assert_array_equal(
            result.datacube["y"].values, y_fine
        )
        self.assertIn("NDVI", result.datacube.data_vars)
        self.assertIn("LST", result.datacube.data_vars)


if __name__ == "__main__":
    unittest.main()

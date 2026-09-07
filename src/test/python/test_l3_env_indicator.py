import unittest
import sys
import os
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig
from L2.L2_Algorithm import L2_output
from L3.L3_Algorithm import L3_result


class TestL3EnvIndicator(unittest.TestCase):
    """Test L3 Environmental Indicator with L2 SpectralIndexFusion."""

    def test_full_pipeline_l1_to_l3_env_indicator(self):
        """Test complete L1 -> L2 (SpectralIndexFusion) -> L3 (EnvIndicator) pipeline."""
        config_path = "pipelines/env_indicator_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        self.assertIsNotNone(l1_data)

        l2_output = pipeline.run_l2(l1_data)
        self.assertIsNotNone(l2_output)
        self.assertIsInstance(l2_output, L2_output)
        self.assertIsNotNone(l2_output.datacube)

        fused_vars = list(l2_output.datacube.data_vars)
        expected_indices = ["NDVI", "NDWI", "EVI", "BSI", "NDCI", "NDDI", "NDTI", "AMWI"]
        for idx in expected_indices:
            self.assertIn(idx, fused_vars, f"Missing {idx} in fused datacube")

        l3_results = pipeline.run_l3(l1_data, l2_output)
        self.assertIsNotNone(l3_results)
        self.assertIsInstance(l3_results, list)
        self.assertGreater(len(l3_results), 0)

        result = l3_results[0]
        self.assertIsInstance(result, L3_result)
        self.assertEqual(result.result_type, "datacube")

        result_ds = result.algorithm_results
        self.assertIsInstance(result_ds, xr.Dataset)

        self.assertIn("environmental_indicator", result_ds.data_vars)
        self.assertIn("explained_variance_ratio", result_ds.data_vars)
        self.assertIn("pca_components", result_ds.data_vars)
        self.assertIn("pca_similarity", result_ds.data_vars)

        env_datacube = result_ds["environmental_indicator"]
        self.assertIn("t", env_datacube.dims)
        self.assertIn("y", env_datacube.dims)
        self.assertIn("x", env_datacube.dims)

        evr = result_ds["explained_variance_ratio"]
        self.assertIn("t", evr.dims)
        self.assertIn("pc", evr.dims)

        components = result_ds["pca_components"]
        self.assertIn("t", components.dims)
        self.assertIn("pc", components.dims)
        self.assertIn("feature", components.dims)

        for t_idx in range(evr.sizes["t"]):
            evr_vals = evr.isel(t=t_idx).values
            if not np.isnan(evr_vals).all():
                self.assertGreater(np.nansum(evr_vals), 0.0)
                self.assertLessEqual(np.nansum(evr_vals), 1.0 + 1e-6)

    def test_env_indicator_requires_l2_fusion(self):
        """Test that EnvIndicator raises error when L2 fusion is not provided."""
        config_path = "pipelines/env_indicator_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()

        l3_algorithm = pipeline.l3_algorithms[0]
        with self.assertRaises(ValueError) as context:
            l3_algorithm.process_data(l1_data, l2_datacube=None)

        self.assertIn("SpectralIndexFusion", str(context.exception))

    def test_fused_datacube_shape(self):
        """Test that fused datacube has correct dimensions."""
        config_path = "pipelines/env_indicator_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)

        self.assertIn("t", l2_output.datacube.dims)
        self.assertIn("y", l2_output.datacube.dims)
        self.assertIn("x", l2_output.datacube.dims)

    def test_pca_sign_consistency(self):
        """Test that PC1 loadings are consistently oriented across dates."""
        config_path = "pipelines/env_indicator_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        result_ds = l3_results[0].algorithm_results
        components = result_ds["pca_components"].values

        n_t = components.shape[0]
        for t_idx in range(n_t - 1):
            pc1_a = components[t_idx, 0, :]
            pc1_b = components[t_idx + 1, 0, :]
            if np.isnan(pc1_a).any() or np.isnan(pc1_b).any():
                continue
            dot = np.dot(pc1_a, pc1_b)
            self.assertGreater(dot, 0, f"PC1 sign flip between t={t_idx} and t={t_idx + 1}")

    def test_pca_similarity_shape(self):
        """Test that pca_similarity has correct dimensions."""
        config_path = "pipelines/env_indicator_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        result_ds = l3_results[0].algorithm_results
        similarities = result_ds["pca_similarity"]
        self.assertIn("t", similarities.dims)
        self.assertIn("pc", similarities.dims)


if __name__ == "__main__":
    unittest.main()

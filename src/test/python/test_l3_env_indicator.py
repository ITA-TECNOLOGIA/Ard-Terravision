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

        env_datacube = result.algorithm_results
        self.assertIsInstance(env_datacube, xr.DataArray)
        self.assertIn("t", env_datacube.dims)
        self.assertIn("y", env_datacube.dims)
        self.assertIn("x", env_datacube.dims)

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


if __name__ == "__main__":
    unittest.main()
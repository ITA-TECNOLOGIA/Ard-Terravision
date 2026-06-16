import unittest
import sys
import os
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig
from L3.L3_Algorithm import L3_result


class TestL3ChangeDetection(unittest.TestCase):
    """Test L3 ChangeDetection algorithm through the full pipeline."""

    def _find_cd_result(self, results):
        for result in results:
            ar = result.algorithm_results
            if isinstance(ar, xr.DataArray) and ar.attrs.get("result_type") == "change_map":
                return result
        return None

    def test_change_detection_runs_successfully(self):
        """Test that ChangeDetection processes data and returns valid results."""
        config_path = "pipelines/satellite_example_canteras_change_detection.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        self.assertIsNotNone(l1_data)

        l2_output = pipeline.run_l2(l1_data)

        l3_results = pipeline.run_l3(l1_data, l2_output)
        self.assertIsNotNone(l3_results)
        self.assertIsInstance(l3_results, list)

    def test_change_detection_result_type(self):
        """Test that change detection result has correct result_type."""
        config_path = "pipelines/satellite_example_canteras_change_detection.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        cd_result = self._find_cd_result(l3_results)
        self.assertIsNotNone(cd_result, "Should have a result with result_type='change_map'")
        self.assertEqual(cd_result.result_type, "datacube")

    def test_change_map_has_valid_dimensions(self):
        """Test that change map has valid 2D numpy array."""
        config_path = "pipelines/satellite_example_canteras_change_detection.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        cd_result = self._find_cd_result(l3_results)
        self.assertIsNotNone(cd_result)
        change_map = cd_result.algorithm_results.values
        self.assertIsNotNone(change_map, "Change map should exist")
        self.assertIsInstance(change_map, np.ndarray)
        self.assertEqual(len(change_map.shape), 2)
        self.assertTrue(change_map.shape[0] > 0 and change_map.shape[1] > 0)

    def test_change_detection_time_indices(self):
        """Test that time indices A and B are recorded correctly."""
        config_path = "pipelines/satellite_example_canteras_change_detection.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        cd_result = self._find_cd_result(l3_results)
        self.assertIsNotNone(cd_result)
        attrs = cd_result.algorithm_results.attrs
        time_index_A = attrs.get("time_index_A")
        time_index_B = attrs.get("time_index_B")
        self.assertIsNotNone(time_index_A)
        self.assertIsNotNone(time_index_B)
        self.assertNotEqual(time_index_A, time_index_B, "Time indices A and B should be different")


if __name__ == "__main__":
    unittest.main()

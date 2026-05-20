import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig
from L3.L3_Algorithm import L3_result


class TestL3LulcClassification(unittest.TestCase):
    """Test L3 LulcClassification algorithm through the full pipeline."""

    def test_lulc_classification_runs_successfully(self):
        """Test that LULC classification processes data and returns valid results."""
        config_path = "pipelines/lulc_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        # Run L1
        l1_data = pipeline.run_l1()
        self.assertIsNotNone(l1_data)

        # Run L2 (empty in this config)
        l2_output = pipeline.run_l2(l1_data)
        self.assertIsNone(l2_output)  # No L2 algorithms in this config

        # Run L3 (LulcClassification)
        l3_results = pipeline.run_l3(l1_data, l2_output)
        self.assertIsNotNone(l3_results, "L3 results should not be None")
        self.assertIsInstance(l3_results, list, "L3 results should be a list")
        self.assertTrue(len(l3_results) > 0, "L3 results should not be empty")

    def test_lulc_result_type_is_mask(self):
        """Test that LULC result has correct result_type."""
        config_path = "pipelines/lulc_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        # Find the LULC result (not ObjectDetection)
        lulc_result = None
        for result in l3_results:
            if result.result_type == "mask":
                lulc_result = result
                break

        self.assertIsNotNone(lulc_result, "Should have a result with result_type='mask'")
        self.assertEqual(lulc_result.result_type, "mask")

    def test_lulc_mask_has_valid_dimensions(self):
        """Test that LULC mask has valid 2D numpy array."""
        config_path = "pipelines/lulc_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        # Find LULC result
        lulc_result = None
        for result in l3_results:
            if result.result_type == "mask":
                lulc_result = result
                break

        self.assertIsNotNone(lulc_result)
        mask = lulc_result.algorithm_results.get("mask")
        self.assertIsNotNone(mask, "Mask should exist in algorithm_results")
        self.assertIsInstance(mask, np.ndarray, "Mask should be numpy array")
        self.assertEqual(len(mask.shape), 2, "Mask should be 2D array")
        self.assertTrue(mask.shape[0] > 0 and mask.shape[1] > 0, "Mask should have positive dimensions")

    def test_lulc_debug_image_is_valid(self):
        """Test that debug image is a valid PIL Image."""
        config_path = "pipelines/lulc_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        # Find LULC result
        lulc_result = None
        for result in l3_results:
            if result.result_type == "mask":
                lulc_result = result
                break

        self.assertIsNotNone(lulc_result.debug_image, "Debug image should not be None")
        self.assertTrue(hasattr(lulc_result.debug_image, 'size'), "Debug image should have size attribute")


if __name__ == "__main__":
    unittest.main()
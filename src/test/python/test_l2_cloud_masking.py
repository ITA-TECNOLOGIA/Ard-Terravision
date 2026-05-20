import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig
from L2.L2_Algorithm import L2_output


class TestL2CloudMasking(unittest.TestCase):
    """Test L2 CloudMasking algorithm through the full pipeline."""

    def test_cloud_masking_runs_successfully(self):
        """Test that CloudMasking processes data and returns valid L2_output."""
        config_path = "pipelines/cloud_masking_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        # Run L1
        l1_data = pipeline.run_l1()
        self.assertIsNotNone(l1_data, "L1 input should not be None")

        # Run L2 (CloudMasking)
        l2_output = pipeline.run_l2(l1_data)
        self.assertIsNotNone(l2_output, "L2 output should not be None")
        self.assertIsInstance(l2_output, L2_output, "L2 output should be L2_output instance")

        # Assert L2_output fields
        self.assertIsNotNone(l2_output.datacube, "Datacube should not be None")
        self.assertIsNotNone(l2_output.debug_image, "Debug image should not be None")
        self.assertIsNotNone(l2_output.processed_band_info, "Processed band info should not be None")

        # Assert processed band info content
        self.assertEqual(l2_output.processed_band_info.get("algorithm"), "CloudMasking")
        self.assertIsInstance(l2_output.processed_band_info.get("processed_band_names"), list)
        self.assertTrue(len(l2_output.processed_band_info.get("processed_band_names", [])) > 0)

    def test_cloud_masked_bands_exist_in_datacube(self):
        """Test that cloud-masked bands (_cm suffix) are added to datacube."""
        config_path = "pipelines/cloud_masking_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)

        # Check that processed_band_names contain bands with _cm suffix
        processed_bands = l2_output.processed_band_info.get("processed_band_names", [])
        cm_bands = [b for b in processed_bands if b.endswith("_cm")]
        self.assertTrue(len(cm_bands) > 0, "Should have cloud-masked bands with _cm suffix")

        # Verify these bands exist in the datacube
        for band in cm_bands:
            self.assertIn(band, l2_output.datacube, f"Band {band} should exist in datacube")


if __name__ == "__main__":
    unittest.main()
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig
from L3.L3_Algorithm import L3_result


class TestL3ObjectDetection(unittest.TestCase):
    """Test L3 ObjectDetectionGroundedSAM2 algorithm through the full pipeline."""

    def test_object_detection_runs_successfully(self):
        """Test that ObjectDetection processes data and returns valid results."""
        config_path = "pipelines/object_detection_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        # Run L1
        l1_data = pipeline.run_l1()
        self.assertIsNotNone(l1_data)

        # Run L2 (empty)
        l2_output = pipeline.run_l2(l1_data)

        # Run L3
        l3_results = pipeline.run_l3(l1_data, l2_output)
        self.assertIsNotNone(l3_results)
        self.assertIsInstance(l3_results, list)

    def test_object_detection_result_type(self):
        """Test that object detection result has correct result_type."""
        config_path = "pipelines/object_detection_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        # Find ObjectDetection result
        od_result = None
        for result in l3_results:
            if result.result_type == "detections":
                od_result = result
                break

        self.assertIsNotNone(od_result, "Should have a result with result_type='detections'")
        self.assertEqual(od_result.result_type, "detections")

    def test_object_detection_has_detections(self):
        """Test that object detection returns detection objects."""
        config_path = "pipelines/object_detection_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        # Find ObjectDetection result
        od_result = None
        for result in l3_results:
            if result.result_type == "detections":
                od_result = result
                break

        self.assertIsNotNone(od_result)
        algo_results = od_result.algorithm_results
        self.assertIsNotNone(algo_results)

        # Check for detections attribute (FrameResult has .detections)
        if hasattr(algo_results, 'detections'):
            detections = algo_results.detections
            self.assertIsInstance(detections, list)

    def test_object_detection_debug_image(self):
        """Test that debug image is a valid PIL Image."""
        config_path = "pipelines/object_detection_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1_data = pipeline.run_l1()
        l2_output = pipeline.run_l2(l1_data)
        l3_results = pipeline.run_l3(l1_data, l2_output)

        # Find ObjectDetection result
        od_result = None
        for result in l3_results:
            if result.result_type == "detections":
                od_result = result
                break

        self.assertIsNotNone(od_result)
        self.assertIsNotNone(od_result.debug_image, "Debug image should not be None")
        self.assertTrue(hasattr(od_result.debug_image, 'size'))


if __name__ == "__main__":
    unittest.main()
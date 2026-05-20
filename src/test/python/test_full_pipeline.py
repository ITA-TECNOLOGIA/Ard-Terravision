import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig
from L2.L2_Algorithm import L2_output
from L3.L3_Algorithm import L3_result


class TestFullPipeline(unittest.TestCase):
    """Test full pipeline integration from L1 through L4."""

    def test_full_pipeline_l1_to_l4(self):
        """Test complete L1 -> L2 -> L3 -> L4 pipeline."""
        config_path = "pipelines/satellite_example_canteras.json"
        pipeline = PipelineConfig.from_json(config_path)

        # Run L1
        l1_data = pipeline.run_l1()
        self.assertIsNotNone(l1_data, "L1 should return data")

        # Run L2
        l2_output = pipeline.run_l2(l1_data)
        if pipeline.l2_algorithms:
            self.assertIsNotNone(l2_output)
            self.assertIsInstance(l2_output, L2_output)

        # Run L3
        l3_results = pipeline.run_l3(l1_data, l2_output)
        if pipeline.l3_algorithms:
            self.assertIsNotNone(l3_results)
            self.assertIsInstance(l3_results, list)

        # Run L4
        if pipeline.l4_algorithm:
            target_time_index = getattr(pipeline.l4_algorithm, 'target_time_index', None)
            l4_result = pipeline.run_l4(l1_data, l3_results, target_time_index)
            self.assertIsNotNone(l4_result)

        # Run full pipeline
        result = pipeline.run()
        self.assertIsNotNone(result)

    def test_pipeline_repr(self):
        """Test that pipeline __repr__ works correctly."""
        config_path = "pipelines/satellite_example_canteras.json"
        pipeline = PipelineConfig.from_json(config_path)

        repr_str = repr(pipeline)
        self.assertIn("PipelineConfig", repr_str)
        self.assertIn("Satellite", repr_str)

    def test_pipeline_from_json_loads_all_layers(self):
        """Test that all pipeline layers are loaded from JSON."""
        config_path = "pipelines/satellite_example_canteras.json"
        pipeline = PipelineConfig.from_json(config_path)

        self.assertIsNotNone(pipeline.l1_input)
        self.assertIsInstance(pipeline.l2_algorithms, list)
        self.assertIsInstance(pipeline.l3_algorithms, list)
        self.assertIsNotNone(pipeline.l4_algorithm)

    def test_pipeline_empty_l2_l3(self):
        """Test pipeline with empty L2 and L3 (just L1 -> L4)."""
        config_path = "pipelines/qwen_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1 = pipeline.run_l1()
        self.assertIsNotNone(l1)

        l2 = pipeline.run_l2(l1)
        self.assertIsNone(l2)

        l3 = pipeline.run_l3(l1, l2)
        self.assertIsInstance(l3, list)
        self.assertEqual(len(l3), 0)

        result = pipeline.run()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
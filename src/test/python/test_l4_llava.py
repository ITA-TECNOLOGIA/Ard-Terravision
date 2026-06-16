import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig


class TestL4LLaVA(unittest.TestCase):
    """Test L4 LLaVACustom algorithm through the full pipeline."""

    def test_llava_runs_successfully(self):
        """Test that LLaVA processes data and returns valid responses."""
        config_path = "pipelines/llava_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        result = pipeline.run()

        self.assertIsNotNone(result, "L4 result should not be None")
        self.assertIsInstance(result, list, "L4 result should be a list of strings")
        self.assertTrue(len(result) > 0, "L4 result should not be empty")

    def test_llava_returns_strings(self):
        """Test that LLaVA returns list of strings."""
        config_path = "pipelines/llava_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        result = pipeline.run()

        for idx, response in enumerate(result):
            self.assertIsInstance(response, str, f"Response {idx} should be a string")
            self.assertTrue(len(response) > 0, f"Response {idx} should not be empty")

    def test_llava_with_l3_context(self):
        """Test LLaVA with L3 results as context."""
        config_path = "pipelines/llava_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        l1 = pipeline.run_l1()
        l2 = pipeline.run_l2(l1)
        l3_results = pipeline.run_l3(l1, l2)

        l4_result = pipeline.run_l4(l1, l3_results)

        self.assertIsNotNone(l4_result)
        self.assertIsInstance(l4_result, list)


if __name__ == "__main__":
    unittest.main()
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig


class TestL4Qwen(unittest.TestCase):
    """Test L4 QwenCustom algorithm through the full pipeline."""

    def test_qwen_runs_successfully(self):
        """Test that Qwen processes data and returns valid responses."""
        config_path = "pipelines/qwen_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        result = pipeline.run()

        self.assertIsNotNone(result, "L4 result should not be None")
        self.assertIsInstance(result, list, "L4 result should be a list of strings")
        self.assertTrue(len(result) > 0, "L4 result should not be empty")

    def test_qwen_returns_strings(self):
        """Test that Qwen returns list of strings."""
        config_path = "pipelines/qwen_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        result = pipeline.run()

        for idx, response in enumerate(result):
            self.assertIsInstance(response, str, f"Response {idx} should be a string")
            self.assertTrue(len(response) > 0, f"Response {idx} should not be empty")

    def test_qwen_with_l3_context(self):
        """Test Qwen with L3 results as context."""
        config_path = "pipelines/lulc_example.json"
        pipeline = PipelineConfig.from_json(config_path)

        from L4.QwenCustom.QwenCustom import QwenCustom
        pipeline.l4_algorithm = QwenCustom(
            args_list=[{"time_index": 0}],
            base_model="Qwen/Qwen2.5-VL-7B-Instruct",
            prompt="Describe the content of the image.",
            system_prompt="You are a helpful assistant."
        )

        result = pipeline.run()

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
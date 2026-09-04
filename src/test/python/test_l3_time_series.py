import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from PipelineConfig import PipelineConfig
from L2.L2_Algorithm import L2_output
from L3.L3_Algorithm import L3_result

CANTERAS_ATMOS = "pipelines/timeseries_anomaly_canteras_atmos.json"
CANTERAS_TILT = "pipelines/timeseries_anomaly_canteras_tilt.json"
THARSIS_TILT = "pipelines/timeseries_anomaly_tharsis_tilt.json"
TERNAMAG_RADAR = "pipelines/timeseries_anomaly_ternamag_radar.json"


def _run_pipeline(config_path):
    pipeline = PipelineConfig.from_json(config_path)
    l1_data = pipeline.run_l1()
    l2_output = pipeline.run_l2(l1_data)
    l3_results = pipeline.run_l3(l1_data, l2_output)
    return pipeline, l1_data, l2_output, l3_results


class TestL3TimeSeriesAnomalyCanterasAtmos(unittest.TestCase):
    """Test L3 TimeSeriesAnomalyDetection — Canteras Atmósfera."""

    def test_runs_successfully(self):
        _, l1_data, l2_output, l3_results = _run_pipeline(CANTERAS_ATMOS)
        self.assertIsNotNone(l1_data)
        self.assertIsInstance(l2_output, L2_output)
        self.assertIsNotNone(l2_output.datacube)
        self.assertEqual(len(l3_results), 1)
        self.assertIsInstance(l3_results[0], L3_result)

    def test_valid_result(self):
        _, _, _, l3_results = _run_pipeline(CANTERAS_ATMOS)
        result = l3_results[0]
        self.assertIsInstance(result.algorithm_results, dict)
        self.assertNotIn("error", result.algorithm_results)

    def test_algorithm_results_structure(self):
        _, _, _, l3_results = _run_pipeline(CANTERAS_ATMOS)
        ar = l3_results[0].algorithm_results

        self.assertIsInstance(ar["features"], list)
        self.assertTrue(len(ar["features"]) > 0)
        self.assertIsInstance(ar["contamination"], float)
        self.assertIsInstance(ar["n_estimators"], int)
        self.assertGreater(ar["n_estimators"], 0)
        self.assertIsInstance(ar["zscore_detection_enabled"], bool)
        self.assertIsInstance(ar["view_var"], (str, list))
        self.assertIsInstance(ar["zscore_variables"], list)
        self.assertIsInstance(ar["theta"], (float, int, dict))
        self.assertIsInstance(ar["n_zscore_anomalies"], int)
        self.assertIsInstance(ar["n_iforest_panels"], int)

    def test_debug_image_is_valid(self):
        _, _, _, l3_results = _run_pipeline(CANTERAS_ATMOS)
        result = l3_results[0]
        self.assertIsNotNone(result.debug_image)
        self.assertTrue(hasattr(result.debug_image, 'size'))


class TestL3TimeSeriesAnomalyCanterasTilt(unittest.TestCase):
    """Test L3 TimeSeriesAnomalyDetection — Canteras Tilt."""

    def test_runs_successfully(self):
        _, l1_data, l2_output, l3_results = _run_pipeline(CANTERAS_TILT)
        self.assertIsNotNone(l1_data)
        self.assertIsInstance(l2_output, L2_output)
        self.assertIsNotNone(l2_output.datacube)
        self.assertEqual(len(l3_results), 1)
        self.assertIsInstance(l3_results[0], L3_result)

    def test_valid_result(self):
        _, _, _, l3_results = _run_pipeline(CANTERAS_TILT)
        result = l3_results[0]
        self.assertIsInstance(result.algorithm_results, dict)
        self.assertNotIn("error", result.algorithm_results)

    def test_algorithm_results_structure(self):
        _, _, _, l3_results = _run_pipeline(CANTERAS_TILT)
        ar = l3_results[0].algorithm_results

        self.assertIsInstance(ar["features"], list)
        self.assertTrue(len(ar["features"]) > 0)
        self.assertIsInstance(ar["contamination"], float)
        self.assertIsInstance(ar["n_estimators"], int)
        self.assertGreater(ar["n_estimators"], 0)
        self.assertIsInstance(ar["zscore_detection_enabled"], bool)
        self.assertIsInstance(ar["view_var"], (str, list))
        self.assertIsInstance(ar["zscore_variables"], list)
        self.assertIsInstance(ar["theta"], (float, int))
        self.assertIsInstance(ar["n_zscore_anomalies"], int)
        self.assertIsInstance(ar["n_iforest_panels"], int)

    def test_debug_image_is_valid(self):
        _, _, _, l3_results = _run_pipeline(CANTERAS_TILT)
        result = l3_results[0]
        self.assertIsNotNone(result.debug_image)
        self.assertTrue(hasattr(result.debug_image, 'size'))


class TestL3TimeSeriesAnomalyTharsisTilt(unittest.TestCase):
    """Test L3 TimeSeriesAnomalyDetection — Tharsis Tilt."""

    def test_runs_successfully(self):
        _, l1_data, l2_output, l3_results = _run_pipeline(THARSIS_TILT)
        self.assertIsNotNone(l1_data)
        self.assertIsInstance(l2_output, L2_output)
        self.assertIsNotNone(l2_output.datacube)
        self.assertEqual(len(l3_results), 1)
        self.assertIsInstance(l3_results[0], L3_result)

    def test_valid_result(self):
        _, _, _, l3_results = _run_pipeline(THARSIS_TILT)
        result = l3_results[0]
        self.assertIsInstance(result.algorithm_results, dict)
        self.assertNotIn("error", result.algorithm_results)

    def test_algorithm_results_structure(self):
        _, _, _, l3_results = _run_pipeline(THARSIS_TILT)
        ar = l3_results[0].algorithm_results

        self.assertIsInstance(ar["features"], list)
        self.assertTrue(len(ar["features"]) > 0)
        self.assertIsInstance(ar["contamination"], float)
        self.assertIsInstance(ar["n_estimators"], int)
        self.assertGreater(ar["n_estimators"], 0)
        self.assertIsInstance(ar["zscore_detection_enabled"], bool)
        self.assertIsInstance(ar["view_var"], (str, list))
        self.assertIsInstance(ar["zscore_variables"], list)
        self.assertIsInstance(ar["theta"], (float, int))
        self.assertIsInstance(ar["n_zscore_anomalies"], int)
        self.assertIsInstance(ar["n_iforest_panels"], int)

    def test_debug_image_is_valid(self):
        _, _, _, l3_results = _run_pipeline(THARSIS_TILT)
        result = l3_results[0]
        self.assertIsNotNone(result.debug_image)
        self.assertTrue(hasattr(result.debug_image, 'size'))


class TestL3TimeSeriesAnomalyTernamagRadar(unittest.TestCase):
    """Test L3 TimeSeriesAnomalyDetection — Ternamag Radar."""

    def test_runs_successfully(self):
        _, l1_data, l2_output, l3_results = _run_pipeline(TERNAMAG_RADAR)
        self.assertIsNotNone(l1_data)
        self.assertIsInstance(l2_output, L2_output)
        self.assertIsNotNone(l2_output.datacube)
        self.assertEqual(len(l3_results), 1)
        self.assertIsInstance(l3_results[0], L3_result)

    def test_valid_result(self):
        _, _, _, l3_results = _run_pipeline(TERNAMAG_RADAR)
        result = l3_results[0]
        self.assertIsInstance(result.algorithm_results, dict)
        self.assertNotIn("error", result.algorithm_results)

    def test_algorithm_results_structure(self):
        _, _, _, l3_results = _run_pipeline(TERNAMAG_RADAR)
        ar = l3_results[0].algorithm_results

        self.assertIsInstance(ar["features"], list)
        self.assertTrue(len(ar["features"]) > 0)
        self.assertIsInstance(ar["contamination"], float)
        self.assertIsInstance(ar["n_estimators"], int)
        self.assertGreater(ar["n_estimators"], 0)
        self.assertIsInstance(ar["zscore_detection_enabled"], bool)
        self.assertIsInstance(ar["view_var"], (str, list))
        self.assertIsInstance(ar["zscore_variables"], list)
        self.assertIsInstance(ar["theta"], (float, int))
        self.assertIsInstance(ar["n_zscore_anomalies"], int)
        self.assertIsInstance(ar["n_iforest_panels"], int)

    def test_debug_image_is_valid(self):
        _, _, _, l3_results = _run_pipeline(TERNAMAG_RADAR)
        result = l3_results[0]
        self.assertIsNotNone(result.debug_image)
        self.assertTrue(hasattr(result.debug_image, 'size'))


if __name__ == "__main__":
    unittest.main()

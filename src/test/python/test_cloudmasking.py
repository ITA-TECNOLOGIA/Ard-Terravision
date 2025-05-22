# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import unittest
import numpy as np
import sys
sys.path.append("src/main/python")

from PipelineConfig import PipelineConfig

class TestCloudMasking(unittest.TestCase):
    def setUp(self):
        # runs before every test method
        self.pipeline = PipelineConfig.from_json("src/test/python/pipelines/cloudmasking.json")
        self.input = self.pipeline.l1_input

    def test_cloudmasking(self):
        # Since there's only one L2 algorithm, grab its output directly:
        cloud_masking_results = self.pipeline.l2_algorithms[0].process_data(self.input)

        # We expect at least one result (one per time‐index / band combo)
        self.assertTrue(len(cloud_masking_results) > 0, "CloudMasking returned no results")

        for idx, l2_res in enumerate(cloud_masking_results):
            cm = l2_res.algorithm_results

            # 1) Shape consistency
            self.assertEqual(cm.image_with_clouds_resized.shape, cm.ground_truth_resized.shape,
                             f"Shapes of cloudy input and GT differ at result #{idx}")
            self.assertEqual(cm.ground_truth_resized.shape[:2], cm.cloud_mask_resized.shape,
                             f"Mask spatial dims mismatch at result #{idx}")

            # 2) GT vs inpaint similarity (small MSE)
            mse = np.mean((cm.ground_truth_resized - cm.inpainted_result) ** 2)
            self.assertLess(mse, 0.02, # I made up this value based on previous experience
                            f"High MSE ({mse:.6f}) between GT and inpainted at result #{idx}")

            # 3) Masked pixels actually got modified
            mask = cm.cloud_mask_resized.astype(bool)
            diff = np.abs(cm.inpainted_result - cm.image_with_clouds_resized)
            self.assertTrue(np.any(diff[~mask] > 1e-6),
                            f"No masked pixels were changed at result #{idx}")

            # 4) Unmasked pixels remain (nearly) identical
            if np.any(~mask):
                max_change = diff[mask].max()
                self.assertLess(max_change, 0.11,
                                f"Unmasked pixels changed by up to {max_change:.6f} at result #{idx}")

# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

from L3.L3_Algorithm import L3_Algorithm
# Detrex imports
from L3.ObjectDetection.detrex.demo.demo import setup, instantiate, DetectionCheckpointer, VisualizationDemo
import argparse
import numpy as np
import json
import os
import sys 

detrex_root = os.path.abspath(
    "src/main/python/L3/ObjectDetection/detrex"
)
if detrex_root not in sys.path:
    sys.path.insert(0, detrex_root)

class Detection:
    def __init__(self, bbox, confidence, class_id):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id


class FrameResult:
    def __init__(self, detections, kwargs=None):
        self.detections = detections
        self.sam_scores = None  # Detrex does not produce SAM scores
        self.kwargs = kwargs or {}
        self.florence_raw_text = None
        self.florence_parsed_output = None


class DetrexOutput:
    def __init__(self, frame_results):
        self.algorithm_results = frame_results
        self.debug_image = None  # optional
        
class ObjectDetectionDetrex(L3_Algorithm):
    def __init__(self,
                 time_indices: list[int],
                 rgb_band_names: list[list[str]]):
        super().__init__()

        config_file = "src/main/python/L3/ObjectDetection/detrex/projects/deta/configs/deta_swin_large_finetune_24ep.py"
        opts = ['train.init_checkpoint=./checkpoints/ObjectDetection/converted_deta_swin_o365_finetune.pth']
        confidence_threshold = 0.5

        self.time_indices = time_indices
        self.rgb_band_names = rgb_band_names

        args = argparse.Namespace(
            config_file=config_file,
            webcam=False,
            video_input=None,
            input=['./detrex/idea.jpg'],
            output='./demo_output.jpg',
            min_size_test=800,
            max_size_test=1333,
            img_format='RGB',
            metadata_dataset='coco_2017_val',
            confidence_threshold=confidence_threshold,
            opts=opts
        )

        cfg = setup(args)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.train.init_checkpoint)
        model.eval()

        self.demo = VisualizationDemo(
            model=model,
            min_size_test=args.min_size_test,
            max_size_test=args.max_size_test,
            img_format=args.img_format,
            metadata_dataset=args.metadata_dataset,
        )

        self.model = model
        self.confidence_threshold = confidence_threshold

    def process_data(self, input):
        frame_results = []

        for time_index in self.time_indices:
            for band_name in self.rgb_band_names:
                print(f"Processing object detection for time index {time_index} and band {band_name}")

                frame = input.get_image(time_index, band_name)
                frame = frame.transpose(1, 2, 0).copy()
                frame = np.nan_to_num(frame, nan=0)

                # OPTIONAL normalization (recommended)
                if frame.max() > 0:
                    frame = frame / frame.max()
                frame = (frame * 255).astype(np.uint8)

                predictions, _vis = self.demo.run_on_image(frame, self.confidence_threshold)
                instances = predictions["instances"].to("cpu")

                boxes = instances.pred_boxes.tensor.detach().numpy()
                confidences = instances.scores.detach().numpy()
                class_ids = np.array(instances.pred_classes.tolist())

                detections = []

                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    if confidence < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = box
                    detection = Detection(
                        bbox={
                            "x": int(x1),
                            "y": int(y1),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1)
                        },
                        confidence=float(confidence),
                        class_id=int(class_id)
                    )
                    detections.append(detection)

                frame_results.append(
                    FrameResult(
                        detections=detections,
                        kwargs={
                            "time_index": time_index,
                            "band_name": band_name
                        }
                    )
                )

        return DetrexOutput(frame_results)

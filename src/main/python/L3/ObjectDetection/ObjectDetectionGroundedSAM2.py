# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import os
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from PIL import Image
import cv2
import supervision as sv
import xarray as xr

from L3.L3_Algorithm import L3_Algorithm, L3_result
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, Florence2ForConditionalGeneration

@dataclass
class Detection:
    bbox: Dict[str, int]
    class_id: str
    confidence: float = None

@dataclass
class FrameResult:
    kwargs: Dict[str, Any]
    input_image: Image.Image
    annotated_boxes_image: Image.Image
    annotated_mask_image: Image.Image
    detections: List[Detection]
    florence_raw_text: str = ""
    florence_parsed_output: Dict[str, Any] = None
    sam_scores: List[float] = None

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None
    assert processor is not None

    prompt = task_prompt + (text_input or "")
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        pixel_values=inputs["pixel_values"].to(model.device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    parsed_output = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return {
        "raw_text": generated_text,
        "parsed_output": parsed_output
    }
    
class ObjectDetectionGroundedSAM2(L3_Algorithm):
    def __init__(self, 
                 detection_prompt: str,
                 args_list: List[Dict[str, Any]] = None):
        super().__init__()
        if args_list is None:
            args_list = []
        if not isinstance(args_list, list) or not all(isinstance(d, dict) for d in args_list):
            raise ValueError("`args_list` must be a list of dicts")
        self.args_list = args_list
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1

        # Config
        # FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
        SAM2_CHECKPOINT = os.getenv("OBJDET_SAM_CHECKPOINT")
        SAM2_CONFIG = os.getenv("OBJDET_SAM_CONFIG")
        self.detection_prompt = detection_prompt

        # parse DEVICE into a torch.device
        DEVICE = os.getenv("DEVICE", "cpu")
        self.torch_device = torch.device(DEVICE)
        if self.torch_device.type == "cuda":
            props = torch.cuda.get_device_properties(self.torch_device.index)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # build Florence-2 on the parsed device
        self.florence2_model = Florence2ForConditionalGeneration.from_pretrained(
            "florence-community/Florence-2-large", trust_remote_code=True, torch_dtype="auto"
        ).eval().to(self.torch_device)
        self.florence2_processor = AutoProcessor.from_pretrained(
            "florence-community/Florence-2-large", trust_remote_code=True
        )

        # build SAM 2 on the same device
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=self.torch_device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def process_data(self, input, l2_datacube: Optional[xr.Dataset] = None) -> List[L3_result]:
        results: List[L3_result] = []
        data_source = l2_datacube if l2_datacube is not None else input
        getter = data_source.get_rgb_image if hasattr(data_source, 'get_rgb_image') else input.get_rgb_image

        args_iter = self.args_list if self.args_list else [{}]

        for time_index in self.time_indices:
            for kwargs in args_iter:
                merged_kwargs = {**kwargs, 'time_index': time_index}
                frame = getter(**merged_kwargs)
                frame = np.nan_to_num(frame, nan=0).astype(np.float32)
                tonemapper = cv2.createTonemapReinhard(1.2,0,0,0)
                ldr = tonemapper.process(frame)
                ldr_8bit = (ldr * 255).clip(0,255).astype(np.uint8)
                pil_input = Image.fromarray(ldr_8bit).convert("RGB")

                # open-vocabulary detection
                device_type = self.torch_device.type
                dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
                with torch.autocast(device_type=device_type, dtype=dtype):
                    prompt = "<OPEN_VOCABULARY_DETECTION>"
                    florence_output = run_florence2(
                        prompt,
                        self.detection_prompt,
                        self.florence2_model,
                        self.florence2_processor,
                        pil_input
                    )

                raw_text = florence_output["raw_text"]
                florence_parsed_output = florence_output["parsed_output"]
                flor = florence_parsed_output[prompt]

                boxes = np.array(flor["bboxes"])
                names = flor["bboxes_labels"]

                # SAM2 mask prediction
                self.sam2_predictor.set_image(np.array(pil_input))
                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=False
                )
                masks = masks.squeeze(1) if masks.ndim == 4 else masks

                sam_scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)

                # annotate
                dets = sv.Detections(xyxy=boxes, mask=masks.astype(bool),
                                      class_id=np.arange(len(names)))
                box_ann = sv.BoxAnnotator().annotate(ldr_8bit.copy(), dets)
                box_ann = sv.LabelAnnotator().annotate(box_ann, dets,
                                                      labels=[str(n) for n in names])
                mask_ann = sv.MaskAnnotator().annotate(box_ann.copy(), dets)

                frame_result = FrameResult(
                    kwargs=merged_kwargs,
                    input_image=pil_input,
                    annotated_boxes_image=Image.fromarray(box_ann),
                    annotated_mask_image=Image.fromarray(mask_ann),
                    detections=[
                        Detection(
                            bbox={"x":int(b[0]), "y":int(b[1]),
                                  "width":int(b[2]-b[0]), "height":int(b[3]-b[1])},
                            class_id=name
                        ) for b, name in zip(boxes, names)
                    ],
                    florence_raw_text=raw_text,
                    florence_parsed_output=florence_parsed_output,
                    sam_scores=sam_scores
                )

                w, h = pil_input.size
                row = Image.new("RGB", (w * 3, h))
                row.paste(pil_input, (0, 0))
                row.paste(frame_result.annotated_boxes_image, (w, 0))
                row.paste(frame_result.annotated_mask_image, (w*2, 0))

                results.append(L3_result(
                    debug_image=row,
                    algorithm_results=frame_result,
                    time_indices=[time_index],
                    result_type="detections"
                ))

        return results

# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import sys
sys.path.append("src/main/python/L4/LLaVACustom/LLaVA")  # TODO: manage import paths properly

import os
from dotenv import load_dotenv

# ─── Load .env and get default DEVICE ─────────────────────────────────────────
load_dotenv()

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union, Optional
import cv2

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

from L4.L4_Algorithm import L4_Algorithm
from L3.L3_Algorithm import L3_result


class LlavaChat:
    def __init__(
        self,
        model_path: str,
        device: str,
        model_base: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        conv_mode: str = None,
    ):
        disable_torch_init()

        self.device = device
        if "cuda" in self.device:
            props = torch.cuda.get_device_properties(self.device.index)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Load & move model to same device
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device_map=self.device,
            device=self.device,
        )
        self.model = self.model.to(self.device)

        # Pick conversation template
        lname = model_name.lower()
        if "llama-2" in lname:
            mode = "llava_llama_2"
        elif "mistral" in lname:
            mode = "mistral_instruct"
        elif "v1.6-34b" in lname:
            mode = "chatml_direct"
        elif "v1" in lname:
            mode = "llava_v1"
        elif "mpt" in lname:
            mode = "mpt"
        else:
            mode = "llava_v0"
        if conv_mode and conv_mode != mode:
            print(f"[WARNING] using conv_mode={conv_mode} instead of inferred={mode}")
            mode = conv_mode

        self.conv = conv_templates[mode].copy()
        self.roles = ("user", "assistant") if "mpt" in lname else self.conv.roles
        self.conv_mode = mode
        self._image_tensor = None
        self._image_size = None
        self._first = True

    def prepare_image(self, image_source: Union[np.ndarray, str, Image.Image]):
        arr = image_source.astype(np.float32)
        minv, maxv = arr.min(), arr.max()
        if maxv > minv:
            arr = (arr - minv) / (maxv - minv) * 255.0
        else:
            arr = np.zeros_like(arr)
        pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        self._image_size = pil.size

        tensor = process_images([pil], self.image_processor, self.model.config)
        if isinstance(tensor, list):
            tensor = [img.to(self.device, dtype=torch.float16) for img in tensor]
        else:
            tensor = tensor.to(self.device, dtype=torch.float16)

        self._image_tensor = tensor
        self._first = True
        self.conv = conv_templates[self.conv_mode].copy()

    def ask(
        self,
        text: str,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        debug: bool = False,
    ) -> str:
        inp = text
        if self._image_tensor is not None and self._first:
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            self._first = False

        self.conv.append_message(self.roles[0], inp)
        self.conv.append_message(self.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self._image_tensor,
                image_sizes=[self._image_size] if self._image_tensor is not None else None,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
            )

        text_out = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        self.conv.messages[-1][-1] = text_out

        if debug:
            print("Prompt:\n", prompt)
            print("Output:\n", text_out)

        return text_out


class LLaVACustom(L4_Algorithm):
    """
    Custom adapter that takes frames via `input.get_rgb_image` and queries LLaVA.
    Frame arrays are normalized internally.
    """
    def __init__(
        self,
        args_list: List[Dict[str, Any]] = None,
        target_time_index: Optional[int] = None,
        model_path: str = None,
        model_base: str = "liuhaotian/llava-v1.5-7b",
        conv_mode: str = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        load_8bit: bool = False,
        load_4bit: bool = False,
        debug: bool = False,
        prompt: str = "Describe the content of the image.",
    ):
        super().__init__()

        DEVICE = os.getenv("DEVICE", "cpu")
        model_path = model_path or os.getenv("LLAVA_CHECKPOINT_DIR")

        self.bot = LlavaChat(
            model_path=model_path,
            device=DEVICE,
            model_base=model_base,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            conv_mode=conv_mode,
        )

        self.prompt = prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.args_list = args_list or []
        self.target_time_index = target_time_index

    def _overlay_result_on_image(self, rgb_image: np.ndarray, l3_result: Any, time_index: int) -> np.ndarray:
        """Overlay L3 algorithm results (mask, detections, etc.) onto RGB image."""
        result_type = l3_result.result_type
        algo_results = l3_result.algorithm_results

        overlay = rgb_image.copy()

        if result_type == "mask" and algo_results is not None:
            mask = algo_results.get("mask")
            if mask is not None:
                mask_normalized = (mask / mask.max() * 255).astype(np.uint8) if mask.max() > 0 else mask.astype(np.uint8)
                cmap = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
                overlay = (overlay / overlay.max() * 255).astype(np.uint8) if overlay.max() > 0 else overlay.astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.7, cmap, 0.3, 0)

        elif result_type == "detections" and algo_results is not None:
            detections = algo_results.get("detections", [])
            for det in detections:
                bbox = det.bbox
                confidence = det.confidence
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2f}"
                cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif result_type == "change_map" and algo_results is not None:
            change_map = algo_results.get("change_map")
            if change_map is not None:
                change_normalized = (change_map / change_map.max() * 255).astype(np.uint8) if change_map.max() > 0 else change_map.astype(np.uint8)
                cmap = cv2.applyColorMap(change_normalized, cv2.COLORMAP_RED)
                overlay = (overlay / overlay.max() * 255).astype(np.uint8) if overlay.max() > 0 else overlay.astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.7, cmap, 0.3, 0)

        return overlay

    def _format_l3_context(self, l3_results: List[Any], time_index: int) -> str:
        """Format L3 results as text context for the VLM prompt."""
        context_parts = []
        for result in l3_results:
            result_type = result.result_type
            algo_results = result.algorithm_results
            algo_name = algo_results.__class__.__name__ if hasattr(algo_results, '__class__') else str(type(algo_results))

            if result_type == "mask" and algo_results is not None:
                mask = algo_results.get("mask")
                if mask is not None:
                    unique_classes = np.unique(mask)
                    context_parts.append(f"Land use classification detected {len(unique_classes)} classes: {list(unique_classes)}")

            elif result_type == "detections" and algo_results is not None:
                if hasattr(algo_results, 'detections'):
                    detections = algo_results.detections
                else:
                    detections = algo_results.get("detections", [])
                if detections:
                    context_parts.append(f"Object detection found {len(detections)} objects")

            elif result_type == "change_map" and algo_results is not None:
                change_map = algo_results.get("change_map")
                if change_map is not None:
                    change_pixels = np.sum(change_map > 0)
                    context_parts.append(f"Change detection identified {change_pixels} changed pixels")

            elif result_type == "datacube" and algo_results is not None:
                context_parts.append("Environmental indicator datacube available for analysis")

        return "; ".join(context_parts) if context_parts else "No analysis results available"

    def process_data(self, input, l3_results: List[Any] = None, target_time_index: Optional[int] = None) -> List[str]:
        results: List[str] = []
        getter = input.get_rgb_image
        time_idx = target_time_index if target_time_index is not None else self.target_time_index

        relevant_results = []
        if l3_results and time_idx is not None:
            rgb_image = getter(time_idx)
            relevant_results = [
                l3_result for l3_result in l3_results
                if time_idx in l3_result.time_indices or not l3_result.time_indices
            ]
            if relevant_results:
                context = self._format_l3_context(relevant_results, time_idx)
                prompt = f"{context}. {self.prompt}"
                self.bot.prepare_image(rgb_image)

                reply = self.bot.ask(
                    text=prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    debug=self.debug,
                )
                results.append(reply)

        if not results:
            for frame_kwargs in self.args_list:
                print(f"Processing frame with args: {frame_kwargs}")
                frame = getter(**frame_kwargs)
                self.bot.prepare_image(frame)

                used_prompt = frame_kwargs.get("prompt", self.prompt)
                reply = self.bot.ask(
                    text=used_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    debug=self.debug,
                )
                results.append(reply)

        return results

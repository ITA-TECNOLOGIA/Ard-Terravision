# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Date: Sep 2025
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es)
# All rights reserved
# --------------------------------------------------------------------------------

import os
from dotenv import load_dotenv

# ─── Load .env and get default DEVICE ─────────────────────────────────────────
load_dotenv()

import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union, Optional
import cv2

import torch
from transformers import AutoProcessor
from peft import PeftModel

# Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration

# Framework base
from L4.L4_Algorithm import L4_Algorithm
from L3.L3_Algorithm import L3_result

def process_vision_info(example):
    """
    Extract and process the image(s) from the example for the processor.
    Assumes the image is already a PIL image or a path to an image file.
    """
    image_data = example[1]["content"][0]["image"]
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data  # Assume it's a PIL Image already
    return [image], None


# ────────────────────────────── Qwen chat core ────────────────────────────────
class QwenVLChat:
    """
    Minimal chat wrapper mirroring LlavaChat's interface:
      - prepare_image(image)
      - ask(text, temperature, max_new_tokens, debug) -> str
    Maintains conversation state and inserts the image token only for the first
    user message after `prepare_image`.
    """

    def __init__(
        self,
        base_model: str,
        device: str,
        lora_checkpoint: str | None = None,
        torch_dtype: str = "float16",
        system_prompt: str = "You are a helpful assistant that analyzes satellite images.",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.device = torch.device(device)
        self.system_prompt = system_prompt
        self._image = None
        self._first_after_image = False

        # Dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float16

        # Load model + optional LoRA
        print("[QwenVLChat] Loading base model…")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        if lora_checkpoint:
            print("[QwenVLChat] Applying LoRA checkpoint…")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)

        self.model.eval()

        # Processor (tokenizer + image preproc)
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code, use_fast=True)

        # Conversation state (Qwen uses chat template from the processor)
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        ]

        # CUDA friendly knobs
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device.index or 0)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    # Normalize any input (ndarray/str/PIL) to a PIL RGB image
    @staticmethod
    def _to_pil_rgb(img: Union[np.ndarray, str, Image.Image]) -> Image.Image:
        if isinstance(img, str):
            pil = Image.open(img).convert("RGB")
            return pil
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            minv, maxv = float(arr.min()), float(arr.max())
            if maxv > minv:
                arr = (arr - minv) / (maxv - minv) * 255.0
            else:
                arr = np.zeros_like(arr)
            pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
            return pil
        raise TypeError(f"Unsupported image type: {type(img)}")

    def prepare_image(self, image_source: Union[np.ndarray, str, Image.Image]):
        """Prepare image for the next prompt; will be attached to the next ask()."""
        rgb_image = np.nan_to_num(image_source, nan=0.0)
        pil = self._to_pil_rgb(rgb_image)
        self._image = pil
        self._first_after_image = True

    def ask(
        self,
        text: str,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        debug: bool = False,
    ) -> str:
        """
        Append a user turn (optionally with the prepared image on first use) and
        generate a response. Keeps conversation state.
        """
        # Build the user content chunk
        user_content: List[Dict[str, Any]] = []
        if self._image is not None and self._first_after_image:
            user_content.append({"type": "image", "image": self._image})
        user_content.append({"type": "text", "text": text})

        # Update state
        self._messages.append({"role": "user", "content": user_content})

        # Convert messages to chat template text
        chat_text = self.processor.apply_chat_template(
            self._messages, tokenize=False, add_generation_prompt=True
        )

        # Ensure the image token is present if we attached an image
        if (self._image is not None) and (self._first_after_image):
            if getattr(self.processor, "image_token", None) and (self.processor.image_token not in chat_text):
                chat_text += f" {self.processor.image_token}"

        # Turn message list into vision input blobs
        image_inputs, _ = process_vision_info(self._messages)

        # Tokenize + prepare tensors
        model_inputs = self.processor(
            text=[chat_text], images=image_inputs, return_tensors="pt"
        ).to(self.device)

        if debug:
            print("\n[QwenVLChat] ===== Prompt (truncated) =====")
            print(chat_text[:1000])
            print("=========================================\n")

        # Generate
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "do_sample": temperature > 0.0,
            "use_cache": True,
        }

        with torch.no_grad(), torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        # Slice off the prompt tokens to keep only the new text
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, output_ids)]
        text_out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        # Commit assistant turn to history
        self._messages.append({"role": "assistant", "content": [{"type": "text", "text": text_out}]})

        # After first use, we won't auto-attach the image again unless prepare_image() is called
        self._first_after_image = False

        if debug:
            print("[QwenVLChat] ===== Output =====")
            print(text_out)
            print("================================")

        return text_out


# ──────────────────────── L4 wrapper (module-exchangeable) ────────────────────
class QwenCustom(L4_Algorithm):
    def __init__(
        self,
        args_list: List[Dict[str, Any]] = None,
        target_time_index: Optional[int] = None,
        base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_checkpoint: str = None,
        torch_dtype: str = "float16",
        device_map: str = "auto",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        debug: bool = False,
        prompt: str = "Describe the content of the image.",
        system_prompt: str = "You are a helpful assistant that analyzes satellite images.",
    ):
        super().__init__()

        DEVICE = os.getenv("DEVICE", "cpu")
        lora_checkpoint = lora_checkpoint or os.getenv("QWEN_CHECKPOINT_DIR")

        self.bot = QwenVLChat(
            base_model=base_model,
            device=DEVICE,
            lora_checkpoint=lora_checkpoint,
            torch_dtype=torch_dtype,
            system_prompt=system_prompt,
            device_map=device_map,
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
                overlay = cv2.addWeighted(overlay, 0.7, cmap, 0.3, 0)

        return overlay

    def _format_l3_context(self, l3_results: List[Any], time_index: int) -> str:
        """Format L3 results as text context for the VLM prompt."""
        context_parts = []
        for result in l3_results:
            result_type = result.result_type
            algo_results = result.algorithm_results

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
        """
        Iterates frames via `input.get_rgb_image(**kwargs)` and queries Qwen.
        Each element of args_list may include a custom 'prompt'.
        """
        results: List[str] = []
        getter = input.get_rgb_image
        time_idx = target_time_index if target_time_index is not None else self.target_time_index

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
        else:
            for frame_kwargs in self.args_list:
                print(f"[QwenVLCustom] Processing frame with args: {frame_kwargs}")
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

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
        if device_map == "auto" and self.device.type == "cuda":
            device_map = {"": str(self.device)}
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
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
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
        ).to(self.model.device)

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
        self.args_list = args_list if args_list is not None else [{}]
        self.debug_time_index: Optional[int] = None  # injected by PipelineConfig from L1
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1

    def process_data(self, input, l3_results: List[Any] = None, target_time_index: Optional[int] = None) -> List[str]:
        results: List[str] = []
        getter = input.get_rgb_image

        for time_index in self.time_indices:
            if l3_results:
                relevant = [
                    r for r in l3_results
                    if time_index in r.time_indices or not r.time_indices
                ]
            else:
                relevant = []

            if relevant:
                rgb_image = getter(time_index)
                context = self._format_l3_context(relevant, time_index)
                self.bot.prepare_image(rgb_image)
                reply = self.bot.ask(
                    text=f"{context}. {self.prompt}",
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    debug=self.debug,
                )
                results.append(reply)
                continue

            for frame_kwargs in self.args_list:
                merged = {**frame_kwargs, 'time_index': time_index}
                frame = getter(**merged)
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

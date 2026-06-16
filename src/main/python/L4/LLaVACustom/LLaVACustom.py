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

        self.device = torch.device(device)
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device.index or 0)
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
            device_map=str(self.device),
            device=str(self.device),
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

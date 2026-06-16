# --------------------------------------------------------------------------------
# ARD - TERRAVISION
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved
# --------------------------------------------------------------------------------

import numpy as np
from typing import List, Dict, Any, Optional

from L4.L4_Algorithm import L4_Algorithm


class DummyImageGetter:
    def get_rgb_image(self, **kwargs) -> np.ndarray:
        return np.zeros((224, 224, 3), dtype=np.uint8)


class DummyLlavaChat:
    def __init__(self, **kwargs):
        print("[DummyLlavaChat] Initialized with kwargs:", kwargs)

    def prepare_image(self, image_source):
        print("[DummyLlavaChat] Image prepared.")

    def ask(self, text: str, temperature: float = 0.2, max_new_tokens: int = 512, debug: bool = False) -> str:
        if debug:
            print("[DummyLlavaChat] Prompt:", text)
        return f"[DUMMY RESPONSE] to prompt: '{text}'"


class DummyLLaVACustom(L4_Algorithm):
    def __init__(self, args_list: List[Dict[str, Any]] = None):
        super().__init__()
        self.prompt = "Describe the content of the image."
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.debug = True
        self.args_list = args_list if args_list is not None else [{}]
        self.debug_time_index: Optional[int] = None  # injected by PipelineConfig from L1
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1
        self.time_indices: List[int] = []  # injected by PipelineConfig from L1
        self.bot = DummyLlavaChat()
        self.input = DummyImageGetter()

    def process_data(self, input=None, l3_results: List[Any] = None, target_time_index: Optional[int] = None) -> List[str]:
        input = input or self.input
        results = []

        for time_index in self.time_indices:
            if l3_results:
                for l3_result in l3_results:
                    if time_index in l3_result.time_indices or not l3_result.time_indices:
                        frame = input.get_rgb_image(time_index=time_index)
                        self.bot.prepare_image(frame)
                        prompt = f"L3 result context: {l3_result.result_type}. {self.prompt}"
                        response = self.bot.ask(
                            text=prompt,
                            temperature=self.temperature,
                            max_new_tokens=self.max_new_tokens,
                            debug=self.debug,
                        )
                        results.append(response)
            else:
                for frame_kwargs in self.args_list:
                    merged = {**frame_kwargs, 'time_index': time_index}
                    frame = input.get_rgb_image(**merged)
                    self.bot.prepare_image(frame)
                    prompt = frame_kwargs.get("prompt", self.prompt)
                    response = self.bot.ask(
                        text=prompt,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        debug=self.debug,
                    )
                    results.append(response)

        return results


# Example usage:
if __name__ == "__main__":
    dummy = DummyLLaVACustom(args_list=[{"prompt": "What do you see?"}])
    results = dummy.process_data()
    print("\nResults:", results)

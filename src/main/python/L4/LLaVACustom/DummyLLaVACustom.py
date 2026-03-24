# --------------------------------------------------------------------------------
# ARD - TERRAVISION 
# Version: 1.0
# Copyright (c) 2025 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: May 2025
# All rights reserved 
# --------------------------------------------------------------------------------

import numpy as np
from typing import List, Dict, Any

class DummyImageGetter:
    def get_rgb_image(self, **kwargs) -> np.ndarray:
        # Return a dummy RGB image (e.g., 224x224 black image)
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


class DummyLLaVACustom:
    def __init__(self, args_list: List[Dict[str, Any]] = None):
        self.prompt = "Describe the content of the image."
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.debug = True
        self.args_list = args_list or [{"dummy_key": "dummy_value"}]
        self.bot = DummyLlavaChat()
        self.input = DummyImageGetter()

    def process_data(self, input=None) -> List[str]:
        input = input or self.input
        results = []
        for frame_kwargs in self.args_list:
            print(f"[DummyLLaVACustom] Processing frame with args: {frame_kwargs}")
            frame = input.get_rgb_image(**frame_kwargs)
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

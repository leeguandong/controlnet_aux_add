import os
import warnings

import torch
import numpy as np
import PIL.Image
from controlnet_aux.util import HWC3, resize_image
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
from huggingface_hub import hf_hub_download


class DepthEstimator:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, cache_dir=None, local_files_only=False, device="cuda"):
        if os.path.isdir(pretrained_model_or_path):
            model_path = pretrained_model_or_path
        else:
            model_path = hf_hub_download(pretrained_model_or_path, cache_dir=cache_dir,
                                         local_files_only=local_files_only)

        model = pipeline(task="depth-estimation", model=model_path, device=device)
        return cls(model)

    @torch.inference_mode()
    def __call__(self, input_image, output_type='pil', **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn(
                "Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, resolution=detect_resolution)
        image = PIL.Image.fromarray(input_image)

        image = self.model(image)
        image = image["depth"]

        detected_map = image
        detected_map = np.array(detected_map).astype(np.uint8)
        detected_map = HWC3(detected_map)
        detected_map = resize_image(detected_map, resolution=image_resolution)

        if output_type == "pil":
            detected_map = PIL.Image.fromarray(detected_map)
        return detected_map

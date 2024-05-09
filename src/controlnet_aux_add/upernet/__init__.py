import os
import cv2
import numpy as np
import PIL.Image
import torch
import warnings
from ..util import resize_image
from controlnet_aux.util import HWC3, ade_palette
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from huggingface_hub import hf_hub_download


class ImageUperNetSegmentor:
    def __init__(self, image_processor, model):
        self.image_processor = image_processor
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, cache_dir=None, local_files_only=False):
        if os.path.isdir(pretrained_model_or_path):
            model_path = pretrained_model_or_path
        else:
            model_path = hf_hub_download(pretrained_model_or_path, cache_dir=cache_dir,
                                         local_files_only=local_files_only)

        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = UperNetForSemanticSegmentation.from_pretrained(model_path)
        return cls(image_processor, model)

    def to(self, device):
        self.model.to(device)
        return self

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

        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        detect_resolution = kwargs.pop("detect_resolution", 512)
        image_resolution = kwargs.pop("image_resolution", 512)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, resolution=detect_resolution)
        image = PIL.Image.fromarray(input_image)

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        outputs = self.model(pixel_values.to(device))

        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        seg = seg.cpu().numpy()
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(ade_palette()):
            color_seg[seg == label, :] = color
        detected_map = color_seg.astype(np.uint8)

        detected_map = resize_image(detected_map, resolution=image_resolution, interpolation=cv2.INTER_NEAREST)
        if output_type == "pil":
            detected_map = PIL.Image.fromarray(detected_map)
        return detected_map

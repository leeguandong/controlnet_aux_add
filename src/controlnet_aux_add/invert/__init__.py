import warnings
import numpy as np
import PIL.Image
from controlnet_aux.util import HWC3


class Invert:
    def __call__(self, input_image=None, output_type=None, **kwargs):
        if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")

        if input_image is None:
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        input_image = HWC3(input_image)
        detected_map = 255 - input_image

        if output_type == "pil":
            detected_map = PIL.Image.fromarray(input_image)
        return detected_map

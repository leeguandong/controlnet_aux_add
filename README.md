# ControlNet auxiliary models

This is a PyPi installable package of [lllyasviel's ControlNet Annotators](https://github.com/lllyasviel/ControlNet/tree/main/annotator)

The code is copy-pasted from the respective folders in https://github.com/lllyasviel/ControlNet/tree/main/annotator and connected to [the 🤗 Hub](https://huggingface.co/lllyasviel/Annotators).

All credit & copyright goes to https://github.com/lllyasviel .

## Install

```
pip install controlnet_aux_add==0.0.1
```

## Usage


You can use the processor class, which can load each of the auxiliary models with the following code
```python
import requests
from PIL import Image
from io import BytesIO

from controlnet_aux_add.processor import Processor

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load processor from processor_id
# options are:
# ["dpt","upernet"]
processor_id = 'dpt'
processor = Processor(processor_id)

processed_image = processor(img, to_pil=True)
```

Each model can be loaded individually by importing and instantiating them as follows
```python
from PIL import Image
import requests
from io import BytesIO
from controlnet_aux_add import DepthEstimator, ImageUperNetSegmentor

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load checkpoints
depth = DepthEstimator.from_pretrained(
        os.path.join(str(current_file_path[5]), "weights/annotators_aux/depth-anything-base-hf/"))
upernet = ImageUperNetSegmentor.from_pretrained(
        os.path.join(str(current_file_path[5]), "weights/annotators_aux/upernet-convnext-small/"))

# specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
# det_config: ./src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py
# det_ckpt: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
# pose_config: ./src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py
# pose_ckpt: https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# process
processed_image_dpt = depth(img)
processed_image_upernet = upernet(img)

```

### Image resolution

In order to maintain the image aspect ratio, `detect_resolution`, `image_resolution` and images sizes need to be using multiple of `64`.

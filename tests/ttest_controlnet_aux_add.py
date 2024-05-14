import os
import sys
from pathlib import Path

current_file_path = Path(__file__).parents
sys.path.append(str(current_file_path[1]) + "/src/")

import torch
import shutil
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from controlnet_aux import LineartDetector
from controlnet_aux_add import DepthEstimator, ImageUperNetSegmentor, Invert

OUTPUT_DIR = "./outputs"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def output(name, img):
    img.save(os.path.join(OUTPUT_DIR, "{:s}.png".format(name)))


def common(name, processor, img):
    output(name, processor(img))
    output(name + "_pil_np", Image.fromarray(processor(img, output_type="np")))
    output(name + "_np_np", Image.fromarray(processor(np.array(img, dtype=np.uint8), output_type="np")))
    output(name + "_np_pil", processor(np.array(img, dtype=np.uint8), output_type="pil"))
    output(name + "_scaled", processor(img, detect_resolution=640, image_resolution=768))


def return_pil(name, processor, img):
    output(name + "_pil_false", Image.fromarray(processor(img, return_pil=False)))
    output(name + "_pil_true", processor(img, return_pil=True))


def img():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    url = "https://hf-mirror.com/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
    return img


def test_lineart(img):
    # lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    lineart = LineartDetector.from_pretrained(os.path.join(str(current_file_path[5]), "weights/annotators_aux/"))
    common("lineart", lineart, img)
    return_pil("lineart", lineart, img)
    output("lineart_coarse", lineart(img, coarse=True))


def test_depth_anything(img):
    depth = DepthEstimator.from_pretrained(
        os.path.join(str(current_file_path[5]), "weights/annotators_aux/depth-anything-base-hf/"))
    common("depth", depth, img)
    return_pil("depth", depth, img)
    output("depth", depth(img))


def test_upernet(img):
    upernet = ImageUperNetSegmentor.from_pretrained(
        os.path.join(str(current_file_path[5]), "weights/annotators_aux/upernet-convnext-small/"))
    upernet = upernet.to(device)
    common("upernet", upernet, img)
    return_pil("upernet", upernet, img)
    output("upernet", upernet(img))


def test_invert(img):
    invert = Invert()
    common("invert", invert, img)
    output("invert", invert(img))


if __name__ == "__main__":
    img = Image.open("./outputs/images_pose.png").convert("RGB").resize((512, 512))
    # test_lineart(img)
    # test_depth_anything(img)
    # test_upernet(img)
    test_invert(img)

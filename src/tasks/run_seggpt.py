"""
Functions for sending a request to the SegGPT TorchServe implementation and processing the results.
"""

import io
import os
import base64

from prefect import task

import requests
from PIL import Image



def prepare_images(image_dir: str):
    """
    Load a directory of images into binary format. Returns the list of binary images, where each
    entry in the list is a tuple of the form (binarized image, image name).
    """
    image_names = os.listdir(image_dir)
    images = [
        [
            base64.b64encode(open(os.path.join(image_dir, image), "rb").read()).decode(
                "utf-8"
            ),
            image,
        ]
        for image in image_names
    ]
    return images


@task()
def request(
    input_dir: str,
    prompt_dir: str,
    target_dir: str,
    output_dir: str,
    patch_images: bool,
    num_prompts: int,
):
    """
    Send a request to SegGPT and save the results.
    """
    input_imgs = prepare_images(input_dir)
    prompt_imgs = prepare_images(prompt_dir)
    target_imgs = prepare_images(target_dir)

    if num_prompts == 0:
        num_prompts_for_request = len(prompt_imgs)
    else:
        num_prompts_for_request = num_prompts

    data = {
        "input": input_imgs,
        "prompts": prompt_imgs,
        "targets": target_imgs,
        "output_dir": output_dir,
        "patch_images": patch_images,
        "num_prompts": num_prompts_for_request,
    }
    response = requests.post("http://localhost:8080/predictions/seggpt", json=data)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mask_index, mask in enumerate(response.json()):
        mask_data = base64.b64decode(mask)
        mask_image = Image.open(io.BytesIO(mask_data))
        original_name, ext = os.path.splitext(input_imgs[mask_index][1])
        out_path = os.path.join(output_dir, f"{original_name}_mask{ext}")
        mask_image.save(out_path)

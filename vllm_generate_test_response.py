import argparse
import torch
import json
import os

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import pandas as pd


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def generate_safety_test_responses(model_path, data_path, image_folder, save_path, model_base=None):

    # model_name = get_model_name_from_path(model_path)
    model_name = "llava"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    test_json = []
    response_list = []
    query_list = []   
    image_idx_list = []
    
    with open(data_path, 'r') as data_file:
        test_json = json.load(data_file)
    
    for idx in range(len(test_json)):
                
        image_idx = test_json[idx]['id']
        image_idx_list.append(image_idx)

        image_path = os.path.join(image_folder, test_json[idx]['image'])
        image_files = []
        image_files.append(image_path)
        query = test_json[idx]['conversations'][0]['value'][8:]
        query_list.append(query)

        response = model_generate(model_name, model, tokenizer, image_processor, image_files, query)
        response_list.append(response)

    df = pd.DataFrame({'image_idx':image_idx_list, 'query':query_list, 'response':response_list})
    df.to_csv(os.path.join(save_path, model_name + "-VLGuard-test-response_lora20epoch.csv"), index=False)



def model_generate(model_name, model, tokenizer, image_processor, image_files, query):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # print(model_name)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"


    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample= False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=250,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    return outputs


model_path = "./LLaVA/checkpoints/merge_ckpt/lora/llava-v1.5-7b"
data_path = "./VLGuard/VLGuard_test_processed.json"
image_folder = "./VLGuard/test/"
save_path = "./"


generate_safety_test_responses(model_path=model_path, model_base=None, data_path=data_path, image_folder=image_folder, save_path=save_path)
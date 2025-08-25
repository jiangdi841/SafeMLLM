# generate response of VLGuard-test

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
from model.openllama import OpenLLAMAPEFTModel
import torch
import json
import pandas as pd


def my_load_model(args, model_name):
    if 'pandagpt' in model_name.lower():
        print('Loading the original models')
        model = OpenLLAMAPEFTModel(**args)
        print('Loading the delta ckpt!')
        delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
        model = model.eval().half().cuda()
        print('Initialize the PandaGPT-7b model over!')
        return model


def pandagpt_generate( 
    model,   
    input_text, 
    image_path, 
    audio_path=None, 
    video_path=None, 
    thermal_path=None, 
    max_length=250, 
    top_p=0.01, 
    temperature=1
    ):
    # prepare the prompt
    # prompt_text = f' ### Human: {input_text}\n### Assistant: '
    prompt_text = input_text
    print("The prompt is: " + prompt_text)
    
    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': []
    })
    print("Here is the response:" + response)
    return response
    

def pandagpt_generate_safety_test_responses(args, model_path, data_path, image_folder, save_path, model_base=None):

    model_name = 'pandagpt'
    model = my_load_model(args, model_name)

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

        query = test_json[idx]['conversations'][0]['value'][8:]
        query_list.append(query)

        response = pandagpt_generate(model, query, image_path)
        response_list.append(response)

    df = pd.DataFrame({'image_idx':image_idx_list, 'query':query_list, 'response':response_list})
    df.to_csv(os.path.join(save_path, model_name + "MLLM-test-response.csv"), index=False)


# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': './imagebind',
    'vicuna_ckpt_path': './PandaGPT/vicuna-fastchat-merged/7b_v0',
    'delta_ckpt_path': './PandaGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
    

model_path = "./PandaGPT/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt"
data_path = "./VLGuard/VLGuard_test_processed_new.json"
image_folder = "./VLGuard/test/"
save_path = "./"
pandagpt_generate_safety_test_responses(args, model_path, data_path, image_folder, save_path, model_base=None)

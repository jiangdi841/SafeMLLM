import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np

def moderate(chat, model, tokenizer, device):
  input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
  output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
  prompt_len = input_ids.shape[-1]
  return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def eval_vlguard_safe(model_path, data_folder, data_csv, save_path, dtype = torch.bfloat16, device = "cuda"):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)


    df = pd.read_csv(os.path.join(data_folder, data_csv))
    llama_guard_3 = []
    for i in range(len(df)):
        image_id = df.iloc[i]['image_idx']
        query = df.iloc[i]['query']
        response = df.iloc[i]['response']
        chat = [{"role": "user", "content": query},
                {"role": "assistant", "content": response}]

        if df.iloc[i]['response'] is np.nan:
            response = "I cannot answer this question"
            chat = [{"role": "user", "content": query},
                    {"role": "assistant", "content": response}]
        if df.iloc[i]['query'] is np.nan:
            query = "What is in the image?"
            chat = [{"role": "user", "content": query},
                    {"role": "assistant", "content": response}]
        output = moderate(chat, model, tokenizer, device)
        if 'unsafe' in output:
            print(output)
        llama_guard_3.append(output)

    df['llama_guard_3'] = llama_guard_3  
    df.to_csv(os.path.join(save_path, data_csv[:-4] + '_llamaguard3.csv'), index=False)  


def eval_advbench_safe(model_path, data_folder, data_csv, save_path, dtype = torch.bfloat16, device = "cuda"):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

    df = pd.read_csv(os.path.join(data_folder, data_csv))
    llama_guard_3 = []
    for i in range(len(df)):
        query = df.iloc[i]['query']
        response = df.iloc[i]['response']
        chat = [{"role": "user", "content": query},
                {"role": "assistant", "content": response}]

        if df.iloc[i]['query'] is np.nan:
            query = "What is in the image?"
            chat = [{"role": "user", "content": query},
                    {"role": "assistant", "content": response}]
        if df.iloc[i]['response'] is np.nan:
            response = "I can't answer this query"
            chat = [{"role": "user", "content": query},
                    {"role": "assistant", "content": response}]

        output = moderate(chat, model, tokenizer, device)

        if 'unsafe' in output:
            print(output)
        llama_guard_3.append(output)
    df['llama_guard_3'] = llama_guard_3  
    df.to_csv(os.path.join(save_path, data_csv[:-4] + '_llamaguard3.csv'), index=False)  



model_path = "./Llama-Guard-3-8B/"
data_folder = "."
data_csv = "./llava-VLGuard-test-response.csv"
save_path = "./"


eval_vlguard_safe(model_path, data_folder, data_csv, save_path)


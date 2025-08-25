import os
import torch
from datasets import load_dataset

adv_bench = load_dataset("./AdvBench/")
dataset = adv_bench['train']

file_path = 'processed_advbench.txt'
with open(file_path, "w") as file:
    for prompt in dataset['prompt']:
        file.write(prompt + '\n')
        
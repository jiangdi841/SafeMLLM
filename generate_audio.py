from diffusers import AudioLDM2Pipeline
import torch
import scipy
import pandas as pd
import os
 

data_path = './llava-v1.5-7b-VLGuard-test-image-discription-one-sentence.csv'
save_path = './MLGuard/Audio'
pipe = AudioLDM2Pipeline.from_pretrained("./audioldm2", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
 

data_df = pd.read_csv(data_path) 

for idx in range(len(data_df)):
    image_id = data_df['images'][idx]
    print("Processing " + image_id)
    prompt = data_df['response'][idx]
    audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]
    audio_name = image_id + ".wav"
    scipy.io.wavfile.write(os.path.join(save_path, audio_name), rate=16000, data=audio)





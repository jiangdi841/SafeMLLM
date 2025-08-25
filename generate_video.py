import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video, export_to_gif
import os
import pandas as pd
 

data_path = './llava-v1.5-7b-VLGuard-test-image-discription-one-sentence.csv'
save_path = './MLGuard/Video'

pipe = DiffusionPipeline.from_pretrained("./zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

data_df = pd.read_csv(data_path)

for idx in range(len(data_df)):
    image_id = data_df['images'][idx]
    print("Processing " + image_id)
    prompt = data_df['response'][idx]
    video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    video_name = image_id + ".mp4"
    output_path = export_to_video(video_frames, output_video_path=os.path.join(save_path, video_name))


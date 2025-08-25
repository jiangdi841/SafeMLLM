# SafeMLLM

## 1. Prepare Pre-trained Models

Deploy LLAVA-v1.5-7b at ./LLaVA

Deploy PandaGPT at ./PandaGPT

## 2. Collect V-Advbench

```bash
# Deploy Stable Diffusion v2
git clone https://github.com/Stability-AI/stablediffusion.git
cd stablediffusion
pip install -r requirements.txt
# download v2-1_768-ema-pruned.ckpt from huggingface and save it in ./checkpoints/
# download advbench from huggingface and save it in ./AdvBench/

# process advbench
python ./process_advbench.py
# generate V-Advbench based on advbech-harmful-behavior
python ./scripts/txt2img.py --from_file ./advbench_prompt.txt --ckpt ./checkpoints/v2-1_768-ema-pruned.ckpt --config ./configs/stable-diffusion/v2-inference-v.yaml --H 512 --W 512 --device cuda --outdir ./AdvBench_Image
```

## 3. Collect MLGuard

```bash
# process VLGuard data
python ./process_VLGuard.py
# generate image description
python ./generate_VLGuard_image_discription.py

python ./generate_audio.py
python ./generate_video.py
```

## 4. SafeMLLM

```bash
# put the SafeMLLM fine-tuned ckpt intu ./PandaGPT/code/ckpt
python ./mllm_generate_responses.py

# put the SafeVLLM fine-tuned ckpt intu ./LLaVA/checkpoints
python ./vllm_generate_responses.py

# evaluation using llama-guard-3
python eval_llama_guard_3.py
# evaluation using safe word list

```


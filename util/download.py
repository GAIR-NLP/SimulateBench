import os
from vllm import LLM, SamplingParams

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

model_root="/data1/ckpts"

HF_TOKEN = "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"

repo="Qwen/Qwen2.5-7B-Instruct"
snapshot_download(repo_id=repo,local_dir = f"{model_root}/{repo}",token=HF_TOKEN,endpoint="https://hf-mirror.com",max_workers=256,resume_download=True)

# model_name = "Meta-Llama-3.1-8B-Instruct"
# model_path = model_root
# model = LLM(model=f"{model_path}/{model_name}")

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "Hello"},
# ]

# results = model.chat(conversation, use_tqdm=False)
# print(results[0].outputs[0].text)

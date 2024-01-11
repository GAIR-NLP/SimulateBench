# 设置代理： export all_proxy=http://127.0.0.1:7890

from huggingface_hub import snapshot_download, login
import os

os.environ['all_proxy'] = 'http://127.0.0.1:7890'

login(token="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")

save_root_path = "/data/yxiao2/huggingface/models"
# model_urls = ["ziqingyang/chinese-alpaca-2-7b"]
# save_paths = ["chinese-alpaca-2/7b"]
# model_names=
model_urls = ["Qwen/Qwen-14B-Chat", "Qwen/Qwen-7B-Chat"]
for model_url in model_urls:
    SNAPSHOT_PATH = snapshot_download(model_url, cache_dir=save_root_path)
    print(f"Downloaded files are located in: {SNAPSHOT_PATH}")

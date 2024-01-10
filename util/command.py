# vicuna-7b-v1.5-16k
# 1.
# python3 -m fastchat.serve.controller --port 21019

# 2.
# CUDA_VISIBLE_DEVICES=1, python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path /data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5-16k/snapshots/9a93d7d11fac7f3f9074510b80092b53bc1a5bec --port 21020 --controller-address http://localhost:21019 --worker-address http://localhost:21020

# 3.
# python3 -m fastchat.serve.openai_api_server --host localhost --port 8031 --controller-address http://localhost:21019

# ----------------------------------------------------------------------------------------------------------------
# longchat-7b-32k-v1.5
# 1.
# python3 -m fastchat.serve.controller --port 21017

# 2.
# CUDA_VISIBLE_DEVICES=2, python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path /data/ckpts/huggingface/models/models--lmsys--longchat-7b-32k-v1.5/snapshots/16deb633ef4d6a18d5750239edc5a85ffeaf3918 --port 21018 --controller-address http://localhost:21017 --worker-address http://localhost:21018
# 3.
# python3 -m fastchat.serve.openai_api_server --host localhost --port 8032 --controller-address http://localhost:21017

# ----------------------------------------------------------------------------------------------------------------
# longchat-13b-16k
# 1.
# python3 -m fastchat.serve.controller --port 21015

# 2.
# CUDA_VISIBLE_DEVICES=3, python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path /data/ckpts/huggingface/models/models--lmsys--longchat-13b-16k/snapshots/70e2e38b82f1e25d8b90b50fbfc2361123bef45f  --port 21016 --controller-address http://localhost:21015 --worker-address http://localhost:21016
# 3.
# python3 -m fastchat.serve.openai_api_server --host localhost --port 8033 --controller-address http://localhost:21015


# ----------------------------------------------------------------------------------------------------------------
# longchat-7b-16k
# 1.
# python3 -m fastchat.serve.controller --port 21013

# 2.
# CUDA_VISIBLE_DEVICES=4, python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path /data/ckpts/huggingface/models/models--lmsys--longchat-7b-16k/snapshots/981bdbd95fcd2098a40982d59a9276abb861147f  --port 21014 --controller-address http://localhost:21013 --worker-address http://localhost:21014
# 3.
# python3 -m fastchat.serve.openai_api_server --host localhost --port 8034 --controller-address http://localhost:21013


# ----------------------------------------------------------------------------------------------------------------
# vicuna-13b-v1.5-16k
# 1.
# python3 -m fastchat.serve.controller --port 21041

# 2.
# CUDA_VISIBLE_DEVICES=5, python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path /data/ckpts/huggingface/models/models--lmsys--vicuna-13b-v1.5-16k/snapshots/277697af19d4b267626ebc9f4e078d19a9a0fddf  --port 21042 --controller-address http://localhost:21041 --worker-address http://localhost:21042
# 3.
# python3 -m fastchat.serve.openai_api_server --host localhost --port 8035 --controller-address http://localhost:21041

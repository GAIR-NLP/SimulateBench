from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

os.environ['HTTP_PROXY'] = '127.0.0.1:7890'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "/data/ckpts/huggingface/models/models--togethercomputer--Llama-2-7B-32K-Instruct/snapshots/35696b9a7ab330dcbe240ff76fb44ab1eccf45bf"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path,
                                             trust_remote_code=False, torch_dtype=torch.float16, device_map="auto")
input_ids = tokenizer.encode("[INST]\nWrite a poem about cats\n[/INST]\n\n", return_tensors="pt")
output = model.generate(input_ids, max_length=1024,
                        temperature=0, repetition_penalty=1.1, top_p=0.7, top_k=50)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

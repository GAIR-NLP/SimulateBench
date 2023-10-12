from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login
model="meta-llama/Llama-2-7b-chat"
login(token="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")
tokenizer = AutoTokenizer.from_pretrained(model, proxies={'https': '127.0.0.1:7890', 'http': '127.0.0.1:7890'})
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    proxies={'https': '127.0.0.1:7890', 'http': '127.0.0.1:7890'}
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    temperature=0.01,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
    proxies={'https': '127.0.0.1:7890', 'http': '127.0.0.1:7890'}
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

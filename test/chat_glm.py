from transformers import AutoTokenizer, AutoModel

path_="/data/ckpts/huggingface/models/models--THUDM--chatglm2-6b-32k/snapshots/455746d4706479a1cbbd07179db39eb2741dc692/"
tokenizer = AutoTokenizer.from_pretrained(path_, trust_remote_code=True)
model = AutoModel.from_pretrained(path_, trust_remote_code=True,device_map="auto").half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

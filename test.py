'''llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY)
text = "How to make a delicious hot chocolate:1. "
llm_results = llm.generate([text])
print(llm_results)'''
'''print(settings_system["OPENAI_API_KEY"])
print(settings_system.max_tokens.gpt_4)
print(settings_person.age)'''
'''print(sys.path)'''
'''from datetime import datetime

now = datetime.now()
print(type(now))
import time

t = time.localtime()
print(type(t))'''

"""a = {'a': 1, 'b': 2}


print('te')"""

import torch
import transformers

"print(transformers.__version__)"
# Use a pipeline as a high-level helper
from transformers import pipeline

from transformers import AutoTokenizer, LlamaForCausalLM

"""model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]"""
"""from transformers import pipeline

# This model is a `zero-shot-classification` model.
# It will classify text, except you are free to choose any label you might imagine
classifier = pipeline(model="facebook/bart-large-mnli")
classifier(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
"""
user=input("what is your name?")
print("hello"+user)
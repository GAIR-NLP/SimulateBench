from person.action.brain.agent import BaseAgent
from person.action.system_setting.system1.chat_template import generate_system_message
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
MODEL_PATH = "/data/ckpts/huggingface/models/models--togethercomputer--Llama-2-7B-32K-Instruct/snapshots/35696b9a7ab330dcbe240ff76fb44ab1eccf45bf"


class Lama27B32K(BaseAgent):
    def __init__(self,profile_version, system_version, model_path=MODEL_PATH, model_name="LLama_2_7B_32K", person_name="monica"):

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=False, torch_dtype=torch.float16,
                                                     device_map="auto")
        super().__init__(model_name=model_name, person_name=person_name, model=model, profile_version=profile_version, system_version=system_version)

        self.system_message = generate_system_message(person_name)[0].content
        self.input_template = f"[INST]\n{self.system_message}"
        self.user_text=f"\n{{user_input}}\n[/INST]\n\n"

    def run(self, input_text):
        input_ = self.input_template+self.user_text.format(user_input=input_text)
        input_ids = self.tokenizer.encode(input_, return_tensors="pt")
        output = self.chat_model.generate(input_ids, max_length=2048, temperature=0, repetition_penalty=1.1, top_p=0.7,
                                          top_k=50)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output_text

    def clear(self):
        pass


if __name__ == "__main__":
    agent = Lama27B32K()
    print(agent.run('tell me a song'))

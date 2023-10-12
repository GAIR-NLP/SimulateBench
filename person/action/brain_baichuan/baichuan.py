import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent

MODEL_PATH = {
    "Baichuan2-13B-Chat": "/data/yxiao2/huggingface/models/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/d022d7264467b2c3bc483e7a3a17105dedba50b8",
    "Baichuan2-7B-Chat": "/data/yxiao2/huggingface/models/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/229e4eb1fab7f6aef90a2344c07085b680487597",
    "Baichuan-13B-Chat": "/data/yxiao2/huggingface/models/models--baichuan-inc--Baichuan-13B-Chat/snapshots/e580bc803f3f4f6b42ddccd0730739c057c7b54c"
}


class BaiChuan(BaseAgent):
    def __init__(self, profile_version, system_version, person_name, model_name):
        model_path = MODEL_PATH[model_name]
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name], use_fast=False,
                                                       trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map="sequential",
                                                     torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH[model_name])
        model.generation_config.temperature = 0.0001
        model.eval()
        super().__init__(person_name=person_name, model=model, model_name=model_name, profile_version=profile_version,
                         system_version=system_version)

        #self.system_message = self.raw_chat_history[0].content
        #print(self.system_message)
        # self.system_message="you will act as homer. your birthday is 1876.2.3. you are a teacher. you have a wife #whose name is maggie."
        self.chat_history.append({"role": "system", "content": self.system_message})

    def run(self, user_input):
        """if len(self.chat_history) == 0:
            user_input = self.system_message + "\n" + user_input"""
        self.chat_history.append({"role": "user", "content": user_input})
        response = self.chat_model.chat(self.tokenizer, self.chat_history)
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def clear(self):
        self.chat_history = [{"role": "system", "content": self.system_message}]
        # self.chat_history = []


if __name__ == "__main__":
    chat_glm = BaiChuan(model_name="Baichuan2-7B-Chat", person_name="rachel", profile_version="profile_v1",
                        system_version="system_v1")
    print(chat_glm.run("what is your birthday?"))
    print(chat_glm.run("what is your name?"))
    print(chat_glm.run("who is your wife?"))
    print(chat_glm.run("who are you?"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent

MODEL_PATH = {
    "XVERSE-13B-Chat": "/data/yxiao2/huggingface/models/models--xverse--XVERSE-13B-Chat/snapshots/4f64cdc8957303f3514e5d75f51536aa549dd49c"
}


class XVerse(BaseAgent):
    def __init__(self, profile_version, system_version, person_name, model_name="XVERSE-13B-Chat"):
        model_path = MODEL_PATH[model_name]
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map=5)
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.temperature = 0.00001
        model = model.eval()

        super().__init__(person_name=person_name, model=model, model_name=model_name, profile_version=profile_version,
                         system_version=system_version)

        self.chat_history.append({"role": "user", "content": self.system_message})

    def run(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})
        response = self.chat_model.chat(self.tokenizer, self.chat_history)
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def clear(self):
        self.chat_history = [{"role": "user", "content": self.system_message}]


if __name__ == "__main__":
    agent = XVerse(profile_version="profile_v1", system_version="system_v1", person_name="homer_african")
    print(agent.run("what is your name?"))
    agent.clear()
    print(agent.run("what is your name?"))
    agent.clear()
    print(agent.run("what is your name?"))

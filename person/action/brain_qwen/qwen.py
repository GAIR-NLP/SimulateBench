from transformers import AutoModelForCausalLM
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent

MODEL_PATH = {
    "Qwen-7B-Chat": "/data/yxiao2/huggingface/models/models--Qwen--Qwen-7B-Chat/snapshots/21ed8f83fe8e900c9b89930bf9a3d2762019beae",
    "Qwen-14B-Chat": "/data/yxiao2/huggingface/models/models--Qwen--Qwen-14B-Chat/snapshots/ce2da0f67e98e106c646f97ce0ffba28e7eb19ff"

}


# MODEL_PATH = "Qwen/Qwen-7B-Chat"


class QWen(BaseAgent):
    def __init__(self, profile_version, system_version, model_name, person_name, device_map=1):
        self.model_path = MODEL_PATH[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=device_map,
                                                     trust_remote_code=True).eval()

        model.generation_config = GenerationConfig.from_pretrained(self.model_path,
                                                                   trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参
        model.generation_config.temperature = 0.00001
        model.generation_config.max_window_size = 7680

        super().__init__(person_name=person_name,
                         model=model, model_name=model_name, profile_version=profile_version,
                         system_version=system_version)

    def run(self, user_input):
        response, history = self.chat_model.chat(
            self.tokenizer,
            query=user_input,
            system=self.system_message,
            history=self.chat_history,
            use_cache=True,
            temperature=0.0001
        )
        self.chat_history = history

        return response

    def clear(self):
        self.chat_history = []


if __name__ == "__main__":
    chat_glm = QWen(profile_version="profile_v1", system_version="system_v1", model_name="Qwen-14B-Chat",
                    person_name="monica")
    print(chat_glm.run("What is your name?"))
    print(chat_glm.run("Could you please share your date of birth?"))
    print(chat_glm.run(
        "Do you have any dependents (e.g., children, elderly parents)?[this is a single-choice question,you should only choose from: Yes;No;There's not enough information to answer this question.]"))

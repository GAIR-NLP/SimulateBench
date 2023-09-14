from transformers import AutoModelForCausalLM
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent

MODEL_PATH = "/data/ckpts/huggingface/models/models--Qwen--Qwen-7B-Chat/snapshots/6463a1ff9fe3b8924b5350d7db00c557c133517e/"


# MODEL_PATH = "Qwen/Qwen-7B-Chat"


class QWen7BChat(BaseAgent):
    def __init__(self,profile_version, system_version, model_name="Qwen-7B-Chat", person_name="monica", model_path=MODEL_PATH):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()

        model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH,
                                                                   trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参

        super().__init__(person_name=person_name, model=model, model_name=model_name, profile_version=profile_version, system_version=system_version)

        # self.system_message = generate_system_message(person_name)[0].content
        self.system_message = F"[INST] <<SYS>>\n{generate_system_message(person_name)[0].content}\n<</SYS>>\n"
        """self.chat_history.append((self.system_message,""))"""

    def run(self, user_input):
        """if len(self.chat_history) == 0:
            user_input = self.system_message + "\nPlease answer blew questions:\n" + user_input
"""
        user_input = self.system_message + user_input + "\n[/INST]\n"
        response, history = self.chat_model.chat(
            self.tokenizer,
            user_input,
            history=self.chat_history,
            use_cache=True,
            temperature=0.0001,
            max_new_tokens=30,
        )
        self.chat_history = history

        return response

    def clear(self):
        self.chat_history = []


if __name__ == "__main__":
    chat_glm = QWen7BChat()
    print(chat_glm.run("What is your name?"))
    print(chat_glm.run("Could you please share your date of birth?"))
    print(chat_glm.run("Do you have any dependents (e.g., children, elderly parents)?[this is a single-choice question,you should only choose from: Yes;No;There's not enough information to answer this question.]"))

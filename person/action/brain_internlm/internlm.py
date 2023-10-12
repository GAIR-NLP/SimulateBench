from transformers import AutoTokenizer, AutoModelForCausalLM
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent

MODEL_PATH = {
    "internlm-chat-20b": "/data/yxiao2/huggingface/models/models--internlm--internlm-chat-20b/snapshots/bb65b6c4c3f0ccfcb5fe0fc1558e97a19b1de364",
    "internlm-chat-7b-8k": "/data/yxiao2/huggingface/models/models--internlm--internlm-chat-7b-8k/snapshots/b6075f6f980b46373b23aade2b2dc136b9d575d2"
}


class InternLm(BaseAgent):
    def __init__(self, profile_version, system_version, person_name, model_name):
        model_path = MODEL_PATH[model_name]
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name], trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[model_name], trust_remote_code=True).cuda()

        model = model.eval()
        super().__init__(
            person_name=person_name, model=model, model_name=model_name, profile_version=profile_version,
            system_version=system_version)

    def run(self, user_input):
        if len(self.chat_history) == 0:
            user_input = self.system_message + "\n" + user_input

        output, history = self.chat_model.chat(self.tokenizer, query=user_input, history=self.chat_history,
                                               temperature=0.00001)
        self.chat_history = history
        return output

    def clear(self):
        self.chat_history = []


if __name__ == "__main__":
    chat_glm = InternLm(model_name="internlm-chat-20b", person_name="walter_white", profile_version="profile_v1",
                        system_version="system_v1")
    print(chat_glm.run("what is your birthday?"))
    chat_glm.clear()
    print(chat_glm.run("what is your name?"))
    chat_glm.clear()

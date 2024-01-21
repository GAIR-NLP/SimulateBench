from transformers import AutoTokenizer, AutoModel

from person.action.system_setting.system1.chat_template import generate_system_message

from person.action.brain.agent import BaseAgent
# from accelerate import infer_auto_device_map


MODEL_PATH = {
    "chatglm2-6b-32k": "/data/ckpts/huggingface/models/models--THUDM--chatglm2-6b-32k/snapshots/455746d4706479a1cbbd07179db39eb2741dc692/",
    "chatglm2-6b": "/data/yxiao2/huggingface/models/models--THUDM--chatglm2-6b/snapshots/8fd7fba285f7171d3ae7ea3b35c53b6340501ed1"
}


class ChatGLM2(BaseAgent):
    def __init__(self, profile_version, system_version, person_name, model_name="chatglm2-6b-32k"):
        model_path = MODEL_PATH[model_name]
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        model = AutoModel.from_pretrained(self.model_path, device_map=3, trust_remote_code=True,
                                          ).half().cuda()

        model.eval()
        super().__init__(person_name=person_name, model=model, model_name=model_name, profile_version=profile_version,
                         system_version=system_version)

    def run(self, user_input):
        if len(self.chat_history) == 0:
            user_input = self.system_message + "\n" + user_input

        response, history = self.chat_model.chat(self.tokenizer, user_input, history=self.chat_history,
                                                 temperature=0.001)
        self.chat_history = history
        return response

    def clear(self):
        self.chat_history = []


if __name__ == "__main__":
    chat_glm = ChatGLM2(model_name="chatglm2-6b", person_name="walter_white", profile_version="profile_v1",
                        system_version="system_v1")
    print(chat_glm.run("who are you?"))

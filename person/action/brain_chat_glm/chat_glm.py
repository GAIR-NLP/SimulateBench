from transformers import AutoTokenizer, AutoModel
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent
from accelerate import infer_auto_device_map


MODEL_PATH = "/data/ckpts/huggingface/models/models--THUDM--chatglm2-6b-32k/snapshots/455746d4706479a1cbbd07179db39eb2741dc692/"


class ChatGLM26B32K(BaseAgent):
    def __init__(self, profile_version, system_version,model_name="chatglm2-6b-32k", person_name="monica", model_path=MODEL_PATH):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True,device_map="sequential" ).half().cuda()

        model.eval()
        super().__init__(person_name=person_name, model=model, model_name=model_name, profile_version=profile_version, system_version=system_version)

        self.system_message = generate_system_message(person_name)[0].content

    def run(self, user_input):
        if len(self.chat_history) == 0:
            user_input = self.system_message + "\n" + user_input

        response, history = self.chat_model.chat(self.tokenizer, user_input, history=self.chat_history)
        self.chat_history = history
        return response

    def clear(self):
        self.chat_history = []


if __name__ == "__main__":
    chat_glm = ChatGLM26B32K()
    print(chat_glm.run("who are you?"))

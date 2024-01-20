import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent
import requests
import json


def do_request(character_info, message_history, character_name, model_name="Baichuan-NPC-Turbo"):
    url = "https://api.baichuan-ai.com/v1/chat/completions"
    api_key = "sk-5b2182dce1ce219f45edcd9a4e1e7eea"
    data = {
        "model": model_name,
        "character_profile": {
            "character_name": character_name,
            "character_info": character_info,
            "user_name": "andy",
            "user_info": f"a friend of {character_name}",
        },
        "messages": message_history,
        "temperature": 0.001,
        "top_k": 10,
        "max_tokens": 512,
        "stream": True
    }

    json_data = json.dumps(data)

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    response = requests.post(url, data=json_data, headers=headers, timeout=60)

    if response.status_code == 200:
        print("请求成功！")
        # print("响应body:", response.text)
        print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        return response.text.choices[0].message.content
    else:
        print("请求失败，状态码:", response.status_code)
        # print("请求失败，body:", response.text)
        print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        return None


class BaiChuanCharacter(BaseAgent):
    def __init__(self, profile_version, system_version, person_name, model_name):
        model = None
        super().__init__(person_name=person_name, model=model, model_name=model_name, profile_version=profile_version,
                         system_version=system_version)

        # self.chat_history.append({"role": "system", "content": self.system_message})
        self.chat_history = []

    def run(self, user_input):
        """if len(self.chat_history) == 0:
            user_input = self.system_message + "\n" + user_input"""
        self.chat_history.append({"role": "user", "content": user_input})
        response = do_request(character_info=self.system_message, message_history=self.chat_history,
                              character_name=self.person_name, model_name=self.model_name)
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def clear(self):
        self.chat_history = []
        # self.chat_history = []


if __name__ == "__main__":
    chat_baichuan_character = BaiChuanCharacter(model_name="Baichuan-NPC-Turbo",
                                                person_name="homer", profile_version="profile_v1",
                                                system_version="system_v1")
    print(chat_baichuan_character.run("what is your birthday?"))
    chat_baichuan_character.clear()

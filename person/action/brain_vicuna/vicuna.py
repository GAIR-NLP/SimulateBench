from person.action.brain.agent import BaseAgent
from person.action.system_setting.system1.chat_template import generate_human_message
import openai
from config.config import settings_system
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage, BaseMessage
)
from typing import List

openai.api_key = "EMPTY"  # supply your API key however you choose
openai.api_base = "http://localhost:8001/v1"


class Vicuna(BaseAgent):
    def __init__(self, profile_version, system_version, person_name, model_name="vicuna-7b-v1.5-16k"):
        temperature = settings_system['temperature']

        if "vicuna-7b-v1.5-16k" == model_name:
            openai_api_base = "http://localhost:8031/v1"
        elif "longchat-7b-32k-v1.5" == model_name:
            openai_api_base = "http://localhost:8032/v1"
        elif "longchat-13b-16k" == model_name:
            openai_api_base = "http://localhost:8033/v1"
        elif "longchat-7b-16k" == model_name:
            openai_api_base = "http://localhost:8034/v1"
        elif "vicuna-13b-v1.5-16k" == model_name:
            openai_api_base = "http://localhost:8035/v1"
        else:
            openai_api_base = "http://localhost:8001/v1"

        model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature,
            openai_api_key="EMPTY",
            openai_api_base=openai_api_base
        )
        super().__init__(person_name=person_name, model_name=model_name, model=model,

                         profile_version=profile_version, system_version=system_version)

        self.chat_history: List[BaseMessage] = [self.system_message_langchain]

    def clear(self):
        self.chat_history = [self.system_message_langchain]

    def run(self, user_input):
        user_message = generate_human_message(user_input)
        self.chat_history.extend(user_message)
        results = self.chat_model.generate([self.chat_history])

        self.chat_history.append(results.generations[0][0].message)

        return results.generations[0][0].text


if __name__ == "__main__":
    pass

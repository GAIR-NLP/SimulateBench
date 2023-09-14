from person.action.brain.agent import BaseAgent
from person.action.system_setting.system1.chat_template import generate_system_message, generate_human_message
import openai
from config.config import settings_system
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)

openai.api_key = "EMPTY"  # supply your API key however you choose
openai.api_base = "http://localhost:8000/v1"


class Vicuna(BaseAgent):
    def __init__(self, profile_version, system_version, person_name, model_name="longchat-7b-32k"):
        temperature = settings_system['temperature']
        model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1"
        )
        super().__init__(person_name=person_name, model_name=model_name, model=model,

                         profile_version=profile_version, system_version=system_version)

        self.chat_history = generate_system_message(person_name=person_name)

    def clear(self):
        self.chat_history = generate_system_message(person_name=self.person_name)

    def run(self, user_input):
        user_message = generate_human_message(user_input)
        self.chat_history.extend(user_message)
        results = self.chat_model.generate([self.chat_history])

        self.chat_history.append(results.generations[0][0].message)

        return results.generations[0][0].text


if __name__ == "__main__":
    pass

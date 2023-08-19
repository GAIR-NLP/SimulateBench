from langchain.chat_models import ChatOpenAI
from GPTMan.config.config import settings_system
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
)
from langchain.chat_models import ChatAnthropic
from GPTMan.log.record_cost import record_cost_gpt

import openai

openai.api_base = "http://openai.plms.ai/v1"

MODEL_NAME = settings_system['model_name']
TEMPERATURE = settings_system['temperature']
OPENAI_API_KEY = settings_system.OPENAI_API_KEY


def generate_open_ai_chat_model(model_name=MODEL_NAME, temperature=TEMPERATURE):
    chat_model = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY, temperature=temperature)
    return chat_model


def generate_Anthropic_chat_model(model_name=MODEL_NAME, temperature=TEMPERATURE):
    chat_model = ChatAnthropic(model_name=model_name, temperature=temperature)
    return chat_model


class OpenAI:
    def __init__(self, system_template="None", model_name=MODEL_NAME, temperature=TEMPERATURE):
        self.chat_model = generate_open_ai_chat_model(model_name=model_name, temperature=temperature)

        self.history = []
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        system_message = system_prompt.format_messages()
        self.history.extend(system_message)

        # human_prompt = HumanMessagePromptTemplate.from_template(human_message)
        # human_message = human_prompt.format_messages()
        # self.history.extend(human_message)
        self.total_tokens_prompt = 0
        self.total_tokens_output = 0

        self.model_name = model_name

    def run(self, user_input):
        human_prompt = HumanMessagePromptTemplate.from_template(user_input)
        human_message = human_prompt.format_messages()
        self.history.extend(human_message)
        results = self.chat_model.generate([self.history])
        token_usage_prompt = results.llm_output['token_usage']['prompt_tokens']
        token_usage_completion = results.llm_output['token_usage']['completion_tokens']
        self.total_tokens_output += token_usage_completion
        self.total_tokens_prompt += token_usage_prompt
        self.history.append(results.generations[0][0].message)

        return results.generations[0][0].text

    def clear_message(self):
        self.history = self.history[0:1]
        record_cost_gpt(self.model_name, self.total_tokens_prompt, self.total_tokens_output)
        self.total_tokens_prompt = 0
        self.total_tokens_output = 0

    def close(self):
        self.clear_message()


if __name__ == "__main__":
    system_plate = \
        "You are an helpful AI. You are helping to provide information about characters in the TV show of Friends. "
    openai = OpenAI(system_template=system_plate)
    # for habits,behavior,
    # question = "what is the habits/traditions/behavior pattern between Monica and Richard?"
    # for routines
    # question="what is the routines of Monica and Richard?"
    # for characteristics,
    # question = "what is the quality or a mental state of Monica as a romantic partner of Richard?"
    # for goal,
    question = "what is the goal of Monica and Richard?"
    # for familiarity, question = "How long have monica known her mother?<choose from - less than a year;1-2 years;3-5 years;more than 5 years>"

    # for judgement,
    # question = "what judgements does monica make about Richard?"
    # for affection,
    # question="what affections does monica have for Richard?"
    print(openai.run(question))
    openai.close()

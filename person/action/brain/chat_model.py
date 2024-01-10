from langchain.chat_models import ChatOpenAI
from config.config import settings_system
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage
from log.record_cost import record_cost_gpt

"""from benchmark.benchmark_template_question import QuestionGenerator"""

MODEL_NAME = settings_system['model_name']
TEMPERATURE = settings_system['temperature']
# OPENAI_API_KEY = settings_system.OPENAI_API_KEY_TB
OPENAI_API_KEY = settings_system.peiqi
#OPENAI_API_KEY = settings_system.chunpu


# openai_api_base="http://openai.plms.ai/v1"  using it in china's server
# openai_api_base="https://api.chatanywhere.com.cn/v1"
def generate_open_ai_chat_model(model_name=MODEL_NAME, temperature=TEMPERATURE):
    chat_model = ChatOpenAI(
        model_name=model_name,
        openai_api_key=OPENAI_API_KEY,
        temperature=temperature,
        openai_api_base="https://api.chatanywhere.com.cn/v1"
    )

    return chat_model


class OpenAI:
    def __init__(self, system_template="None", model_name=MODEL_NAME, temperature=TEMPERATURE):
        self.chat_model = generate_open_ai_chat_model(model_name=model_name, temperature=temperature)

        self.history = []
        # system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        # system_message = system_prompt.format_messages()
        system_message = SystemMessage(content=system_template)

        self.history.append(system_message)

        # human_prompt = HumanMessagePromptTemplate.from_template(human_message)
        # human_message = human_prompt.format_messages()
        # self.history.extend(human_message)
        self.total_tokens_prompt = 0
        self.total_tokens_output = 0

        self.model_name = model_name

    def run(self, user_input):
        # human_prompt = HumanMessagePromptTemplate.from_template("{text}")
        human_message = HumanMessage(content=user_input)

        self.history.append(human_message)
        results = self.chat_model.generate([self.history])
        token_usage_prompt = results.llm_output['token_usage']['prompt_tokens']
        token_usage_completion = results.llm_output['token_usage']['completion_tokens']
        self.total_tokens_output += token_usage_completion
        self.total_tokens_prompt += token_usage_prompt
        self.history.append(results.generations[0][0].message)

        return results.generations[0][0].text

    def clear(self):
        self.history = self.history[0:1]
        record_cost_gpt(self.model_name, self.total_tokens_prompt, self.total_tokens_output)
        self.total_tokens_prompt = 0
        self.total_tokens_output = 0

    def close(self):
        self.clear()


if __name__ == "__main__":
    """"generator = QuestionGenerator()
    sys_template, user_input = generator.generator_basic_information_v2()

    openai = OpenAI(system_template=sys_template)
    result = openai.run(user_input)
    print(result)

    openai.close()"""

    """
    gpt-4
As of my knowledge update in October 2021, Amazon has not released a detailed plot for "The Lord of the Rings: The Rings of Power" TV series. The series is set in the Second Age of Middle-earth, thousands of years before the events of J.R.R. Tolkien's "The Lord of the Rings" books. It will reportedly explore new storylines preceding Tolkien's "The Fellowship of the Ring." For the most accurate and up-to-date information, please refer to official announcements from Amazon.
    """
    """open_ai = OpenAI()
    print(open_ai.model_name)
    print(open_ai.run("do you know the detailed plot of  The Lord of the Rings: The Rings of Power tv series?"))
"""

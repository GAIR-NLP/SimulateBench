from typing import List

from langchain.schema import SystemMessage, BaseMessage

from person.action.system_setting.system1.chat_template import generate_human_message
from person.action.system_setting.profile_for_agent import ProfileSystem
from person.action.brain.chat_model import generate_open_ai_chat_model
from person.action.brain.chat_model import MODEL_NAME
from person.profile.profile import load_name
from log.record_cost import record_cost_gpt


class BaseAgent:
    """_summary_"""

    def __init__(self, person_name, model, model_name, profile_version, system_version):
        self.chat_model = model
        self.person_name = person_name
        self.name = load_name(person_name=person_name)
        self.model_name = model_name

        # profile version and system version, in case of the profile and system change
        self.profile_version = profile_version
        self.system_version = system_version

        profile_system_obj = ProfileSystem(
            person_name=person_name,
            profile_version=profile_version,
            system_version=system_version,
        )
        self.system_message = profile_system_obj.generate_system_message()[0].content
        self.system_message_langchain = SystemMessage(content=self.system_message)
        self.chat_history = []

    def run(self, user_input: str):
        """_summary_

        Args:
            user_input (str): _description_
        """
        pass

    def clear(self):
        """_summary_"""
        pass


class Agent(BaseAgent):
    """_summary_

    Args:
        BaseAgent (_type_): _description_
    """

    def __init__(
        self, person_name, profile_version, system_version, model_name=MODEL_NAME
    ):
        model = generate_open_ai_chat_model(model_name=model_name)
        super().__init__(
            person_name=person_name,
            model=model,
            model_name=model_name,
            profile_version=profile_version,
            system_version=system_version,
        )

        self.chat_history: List[BaseMessage] = [
            self.system_message_langchain
        ]  # [system,example,human,ai,system,example]

        self.total_tokens_prompt = 0
        self.total_tokens_output = 0

    def run(self, user_input: str):
        """_summary_

        Args:
            user_input (str): _description_

        Returns:
            _type_: _description_
        """
        user_message = generate_human_message(user_input)
        self.chat_history.extend(user_message)
        results = self.chat_model.generate([self.chat_history])
        token_usage_prompt = results.llm_output["token_usage"]["prompt_tokens"]
        token_usage_completion = results.llm_output["token_usage"]["completion_tokens"]
        self.total_tokens_output += token_usage_completion
        self.total_tokens_prompt += token_usage_prompt
        self.chat_history.append(results.generations[0][0].message)

        return results.generations[0][0].text

    def clear(self):
        """_summary_"""
        self.chat_history = [self.system_message_langchain]
        record_cost_gpt(
            self.model_name, self.total_tokens_prompt, self.total_tokens_output
        )
        self.total_tokens_prompt = 0
        self.total_tokens_output = 0


class Agent_self_consistency(BaseAgent):
    """_summary_

    Args:
        BaseAgent (_type_): _description_
    """

    def __init__(
        self, person_name, profile_version, system_version, model_name=MODEL_NAME
    ):
        model = generate_open_ai_chat_model(model_name=model_name)
        super().__init__(
            person_name=person_name,
            model=model,
            model_name=model_name,
            profile_version=profile_version,
            system_version=system_version,
        )

        self.chat_history: List[BaseMessage] = [
            self.system_message_langchain
        ]  # [system,example,human,ai,system,example]

        self.total_tokens_prompt = 0
        self.total_tokens_output = 0

    def run(self, user_input: str):
        user_message = generate_human_message(user_input)
        self.chat_history.extend(user_message)
        results = self.chat_model.generate([self.chat_history])
        token_usage_prompt = results.llm_output["token_usage"]["prompt_tokens"]
        token_usage_completion = results.llm_output["token_usage"]["completion_tokens"]
        self.total_tokens_output += token_usage_completion
        self.total_tokens_prompt += token_usage_prompt
        self.chat_history.append(results.generations[0][0].message)

        return results.generations[0][0].text

    def clear(self):
        self.chat_history = [self.system_message_langchain]
        record_cost_gpt(
            self.model_name, self.total_tokens_prompt, self.total_tokens_output
        )
        self.total_tokens_prompt = 0
        self.total_tokens_output = 0


if __name__ == "__main__":
    PERSON_NAME = "monica"
    NAME = load_name(person_name=PERSON_NAME)
    agent = Agent(
        person_name=PERSON_NAME,
        profile_version="profile_v1",
        system_version="system_v1",
    )
    print(
        f"hello, I am {NAME}, what can I do for you? If you want to quit, please input 'quit' or 'q'\n{'-' * 10}"
    )
    user_input_ = input("user:")
    try:
        while user_input_ != "quit" and user_input_ != "q":
            print(agent.run(user_input_))
            user_input_ = input("user:\n")

    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        print("bye")
        agent.clear()

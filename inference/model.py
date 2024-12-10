from typing import List

from vllm import LLM, SamplingParams

from person.action.system_setting.profile_for_agent import ProfileSystem
from person.profile.profile import load_name
from config.config import settings_system


model_root = "/data1/ckpts"


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
        self.system_message = profile_system_obj.generate_system_message()

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


class Llama(BaseAgent):
    """_summary_

    Args:
        BaseAgent (_type_): _description_
    """

    def __init__(
        self,
        profile_version,
        system_version,
        person_name,
        model_name="Meta-Llama-3.1-8B-Instruct",
    ):
        temperature = settings_system["temperature"]
        self.sampling_params = SamplingParams(temperature=temperature)

        model_path = model_root
        model = LLM(model=f"{model_path}/{model_name}")
        super().__init__(
            person_name=person_name,
            model_name=model_name,
            model=model,
            profile_version=profile_version,
            system_version=system_version,
        )

        self.chat_history = [{"role": "system", "content": self.system_message}]

    def clear(self):
        self.chat_history = [{"role": "system", "content": self.system_message}]

    def run(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})
        results = self.chat_model.chat(
            self.chat_history, sampling_params=self.sampling_params, use_tqdm=False
        )

        self.chat_history.append(results[0].outputs[0].text)

        return results[0].outputs[0].text


if __name__ == "__main__":
    pass

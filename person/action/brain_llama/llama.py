from typing import List

from vllm import LLM, SamplingParams

from person.action.brain.agent import BaseAgent
from person.action.system_setting.system1.chat_template import generate_human_message
from config.config import settings_system
from util.download import model_root

class Llama(BaseAgent):
    """_summary_

    Args:
        BaseAgent (_type_): _description_
    """
    def __init__(self, profile_version, system_version, person_name, model_name="Meta-Llama-3.1-8B-Instruct"):
        temperature = settings_system['temperature']
        self.sampling_params = SamplingParams(temperature=temperature)

        model_path=model_root
        model = LLM(model=f"{model_path}/{model_name}")
        super().__init__(person_name=person_name, model_name=model_name, model=model,

                         profile_version=profile_version, system_version=system_version)

        self.chat_history = [{
            "role": "system",
            "content":self.system_message_langchain.content}]

    def clear(self):
        self.chat_history = [self.system_message_langchain]

    def run(self, user_input):
        self.chat_history.extend({
            "role": "user",
            "content": user_input
        })
        results = self.chat_model.chat(self.chat_history,
                   sampling_params=self.sampling_params,
                   use_tqdm=False)

        self.chat_history.append(results[0].outputs[0].text)

        return results[0].outputs[0].text


if __name__ == "__main__":
    pass

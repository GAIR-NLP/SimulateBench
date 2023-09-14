from langchain.schema import BaseMessage
from langchain.prompts import load_prompt
from util.file_util import load_json_file
from langchain.prompts.chat import SystemMessagePromptTemplate

ROOT_PATH = "/home/yxiao2/pycharm/GPTMan/db"
SYSTEM_PATH = f"{ROOT_PATH}/agent_system_prompt/{{system_version}}.json"


class Profile:
    def __init__(self, profile_version="profile_v1"):
        self.profile_version = profile_version

    def load_basic_information(self):
        pass

    def load_roles(self):
        pass

    def load_examples(self):
        pass

    def load_name(self):
        pass


class System:
    def __init__(self, system_version="system_v1"):
        self.system_version = system_version

    def load_sys_template(self):
        """
        load system prompt which is used to generate system message.

        Returns:

        """
        path_ = SYSTEM_PATH.format(system_version=self.system_version)
        prompt_obj = load_json_file(path_)
        prompt_template = prompt_obj["SYSTEM_TEMPLATE"]
        role_explain = prompt_obj["ROLES_EXPLAIN"]

        prompt_ = SystemMessagePromptTemplate.from_template(prompt_template)

        return prompt_, role_explain


class ProfileSystem:
    def __init__(self, profile_version="profile_v1", system_version="system_v1"):
        self.profile_version = profile_version
        self.system_version = system_version

    def generate_system_message(self) -> BaseMessage:
        """
        Generate system message, according to the version of profile and system prompt.

        Returns:
            BaseMessage: langchain class of message

        """
        pass


if __name__ == "__main__":
    test = System()
    prompt,_ = test.load_sys_template()
    print(prompt)

from langchain.schema import BaseMessage
from langchain.prompts import load_prompt
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    BaseMessage,
)
from person.profile.base_data_class import load_person_files
from util.file_util import load_json_file
from langchain.prompts.chat import SystemMessagePromptTemplate
from person.profile.basic_information import load_basic_information
from person.profile.role import load_social_personas
from person.profile.profile import load_name
from typing import List
import json
from util.format_json import delete_none
from person.action.system_setting.system1.chat_template import SYS_TEMPLATE, ROLES_EXPLAIN, HUMAN_TEMPLATE

ROOT_PATH = "/home/yxiao2/pycharm/GPTMan/db"
SYSTEM_PATH = f"{ROOT_PATH}/agent_system_prompt/{{system_version}}.json"


class Profile:
    def __init__(self, profile_version, person_name):
        """

        Args:
            profile_version: such as profile_v1 or profile_v1_only_roles
            person_name:
        """
        self.profile_version = profile_version
        self.person_name = person_name

        self.only_roles = False
        self.only_roles_and_pad_with_white_space = False

        if "_pad_basic_with_white_space" in profile_version:
            self.profile_version = profile_version[:-27]
            self.only_roles_and_pad_with_white_space = True

        if "_only_roles" in profile_version:
            self.profile_version = profile_version[:-11]
            self.only_roles = True

    def load_basic_information(self):
        path_ = f"{ROOT_PATH}/profile/{self.person_name}/{self.profile_version}/basic_information.json"
        with open(path_, 'r', encoding="utf-8") as f:
            data_ = json.load(f)
            data__ = delete_none(data_)
            return json.dumps(data__, separators=(',', ':'), indent=None)

    def load_roles(self):
        path_ = f"{ROOT_PATH}/profile/{self.person_name}/{self.profile_version}/roles.json"
        with open(path_, 'r', encoding="utf-8") as f:
            data_ = json.load(f)
            data__ = delete_none(data_)
            return json.dumps(data__, separators=(',', ':'), indent=None)

    def load_examples(self):
        results = ''
        json_path = load_person_files(self.person_name)['examples_path']
        with open(json_path, 'r', encoding="utf-8") as f:
            examples = json.load(f)
        for _example in examples['examples']:
            requirement = _example['requirement']
            if '{person_name}' in requirement:
                requirement = requirement.replace('{person_name}', self.person_name)

            results += requirement + '\n' + 'some examples are below:\n'

            if 'examples' not in _example.keys():
                results += 'no examples\n'
                continue
            for example in _example['examples']:
                results += example + '\n'

        return results

    def load_name(self):
        path_ = f"{ROOT_PATH}/profile/{self.person_name}/{self.profile_version}/basic_information.json"
        with open(path_, 'r', encoding="utf-8") as f:
            data_ = json.load(f)
            data__ = delete_none(data_)
            return data__["basic_information"]["name"]

    def load_profile(self):
        name = self.load_name()
        basic_information_len = len(self.load_basic_information())
        if self.only_roles_and_pad_with_white_space:
            basic_information = ' ' * (basic_information_len - 10 - len(name)) + '\n{' + f'"name":{name}' + '}'
        elif self.only_roles:
            basic_information = f'"name":{name}'
        else:
            basic_information = self.load_basic_information()

        examples = self.load_examples()
        return {
            "person_name": name,
            "basic_information": basic_information,
            "profile_format": "json",
            "roles": self.load_roles(),
            "examples_and_explain": examples,
        }


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
    def __init__(self, person_name, profile_version, system_version):
        """

        Args:
            person_name:
            profile_version: such as profile_v1 or profile_v1_only_roles
            system_version: such as system_v1
        """
        self.profile_version = profile_version
        self.system_version = system_version
        self.person_name = person_name

        self.profile_obj = Profile(profile_version, person_name)
        self.system_obj = System(system_version)

    def generate_system_message(self) -> List[BaseMessage]:
        character_information = self.profile_obj.load_profile()
        system_message_prompt, role_explain = self.system_obj.load_sys_template()
        return system_message_prompt.format_messages(
            person=character_information["person_name"],
            basic_information=character_information["basic_information"],
            profile_format=character_information["profile_format"],
            roles=character_information['roles'],
            role_explain=role_explain,
            examples_and_explain=character_information["examples_and_explain"]
        )


if __name__ == "__main__":
    obj_ = ProfileSystem(system_version="system_change_position_basic_and_relation", profile_version="profile_v1_pad_basic_with_white_space",
                         person_name="homer")
    system_content = obj_.generate_system_message()[0].content
    print(system_content)

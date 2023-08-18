from dataclasses import dataclass
from typing import List, Union
from GPTMan.person.profile.basic_information import BasicInformation, load_basic_information
from GPTMan.person.profile.role import SocialPersona, load_social_personas
from GPTMan.person.profile.base_data_class import load_person_files
import json
from GPTMan.util.format_json import delete_none


@dataclass
class Profile:
    basic_information: BasicInformation
    social_personas: List[SocialPersona]

    def find_persona(self, description: str) -> (bool, SocialPersona):
        """
        :param description: find the persona that meet with the description
        :return:
        """
        pass

    def add_persona(self, persona: SocialPersona):
        self.social_personas.append(persona)


def load_profile(person_name: str = "monica", pure_str=True) -> Union[Profile, None, str]:
    """
    :param person_name:
    :param pure_str: if True, return json string, else return Profile object
    :return: Profile object or json string
    """
    json_file = load_person_files(person_name)['profile_path']
    with open(json_file, 'r') as f:
        data_ = json.load(f)
        if pure_str:
            data__ = delete_none(data_)
            return json.dumps(data__, separators=(',', ':'), indent=None)
        else:
            basic_information = load_basic_information(data_['basic_information'])
            social_personas = load_social_personas(data_['roles'])
            results = Profile(basic_information, social_personas)
            return results


def load_name(person_name: str = 'monica') -> str:
    json_file= load_person_files(person_name)['basic_information_path']
    with open(json_file, 'r') as f:
        data_ = json.load(f)
        return data_['basic_information']['name']


if __name__ == '__main__':
    profile = load_profile(pure_str=True)

    print(profile)

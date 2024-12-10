"""_summary_

    Returns:
        _type_: _description_
"""

import json

# from person.profile.basic_information import load_basic_information
from person.profile.base_data_class import load_person_files
from util.format_json import delete_none


# @dataclass
# class Profile:
#     """_summary_"""

#     basic_information: BasicInformation
#     social_personas: List[SocialPersona]

#     def find_persona(self, description: str) -> tuple[bool, SocialPersona]:
#         """
#         :param description: find the persona that meet with the description
#         :return:
#         """
#         pass

#     def add_persona(self, persona: SocialPersona):
#         self.social_personas.append(persona)


def load_profile(person_name: str, pure_str=True) -> str:
    """
    :param person_name:
    :param pure_str: if True, return json string, else return Profile object
    :return: Profile object or json string
    """
    json_file = load_person_files(person_name)["profile_path"]
    with open(json_file, "r", encoding="utf-8") as f:
        data_ = json.load(f)
        if pure_str:
            data__ = delete_none(data_)
            return json.dumps(data__, separators=(",", ":"), indent=None)
        else:
            raise NotImplementedError


def load_name(person_name: str) -> str:
    """_summary_

    Args:
        person_name (str): _description_

    Returns:
        str: _description_
    """
    json_file = load_person_files(person_name)["basic_information_path"]
    with open(json_file, "r", encoding="utf-8") as f:
        data_ = json.load(f)
        return data_["basic_information"]["name"]


if __name__ == "__main__":
    pass

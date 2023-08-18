from dataclasses import dataclass
from typing import List, Union
from datetime import datetime
from GPTMan.config.config import settings_system


def load_person_files(person_name: str):
    """
    load the person files path from config file
    :param person_name:
    :return:
    """

    return {
        'profile_path': settings_system[person_name]['profile_path'],
        'basic_information_path': settings_system[person_name]['basic_information_path'],
        'roles_path': settings_system[person_name]['roles_path'],
        'examples_path': settings_system[person_name]['examples_path']
    }


@dataclass(frozen=True)
class Person:
    name: str


@dataclass(frozen=True)
class Description:
    description: str
    time: str = None


@dataclass(frozen=True)
class Example:
    """
    For some actual examples for personality and habit
    """
    example: str
    time: str = None


@dataclass
class DescriptionAndExample:
    """
    For some data type that contains description and example, like personality and habit.
    """
    description: Union[Description, List[Description]]
    examples: List[Example]


@dataclass
class DescriptionAndExamples:
    description_and_examples: List[DescriptionAndExample]

    def update(self):
        pass


def load_description_and_examples(json_list: Union[None, List[dict]]) -> \
        Union[DescriptionAndExamples, None]:
    """
    load the data from a basic data class to the current data class
    :param json_list:
    :return:
    """
    if json_list is None:
        return None

    results = []
    for json_object in json_list:
        description = Description(**json_object['description'])
        examples = []
        for example in json_object['examples']:
            if example is not None:
                examples.append(Example(**example))
        results.append(DescriptionAndExample(description=description, examples=examples))


@dataclass
class Descriptions:
    """
    The descriptions that will be deleted and added by checking the conflict
    """
    descriptions: List[Description]

    def update(self, descriptions: List[Description]):
        for description in descriptions:
            check_result = self.conflict(description)
            if not check_result[0]:
                self.add(description)
            else:
                self.delete(check_result[1])
                self.add(description)
        self.descriptions.extend(descriptions)

    def add(self, description: Description):
        self.descriptions.append(description)

    def delete(self, description: Description):
        self.descriptions.remove(description)

    def conflict(self, description: Description) -> (bool, Description):
        """
        :param description: check the conflict with the current description
        :return: (bool,Description) if conflicted, return True and the description that need to be deleted
        """

        pass


@dataclass
class DescriptionsFrozen:
    """"
    The descriptions that will not be deleted, but onlu added
    [experiences ]
    """
    descriptions: List[Description]

    def update(self, descriptions: List[Description]):
        self.descriptions.extend(descriptions)


def load_descriptions_variant(json_list: Union[None, List[dict]], target_class):
    """
    load the data from a basic data class to the current data class
    :param target_class:
    :param json_list:
    :return:
    """
    if json_list is None:
        return None

    results = []
    for json_object in json_list:
        results.append(Description(**json_object))
    return target_class(descriptions=results)


@dataclass(frozen=True)
class Persona:
    description: str

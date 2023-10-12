import json
from dataclasses import dataclass
from typing import List, Union
from datetime import datetime
from person.profile.base_data_class import Person, DescriptionAndExamples, Descriptions,  \
    load_description_and_examples, load_descriptions_variant, load_person_files
from util.format_json import delete_none


@dataclass
class Nickname:
    nickname: str
    people_use_the_name_exact: List[Person]
    people_use_the_name_range: List[str]


@dataclass
class Nicknames:
    nicknames: List[Nickname]

    def find_nickname_by_person(self, person):
        results = []
        for nickname in self.nicknames:
            if self.check_person_use_nicknames(person, nickname):
                results.append(nickname.nickname)
        return results

    def add_nickname(self, nickname):
        self.nicknames.append(nickname)

    def check_person_use_nicknames(self, person, nickname: Nickname) -> bool:
        """
        consider two kinds of people_use_nicknames: Person and str

        if the person is Person type it is easy, just exact check
        if the person is str type, just like everyone or in high school, then it must be processed carefully.

        if the person is in the people_use_nicknames, return True
        else return False
        :param nickname:
        :param person:
        :return:
        """
        pass

    '''def update_nickname_by_person(self,person, nickname):
        for nick in self.nicknames:
            if nick.person_give_the_name == person:
                nick.nickname = nickname'''


@dataclass
class Education:
    school: str
    major: str
    degree: str
    start_time: str
    end_time: str


@dataclass
class BasicInformation:
    name: Person
    gender: str
    home: str
    nicknames: Nicknames
    date_of_birth: str
    educations: List[Education]
    race: str
    personality: DescriptionAndExamples
    catchphrase: Descriptions
    habits: DescriptionAndExamples


def load_basic_information(person_name: str , pure_str: bool = True) -> Union[BasicInformation, None, str]:
    """
    load the basic information from the json file
    :param person_name:
    :param pure_str:
    :return: the basic information objects
    """
    json_file = load_person_files(person_name)['basic_information_path']

    with open(json_file, 'r',encoding="utf-8") as f:
        data_ = json.load(f)

        if pure_str:
            data__ = delete_none(data_)
            return json.dumps(data__, separators=(',', ':'), indent=None)
        else:
            basic_information = data_['basic_information']
            person = Person(basic_information['name'])
            nicknames = Nicknames([])
            for _nickname in basic_information['nicknames']:
                nickname = Nickname(
                    _nickname['nickname'],
                    _nickname['people_use_the_name_exact'],
                    _nickname['people_use_the_name_range']
                )
                nicknames.add_nickname(nickname)

            if basic_information['education'] is None:
                educations = None
            else:
                educations = []
                for _education in basic_information['education']:
                    education = Education(**_education)
                    educations.append(education)

            personality = load_description_and_examples(basic_information['personality'])
            catchphrase = load_descriptions_variant(basic_information['catchphrase'], Descriptions)
            habits = load_description_and_examples(basic_information['habits'])

            return BasicInformation(
                name=person,
                gender=basic_information['gender'],
                home=basic_information['home'],
                nicknames=nicknames,
                date_of_birth=basic_information['date_of_birth'],
                educations=educations,
                race=basic_information['race'],
                personality=personality,
                catchphrase=catchphrase,
                habits=habits
            )


if __name__ == "__main__":
    basic_information = load_basic_information()
    print(basic_information)

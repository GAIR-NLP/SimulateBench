from dataclasses import dataclass
from typing import List, Union
from datetime import datetime
import json
from person.profile.base_data_class import Person, Descriptions, \
    DescriptionsFrozen, Description, Persona, load_descriptions_variant, load_person_files
from util.format_json import delete_none

DIALOGUE_ACT = ['questions', 'answers', 'elaborations', 'announcements', 'appreciation', 'agreements', 'disagreements',
                'negative_reactions', 'humor']


@dataclass
class Characteristics(Descriptions):
    pass


@dataclass
class GoalsOrPlans(Descriptions):
    pass


@dataclass
class RoutinesOrHabits(Descriptions):
    pass


@dataclass
class Experiences(DescriptionsFrozen):
    pass


@dataclass(frozen=True)
class Utterance:
    person: Person
    content: str
    discourse_acts: str


@dataclass
class Dialogue:
    time: datetime
    participant: List[Person]
    utterances: List[Utterance]

    def update(self, utterance: Utterance):
        self.utterances.append(utterance)


@dataclass
class Familiarity(Descriptions):
    pass


@dataclass
class Judgements(Descriptions):
    pass


@dataclass
class Affection(Descriptions):
    pass


@dataclass
class BehaviorPattern(Descriptions):
    pass


@dataclass
class Relationship(Descriptions):
    pass


@dataclass
class Relation:
    """

    """
    origin: Person  # the person is the agent its self
    destination: Person
    familiarity: Familiarity  # the familiarity of the origin to the destination
    judgement: Judgements  # the judgement of the origin to the destination
    affection: Affection  # the affection of the origin to the destination
    communication_history: List[Dialogue]
    behavior_pattern: BehaviorPattern
    relationship: Relationship

    def update_communication_history(self, dialogue: Dialogue):
        self.communication_history.append(dialogue)

    def update_affection(self, affection: List[Description]):
        self.affection.update(affection)

    def update_judgement(self, judgement: List[Description]):
        self.judgement.update(judgement)

    def update_familiarity(self, familiarity: List[Description]):
        self.familiarity.update(familiarity)


def load_communication_history(communication_history: List[dict]) -> Union[List[Dialogue], None]:
    return None


def load_relations(json_list: Union[None, List[dict]]) -> Union[None, List[Relation]]:
    if json_list is None:
        return None
    relations = []
    for _json in json_list:
        origin = Person(_json['origin'])
        destination = Person(_json['destination'])
        familiarity = load_descriptions_variant(_json['familiarity'], Familiarity)
        judgement = load_descriptions_variant(_json['judgement'], Judgements)
        affection = load_descriptions_variant(_json['affection'], Affection)
        communication_history = load_communication_history(_json['communication_history'])
        behavior_pattern = load_descriptions_variant(_json['behavior_pattern'], BehaviorPattern)
        relationship = load_descriptions_variant(_json['relationship'], Relationship)
        relations.append(Relation(origin, destination, familiarity, judgement, affection, communication_history,
                                  behavior_pattern, relationship))

    return relations


@dataclass
class SocialPersona:
    time: str
    persona: Persona
    characteristics: Characteristics
    goals_or_plans: GoalsOrPlans
    routines_or_habits: RoutinesOrHabits
    experiences: Experiences
    relations: List[Relation]

    def show_relation(self, person: Person) -> Union[None, Relation]:
        for relation in self.relations:
            if relation.destination == person:
                return relation
        return None

    def exist_relation(self, person: Person) -> bool:
        for relation in self.relations:
            if relation.destination == person:
                return True
        return False

    def add_relation(self, relation: Relation):
        self.relations.append(relation)

    def update_characteristics(self, characteristics: List[Description]):
        self.characteristics.update(characteristics)

    def update_goals_or_plans(self, goals_or_plans: List[Description]):
        self.goals_or_plans.update(goals_or_plans)

    def update_routines_or_habits(self, routines_or_habits: List[Description]):
        self.routines_or_habits.update(routines_or_habits)

    def update_experiences(self, experiences: List[Description]):
        self.experiences.update(experiences)


def load_social_personas(person_name: str = 'monica', pure_str=True) -> Union[List[SocialPersona], str, None]:
    json_file = load_person_files(person_name)['roles_path']
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    if pure_str:
        data_ = delete_none(json_dict)
        return json.dumps(data_, separators=(',', ':'), indent=None)

    social_personas = []
    for _persona, _json in json_dict.items():
        if _json is None or len(_json.keys()) == 0:
            continue
        time = _json['time']
        persona = Persona(_persona)
        characteristics = load_descriptions_variant(_json['characteristics'], Characteristics)
        routines_or_habits = load_descriptions_variant(_json['routines_or_habits'], RoutinesOrHabits)
        experiences = load_descriptions_variant(_json['experiences'], Experiences)
        goals_or_plans = load_descriptions_variant(_json['goals_or_plans'], GoalsOrPlans)
        relations = load_relations(_json['relations'])
        social_persona = SocialPersona(time, persona, characteristics, goals_or_plans, routines_or_habits, experiences,
                                       relations)
        social_personas.append(social_persona)

    return social_personas


def load_roles_categories(person_name: str = 'monica') -> Union[List[str], None]:
    json_file = load_person_files(person_name)['roles_path']
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    results = []
    roles_content = json_dict['roles']
    for persona in roles_content.keys():
        if len(roles_content[persona].keys()) == 0:
            continue
        else:
            results.append(persona)

    if len(results) == 0:
        return None
    else:
        return results


def load_roles_categories_and_des_person(person_name: str = 'monica') -> Union[dict, None]:
    """
    return the persona(role) and the destination person/people of the specific persona(role),
    the destination person/people has/have a relation with person_name
    :param person_name:
    :return:
    """
    json_file = load_person_files(person_name)['roles_path']
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    results = {}
    roles_content = json_dict['roles']
    for persona in roles_content.keys():
        if len(roles_content[persona].keys()) == 0 or  roles_content[persona]['relations'] is None:
            continue
        else:
            persona_dest_person = []
            for relation in roles_content[persona]['relations']:
                persona_dest_person.append(relation['destination'])
            if len(persona_dest_person) != 0:
                results[persona] = persona_dest_person

    if len(results.keys()) == 0:
        return None
    else:
        return results


if __name__ == "__main__":
    print(load_roles_categories_and_des_person())

import json
from typing import List
import os

from GPTMan.person.profile.role import load_roles_categories, load_social_personas, \
    load_roles_categories_and_des_person

COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q = \
    "/GPTMan/db/benchmark/common_v2/profile-basic-information-questionnaire.json"
COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q = \
    "/GPTMan/db/benchmark/common_v2/profile-roles-non-relation-questionnaire.json"
COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q = \
    "/GPTMan/db/benchmark/common_v2/profile-roles-relation-questionnaire.json"
QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q = \
    "/GPTMan/db/benchmark/{person_name}_v2/profile-basic-information-questionnaire.json"
QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q = \
    "/GPTMan/db/benchmark/{person_name}_v2/profile-roles-non-relation-questionnaire.json"
QUESTIONNAIRE_PATH_ROLES_RELATION_Q = \
    "/GPTMan/db/benchmark/{person_name}_v2/profile-roles-relation-questionnaire.json"


def load_json_file(path):
    with open(path, 'r') as f:
        raw_data = json.load(f)

    return raw_data


def load_template_questions(file_name="profile-roles-non-relation-questionnaire.json"):
    path = f"/GPTMan/db/benchmark/common_v2/{file_name}"

    return load_json_file(path)


def rename(path):
    obj = load_json_file(path)
    for count in range(len(obj)):
        obj_count = obj[count]
        content = obj_count['generated_answer']
        obj_count['generated_answer_gpt-4'] = content
        del obj_count['generated_answer']

    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


if __name__ == '__main__':
    rename(QUESTIONNAIRE_PATH_ROLES_RELATION_Q.format(person_name='monica'))
    pass
# generate_json_roles_non_relation_q(person_name='monica')
# generate_json_roles_relation_q(person_name='monica')
# print(s)
# pass

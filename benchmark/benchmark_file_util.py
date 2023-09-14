import json
from typing import List
import os
from random import shuffle
from person.profile.role import load_roles_categories, load_social_personas, \
    load_roles_categories_and_des_person

ROOT_PATH = "/home/yxiao2/pycharm/GPTMan/db/benchmark"
COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q = \
    f"{ROOT_PATH}/basic_information/questions.json"
COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q = \
    f"{ROOT_PATH}/role_non_relation/questions.json"
COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q = \
    f"{ROOT_PATH}/role_relation/questions.json"
QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q = \
    f"{ROOT_PATH}/basic_information/{{person_name}}/{{profile_version}}/{{system_version}}/{{benchmark_version}}/questions.json"
QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q = \
    f"{ROOT_PATH}/role_non_relation/{{person_name}}/{{profile_version}}/{{system_version}}/{{benchmark_version}}/questions.json"
QUESTIONNAIRE_PATH_ROLES_RELATION_Q = \
    f"{ROOT_PATH}/role_relation/{{person_name}}/{{profile_version}}/{{system_version}}/{{benchmark_version}}/questions.json"

ANSWER_PATH_BASIC_INFORMATION_Q = \
    f"{ROOT_PATH}/basic_information/{{person_name}}/{{profile_version}}/{{system_version}}/{{benchmark_version}}/{{prompt_kind}}/{{prompt_name}}.json"
ANSWER_PATH_ROLES_NON_RELATION_Q = \
    f"{ROOT_PATH}/role_non_relation/{{person_name}}/{{profile_version}}/{{system_version}}/{{benchmark_version}}/{{prompt_kind}}/{{prompt_name}}.json"
ANSWER_PATH_ROLES_RELATION_Q = \
    f"{ROOT_PATH}/role_relation/{{person_name}}/{{profile_version}}/{{system_version}}/{{benchmark_version}}/{{prompt_kind}}/{{prompt_name}}.json"

# make the agent/LLM to answer the question in the benchmark,
# so need to record the different prompt, which will cause different
PROMPT_PATH = f"{ROOT_PATH}/agent_answer_question_prompt/prompt_record.json"


def load_json_file(path):
    with open(path, 'r') as f:
        raw_data = json.load(f)

    return raw_data


def load_prompt(prompt_name, prompt_kind="few_shot"):
    path = PROMPT_PATH
    object_prompt = load_json_file(path)
    prompt = object_prompt["LLM_answer_prompt"][prompt_kind][prompt_name]['content']
    return prompt


def load_template_questions(file_name="profile-roles-non-relation-questionnaire.json"):
    path = f"/GPTMan/db/benchmark/common_v2/{file_name}"

    return load_json_file(path)


def process(model_name="longchat-7b-32k-profile_v1.5"):
    path_ = ANSWER_PATH_ROLES_RELATION_Q.format(person_name="monica", prompt_kind="few_shot", prompt_name="prompt1")
    obj_raw = load_json_file(path_)
    obj = obj_raw[f"few_shot_prompt1"]
    for count in range(len(obj)):
        obj_count = obj[count]
        content = obj_count[f'generated_answer_{model_name}']
        if "Your answer:" in content:
            content_index = content.rfind("Your answer:\n")
            text = content[content_index + 13:]
            obj_raw["few_shot_prompt1"][count][f'generated_answer_{model_name}'] = text
        else:
            continue

    with open(path_, 'w') as f:
        json.dump(obj_raw, f, indent=4)


def rename(path):
    obj = load_json_file(path)
    for count in range(len(obj)):
        obj_count = obj[count]
        content = obj_count['generated_answer']
        obj_count['generated_answer_gpt-4'] = content
        del obj_count['generated_answer']

    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def rotate_answer_order(path):
    old_path = path.replace("benchmark_v2", "benchmark_v1")
    obj_ = load_json_file(old_path)
    for count in range(len(obj_)):
        obj_count = obj_[count]
        content = obj_count['question']
        if "[this is a single-choice question,you should only choose from:" in content:
            content_index = content.find("[this is a single-choice question,you should only choose from:")

            old_text = content[content_index + len("[this is a single-choice question,you should only choose from:"):-1]

            test_list = old_text.split(";")
            test_list[0] = test_list[0].strip()

            shuffle(test_list)

            new_text = " " + ";".join(test_list)

            obj_[count]['question'] = content.replace(old_text, new_text)
        else:
            continue

    with open(path, 'w') as f:
        json.dump(obj_, f, indent=4)


if __name__ == '__main__':
    """rename(QUESTIONNAIRE_PATH_ROLES_RELATION_Q.format(person_name='monica'))
    pass"""
    """"process(model_name="longchat-13b-16k")
    process(model_name="longchat-7b-32k-profile_v1.5")
    process(model_name="longchat-7b-16k")"""
    # generate_json_roles_non_relation_q(person_name='monica')
    # generate_json_roles_relation_q(person_name='monica')
    # print(s)
    # pass
    # generate_json_basic_information_q(person_name='monica')
    benchmark_type = "role_relation"
    path_ = f"{ROOT_PATH}/{benchmark_type}/profile_v1/system_v1/benchmark_v2/monica/questions.json"
    rotate_answer_order(path_)

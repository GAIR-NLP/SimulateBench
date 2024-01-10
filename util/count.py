import os
from util.gpt_token_usage import num_tokens_from_chat_messages


def count_dir_number(path):
    count = 0
    for fn in os.listdir(path):  # fn 表示的是文件名
        count = count + 1
    return count


def count_profile_avg_length(path):
    count_length = 0
    count_number = 0
    dir_names = os.listdir(path)
    dir_names.remove("__init__.py")
    dir_names.remove("template")
    for fn in dir_names:
        basic_path = os.path.join(path, fn, "profile_v1", "basic_information.json")
        role_path = os.path.join(path, fn, "profile_v1", "roles.json")
        content = {}
        with open(basic_path) as user_file:
            file_contents = user_file.read()
            content["basic_information"] = file_contents
        with open(role_path) as user_file:
            file_contents = user_file.read()
            content["roles"] = file_contents
        count_length += num_tokens_from_chat_messages([content])
        count_number += 1
    print(count_length / count_number)


def count_questions(path):
    import json
    count = 0
    count_number = 0
    dir_names = os.listdir(path)
    dir_names.remove("template_questions")
    dir_names.remove("questions.json")
    if "role_relation" in path:
        dir_names.remove("galadriel")
    for fn in dir_names:
        path_ = os.path.join(path, fn, "profile_v1", "system_v1", "benchmark_v2", "questions.json")
        with open(path_) as f:
            tem_count = len(json.load(f))
            count += tem_count
            if "monica" in path_:
                print(tem_count)
        count_number += 1
    print(count / count_number)


def count_question_tokens(path):
    import json
    count_questions = 0
    count_tokens = 0
    dir_names = os.listdir(path)
    dir_names.remove("template_questions")
    dir_names.remove("questions.json")
    if "role_relation" in path:
        dir_names.remove("galadriel")
    for fn in dir_names:
        path_ = os.path.join(path, fn, "profile_v1", "system_v1", "benchmark_v2", "questions.json")
        with open(path_) as f:
            questions = json.load(f)
        for i in range(len(questions)):
            questions[i] = {"question": questions[i]["question"]}
        # print(questions[0])
        count_tokens += num_tokens_from_chat_messages(questions)
        count_questions += len(questions)
    return count_tokens, count_questions


if __name__ == '__main__':
    # path = '/home/yxiao2/pycharm/GPTMan/db/profile'
    # print(count_dir_number(path))
    basic_path = "/home/yxiao2/pycharm/GPTMan/db/benchmark/basic_information"
    role_path = "/home/yxiao2/pycharm/GPTMan/db/benchmark/role_non_relation"
    relation_ship_path = "/home/yxiao2/pycharm/GPTMan/db/benchmark/role_relation"
    # count_profile_avg_length(path)
    # count_questions(basic_path)
    # count_questions(role_path)
    # count_questions(relation_ship_path)
    all_tokens = 0
    all_questions = 0
    tem_token, tem_question = count_question_tokens(basic_path)
    all_tokens += tem_token
    all_questions += tem_question
    tem_token, tem_question = count_question_tokens(role_path)
    all_tokens += tem_token
    all_questions += tem_question
    tem_token, tem_question = count_question_tokens(relation_ship_path)
    all_tokens += tem_token
    all_questions += tem_question
    print(all_tokens / all_questions)

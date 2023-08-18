import json
from GPTMan.benchmark.benchmark_file_util import \
    QUESTIONNAIRE_PATH_ROLES_RELATION_Q, \
    QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q, \
    QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, \
    load_json_file, \
    COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, \
    COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q, \
    COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q

from GPTMan.benchmark.generate_benchmark_gold_answer_for_question import answer_basic_information, \
    answer_roles_non_relation, answer_roles_relation
from GPTMan.person.action.brain.agent import Agent
from GPTMan.person.profile.role import load_roles_categories_and_des_person
from GPTMan.person.profile.basic_information import load_basic_information
from GPTMan.person.profile.role import load_social_personas
from GPTMan.log.logger import logger
import random

RANDOM_CITIES = [
    'Tokyo', 'Delhi', 'Shanghai', 'Mexico City', 'Beijing', 'New York', 'Chongqing', 'Lagos', 'Tianjin', 'Los Angeles'
]
RANDOM_NAMES = [
    'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Charles', 'Joseph', 'Thomas'
]


def load_q(path):
    with open(path, 'r') as f:
        benchmark_q = json.load(f)

    return benchmark_q


def process_basic_question_general(input_question):
    input_question = input_question.replace('\n- ', ';')

    return input_question


def process_roles_non_relation_question_general(input_question):
    input_question = input_question.replace('\n\n- ', '<choose from: ')
    input_question = input_question.replace('\n- ', ';')
    input_question += '>'

    return input_question


def process_roles_relation_question_general(input_question):
    input_question = input_question.replace('\n\n- ', '<choose from: ')
    input_question = input_question.replace('\n- ', ';')
    input_question += '>'

    return input_question


class Benchmark:
    def __init__(self,
                 basic_information_path_template=COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 basic_information_path_benchmark=QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 person_name='monica',
                 ):

        self.basic_information_path_template = basic_information_path_template
        self.roles_non_relation_path_template = roles_non_relation_path_template
        self.roles_relation_path_template = roles_relation_path_template

        self.basic_information_path_benchmark = basic_information_path_benchmark.format(person_name=person_name)
        self.roles_non_relation_path_benchmark = roles_non_relation_path_benchmark.format(person_name=person_name)
        self.roles_relation_path_benchmark = roles_relation_path_benchmark.format(person_name=person_name)

        self.person_name = person_name

        self.relations = load_roles_categories_and_des_person(self.person_name)

    def process_question_home(self, input_question):
        information_obj = json.loads(load_basic_information(person_name=self.person_name, pure_str=True))
        gold_home = information_obj['basic_information']['home']
        other_city = random.sample(RANDOM_CITIES, 3)

        city_string = ';'.join(other_city)
        city = f"choose from: {city_string};{gold_home}"

        input_question = input_question.replace(
            'City A;City B;City C;City <gold_city>', city)
        return input_question

    def process_question_nickname(self):
        information_obj = json.loads(load_basic_information(person_name=self.person_name, pure_str=True))
        nick_names = information_obj['basic_information']['nicknames']

        results = []
        for nick_name_obj in nick_names:
            nick_name = nick_name_obj['nickname']

            fake_caller = random.sample(RANDOM_NAMES, 3)

            gold_caller_list = []

            if 'people_use_the_name_exact' in nick_name_obj.keys():
                gold_caller_list.extend(nick_name_obj['people_use_the_name_exact'])

            if 'people_use_the_name_range' in nick_name_obj.keys():
                gold_caller_list.extend(nick_name_obj['people_use_the_name_range'])

            gold_caller = ' and '.join(gold_caller_list)

            result = f'Who will call you by your nickname {nick_name}?'
            result += f'<choose from: {gold_caller};{";".join(fake_caller)};I don\'t know>'
            results.append(result)
        return results

    def process_catchphrase(self):
        information_obj = json.loads(load_basic_information(person_name=self.person_name, pure_str=True))
        catchphrases = information_obj['basic_information']['catchphrase']

        results = []

        for catchphrase_obj in catchphrases:
            catchphrase = catchphrase_obj['description']

            result = f"Is '{catchphrase}' your catchphrase or favorite saying?"
            result += f"<choose from: Yes;No;I don't know>"
            results.append(result)

        return results

    def process_race(self, input_question):
        information_obj = json.loads(load_basic_information(person_name=self.person_name, pure_str=True))
        race = information_obj['basic_information']['race']
        input_question = input_question.replace('[race]', race)
        input_question = input_question.replace('\n- ', ';')

        return input_question

    def write_question_basic_information_file(self):
        logger.info(f'generating basic information questions into file {self.basic_information_path_benchmark}')

        raw_questions = load_json_file(self.basic_information_path_template)
        benchmark = []

        for question_obj in raw_questions:

            raw_question = question_obj['question']
            processed_questions = []

            if 'What is the location of your home?' in raw_question:
                processed_questions.append(self.process_question_home(raw_question))
            elif 'nickname' in raw_question:
                processed_questions.extend(self.process_question_nickname())
            elif 'catchphrase' in raw_question:
                processed_questions.extend(self.process_catchphrase())
            elif '[race]' in raw_question:
                processed_questions.append(self.process_race(raw_question))
            else:
                processed_questions.append(process_basic_question_general(raw_question))

            for processed_question in processed_questions:
                benchmark.append({
                    'question': processed_question,
                    'gold_answer': None,
                    'generated_answer': None,
                })

        with open(self.basic_information_path_benchmark, 'w') as f:
            json.dump(benchmark, f, indent=4)

        logger.info('generating basic information questions finished')

    def gold_answer(self, path, batch_size=10):
        logger.info(
            f'generating gold answer into file {path}')

        benchmark_q = load_q(path)
        count_list = [i for i in range(len(benchmark_q))]
        chunk_list = [count_list[i:i + batch_size] for i in range(0, len(count_list), batch_size)]

        if "basic_information" in path:
            answer_fun = answer_basic_information
        elif "roles_non_relation" in path:
            answer_fun = answer_roles_non_relation
        else:
            answer_fun = answer_roles_relation

        for chunk in chunk_list:

            orig_question = "finish blew questions,and organize the answer for every question " \
                            "in the json format as a list as:[<answer for question1>,<answer for question2>,....]\n"
            for count in chunk:

                question = benchmark_q[count]['question']
                if "choose from" in question:
                    question = question.replace(
                        "choose from",
                        "this is a single-choice question,you should only choose from")
                orig_question += f"question {count + 1}: {question}\n"
                orig_question += "Your answer:\n"

            results = answer_fun(
                question=orig_question,
                person=self.person_name
            )
            print(results)
            results = json.loads(results)

            try:
                if len(results) != len(chunk):
                    raise Exception("the number of answers is not equal to the number of questions")
                else:
                    for count in chunk:
                        benchmark_q[count]['gold_answer'] = results[count % batch_size]
                    with open(path, 'w') as f:
                        json.dump(benchmark_q, f, indent=4)
            except Exception as e:
                print(e)

        logger.info(f'generating gold answer finished for {path}')

    def gold_answer_question_basic_information(self):
        self.gold_answer(self.basic_information_path_benchmark)

    def process_characteristic(self, input_question):
        input_question = process_roles_non_relation_question_general(input_question)
        information_obj = json.loads(load_social_personas(person_name=self.person_name, pure_str=True))

        roles_and_des = load_roles_categories_and_des_person(person_name=self.person_name)

        results = []

        for role in roles_and_des.keys():
            if 'characteristics' not in information_obj['roles'][role].keys():
                continue
            characteristics = information_obj['roles'][role]['characteristics']

            if characteristics is None:
                continue

            for characteristic in characteristics:
                new_question = input_question.replace('specific skill related to the role', characteristic)
                new_question = new_question.replace('role', role)
                results.append(new_question)

        return results

    def process_routines(self, input_question):
        input_question = process_roles_non_relation_question_general(input_question)
        information_obj = json.loads(load_social_personas(person_name=self.person_name, pure_str=True))

        roles_and_des = load_roles_categories_and_des_person(person_name=self.person_name)

        results = []

        for role in roles_and_des.keys():

            if 'routines_or_habits' not in information_obj['roles'][role].keys():
                continue

            routines = information_obj['roles'][role]['routines_or_habits']

            if routines is None:
                continue

            for routine in routines:
                new_question = input_question.replace('routines or habits', routine)
                new_question = new_question.replace('role', role)
                results.append(new_question)

        return results

    def process_goals(self, input_question):
        input_question = process_roles_non_relation_question_general(input_question)
        information_obj = json.loads(load_social_personas(person_name=self.person_name, pure_str=True))

        roles_and_des = load_roles_categories_and_des_person(person_name=self.person_name)

        results = []

        for role in roles_and_des.keys():
            if 'goals_or_plans' not in information_obj['roles'][role].keys():
                continue

            goals = information_obj['roles'][role]['goals_or_plans']

            if goals is None:
                continue

            for goal in goals:
                new_question = input_question.replace('goal', goal)
                new_question = new_question.replace('role', role)
                results.append(new_question)

        return results

    def write_question_roles_non_relation_file(self):
        logger.info(f'generating roles non relation questions into file {self.roles_non_relation_path_benchmark}')

        raw_questions = load_json_file(self.roles_non_relation_path_template)
        benchmark = []

        for topic in raw_questions.keys():
            questions = raw_questions[topic]

            for raw_question in questions:

                processed_questions = []

                if 'specific skill related to the role' in raw_question:
                    processed_questions.extend(self.process_characteristic(raw_question))
                elif 'routines or habits' in raw_question:
                    processed_questions.extend(self.process_routines(raw_question))
                elif '[goal]' in raw_question:
                    processed_questions.extend(self.process_goals(raw_question))

                if len(processed_questions) == 0:
                    break

                for processed_question in processed_questions:
                    benchmark.append({
                        'question': processed_question,
                        'gold_answer': None,
                        'generated_answer': None,
                    })

        with open(self.roles_non_relation_path_benchmark, 'w') as f:
            json.dump(benchmark, f, indent=4)

        logger.info('generating roles non relation questions finished')

    def gold_answer_question_roles_non_relation(self):
        self.gold_answer(self.roles_non_relation_path_benchmark)

    def gold_answer_question_roles_relation(self):
        self.gold_answer(self.roles_relation_path_benchmark)

    def write_question_roles_relation_file(self):
        logger.info(f'generating roles relation questions into file {self.roles_relation_path_benchmark}')

        raw_questions_ = load_json_file(self.roles_relation_path_template)
        raw_questions = []
        for topic in raw_questions_.keys():
            raw_questions.extend(raw_questions_[topic])

        benchmark = []

        roles_and_des = load_roles_categories_and_des_person(person_name=self.person_name)

        for raw_question in raw_questions:
            for role in roles_and_des.keys():
                for des in roles_and_des[role]:
                    processed_question = process_roles_relation_question_general(raw_question)
                    processed_question = processed_question.replace('<role>', role)
                    processed_question = processed_question.replace('<the individual>', des)
                    benchmark.append({
                        'question': processed_question,
                        'gold_answer': None,
                        'generated_answer': None,
                    })

        with open(self.roles_relation_path_benchmark, 'w') as f:
            json.dump(benchmark, f, indent=4)

        logger.info('generating roles relation questions finished')

    def answer_question_roles_non_relation(self, agent):
        self.answer_batch(self.roles_non_relation_path_benchmark, agent)

    def answer_question_roles_relation(self, agent):
        self.answer_batch(self.roles_relation_path_benchmark, agent)

    def answer_question_basic_information(self, agent):
        self.answer_batch(self.basic_information_path_benchmark, agent)

    def answer_batch(self, path, agent, split=10):
        logger.info(
            f'generating answer into file {path}')

        benchmark_q = load_q(path)

        count_list = [i for i in range(len(benchmark_q))]
        chunk_list = [count_list[index:index + split] for index in range(0, len(benchmark_q), split)]

        for chunk in chunk_list:
            """agent = Agent(person_name=self.person_name)"""
            orig_question = "finish blew questions,and organize the answer for every question " \
                            "in the json format as a list as:[<answer for question1>,<answer for question2>,....]\n"

            if f"generated_answer_{agent.model_name}" in benchmark_q[chunk[0]].keys():
                continue

            for count in chunk:

                question = benchmark_q[count]['question']
                if "choose from" in question:
                    question = question.replace(
                        "choose from",
                        "this is a single-choice question,you should only choose from")
                orig_question += f"question {count + 1}: {question}\n"
                orig_question += "Your answer:\n"

            results = agent.run(orig_question)
            print(results)
            results = json.loads(results)

            try:
                if len(results) != len(chunk):
                    raise Exception("the number of answers is not equal to the number of questions")
                else:
                    for count in chunk:
                        benchmark_q[count][f'generated_answer_{agent.model_name}'] = results[count % split]
                    with open(path, 'w') as f:
                        json.dump(benchmark_q, f, indent=4)
            except Exception as e:
                print(e)
            finally:
                agent.clear_message()


if __name__ == "__main__":
    # write_basic_information_gold_a(person_name='monica')
    # write_roles_non_relation_gold_a(person_name='monica')
    # write_roles_relation_gold_a(person_name='monica')
    # write_basic_information_a(person_name='monica')
    # write_roles_non_relation_a(person_name='monica')
    # write_roles_relation_a(person_name='monica')
    benchmark = Benchmark()
    # benchmark.gold_answer_question_roles_non_relation()
    # benchmark.gold_answer_question_roles_relation()
    # benchmark.write_question_roles_non_relation_file()
    # benchmark.write_question_roles_relation_file()
    agent = Agent(person_name='monica', model_name='gpt-3.5-turbo-16k')
    benchmark.answer_question_roles_non_relation(agent=agent)

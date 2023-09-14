import json
from benchmark.benchmark_file_util import \
    QUESTIONNAIRE_PATH_ROLES_RELATION_Q, \
    QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q, \
    QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, \
    COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q, \
    COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q, \
    COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q, \
    ANSWER_PATH_BASIC_INFORMATION_Q, \
    ANSWER_PATH_ROLES_NON_RELATION_Q, \
    ANSWER_PATH_ROLES_RELATION_Q

import copy
from benchmark.benchmark_gold_answer_generate import answer_basic_information, \
    answer_roles_non_relation, answer_roles_relation
from benchmark.benchmark_file_util import load_json_file, load_prompt
from person.action.brain.agent import Agent
from person.action.brain.chat_model import OpenAI
from person.action.brain_chat_glm.chat_glm import ChatGLM26B32K
from person.action.brain_qwen.qwen import QWen7BChat
from person.action.brain_vicuna.vicuna import Vicuna
from person.profile.role import load_roles_categories_and_des_person
from person.profile.basic_information import load_basic_information
from person.profile.role import load_social_personas
from log.logger import logger
import random
from tqdm import tqdm
from person.action.brain.translator import Rewrite
import os
from benchmark.benchmark_file_util import ROOT_PATH

RANDOM_CITIES = [
    'Tokyo', 'Delhi', 'Shanghai', 'Mexico City', 'Beijing', 'New York', 'Chongqing', 'Lagos', 'Tianjin', 'Los Angeles'
]
RANDOM_NAMES = [
    'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Charles', 'Joseph', 'Thomas'
]


def process_question(input_question):
    if "\n\n- " not in input_question:
        return input_question
    input_question = input_question.replace('\n\n- ', '[this is a single-choice question,you should only choose from: ')

    input_question = input_question.replace('\n- ', ';')

    input_question += ']'

    if 'you should only choose from: ' in input_question:
        answer_start = input_question.find('you should only choose from: ')
        origin_answers = input_question[answer_start + len('you should only choose from: '):-1]
        answers = origin_answers.split(';')
        random.shuffle(answers)
        input_question = input_question[:answer_start + len('you should only choose from: ')] + ';'.join(answers) + ']'

    return input_question


def make_dir(person_name, profile_version, system_version, benchmark_version, file_type):
    dir_path = os.path.join(ROOT_PATH, file_type, person_name, profile_version, system_version, benchmark_version)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Benchmark:
    def __init__(self,
                 person_name, prompt_name, prompt_kind,
                 profile_version, system_version,
                 benchmark_version,
                 basic_information_path_template=COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 basic_information_path_benchmark=QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_RELATION_Q,

                 ):
        self.basic_information_path_template = basic_information_path_template.format(
            profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version,
        )
        self.roles_non_relation_path_template = roles_non_relation_path_template.format(
            profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version,
        )
        self.roles_relation_path_template = roles_relation_path_template.format(
            profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version,
        )

        self.basic_information_path_benchmark = basic_information_path_benchmark.format(
            person_name=person_name,
            benchmark_version=benchmark_version,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            profile_version=profile_version,
            system_version=system_version)

        self.roles_non_relation_path_benchmark = roles_non_relation_path_benchmark.format(
            person_name=person_name,
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name)
        self.roles_relation_path_benchmark = roles_relation_path_benchmark.format(
            person_name=person_name,
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name)

        self.person_name = person_name
        self.prompt_kind = prompt_kind
        self.prompt_name = prompt_name

        self.profile_version = profile_version
        self.system_version = system_version
        self.benchmark_version = benchmark_version

        # include the person's roles and other related in this role
        self.relations = load_roles_categories_and_des_person(self.person_name)


class BenchmarkQuestionGenerator(Benchmark):
    def __init__(self, person_name, prompt_name, prompt_kind,
                 profile_version, system_version,
                 benchmark_version,
                 basic_information_path_template=COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 basic_information_path_benchmark=QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_RELATION_Q,

                 ):
        super(BenchmarkQuestionGenerator, self).__init__(
            basic_information_path_template=basic_information_path_template,
            roles_non_relation_path_template=roles_non_relation_path_template,
            roles_relation_path_template=roles_relation_path_template,
            basic_information_path_benchmark=basic_information_path_benchmark,
            roles_non_relation_path_benchmark=roles_non_relation_path_benchmark,
            roles_relation_path_benchmark=roles_relation_path_benchmark,
            person_name=person_name, prompt_kind=prompt_kind, prompt_name=prompt_name,
            profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version
        )

        self.rewriter = Rewrite()

    def process_question_home(self, input_question):
        information_obj = json.loads(load_basic_information(person_name=self.person_name, pure_str=True))
        gold_home = information_obj['basic_information']['home']
        other_city = random.sample(RANDOM_CITIES, 3)

        other_city.append(gold_home)
        other_city.append("There's not enough information to answer this question.")
        random.shuffle(other_city)

        city = ";".join(other_city)

        input_question = f"What is the location of your home?[this is a single-choice question,you should only choose from: {city}]"

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
            result += f'[this is a single-choice question,you should only choose from: {gold_caller};{";".join(fake_caller)};There\'s not enough information to answer this question.]'
            results.append(result)
        return results

    def process_catchphrase(self):
        information_obj = json.loads(load_basic_information(person_name=self.person_name, pure_str=True))
        catchphrases = information_obj['basic_information']['catchphrase']

        results = []

        for catchphrase_obj in catchphrases:
            catchphrase = catchphrase_obj

            result = f"Is '{catchphrase}' your catchphrase or favorite saying?"
            result += f"[this is a single-choice question,you should only choose from: Yes;No;There's not enough information to answer this question.]"
            results.append(result)

        return results

    def process_race(self, input_question):
        information_obj = json.loads(load_basic_information(person_name=self.person_name, pure_str=True))
        if 'race' not in information_obj['basic_information'].keys():
            return None
        race = information_obj['basic_information']['race']
        input_question = input_question.replace('[race]', race)
        input_question = input_question.replace('\n- ', ';')

        return input_question

    def write_question_basic_information_file(self):
        logger.info(f'generating basic information questions into file {self.basic_information_path_benchmark}')

        make_dir(person_name=self.person_name,
                 profile_version=self.profile_version,
                 system_version=self.system_version,
                 benchmark_version=self.benchmark_version,
                 file_type="basic_information"
                 )

        raw_questions = load_json_file(self.basic_information_path_template)
        benchmark = []

        for question_obj in raw_questions:
            raw_question = process_question(question_obj['question'])
            processed_questions = []
            if 'What is the location of your home?' in raw_question:
                processed_questions.append(self.process_question_home(raw_question))
            elif 'nickname' in raw_question:
                processed_questions.extend(self.process_question_nickname())
            elif 'Is [catchphrase] your catchphrase' in raw_question:
                processed_questions.extend(self.process_catchphrase())
            elif '[race]' in raw_question:
                result = self.process_race(raw_question)
                if result is not None:
                    processed_questions.append(result)
            else:
                processed_questions.append(raw_question)

            for processed_question in processed_questions:
                benchmark.append({
                    'question': processed_question,
                    'gold_answer': None,
                })

        with open(self.basic_information_path_benchmark, 'w') as f:
            json.dump(benchmark, f, indent=4)

        logger.info('generating basic information questions finished')

    def rewrite(self, question, number):
        """

        Args:
            question:
            number: for thr non relation question, the origin statement need to be rewritten for more questions

        Returns:

        """
        self.rewriter.set_rewrite_times(number)
        results = self.rewriter.run(question)

        results = json.loads(results)
        return results

    def process_characteristic(self, input_question, number=2):
        """

        Args:
            input_question:
            number: for thr non relation question, the origin statement need to be rewritten for more questions

        Returns:

        """
        input_question = process_question(input_question)
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
                self.process_non_relation_general(
                    characteristic,
                    input_question,
                    number,
                    results,
                    role,
                    replaced_text="specific skill related to the role"
                )

        return results

    def process_non_relation_general(self, origin_sentence, input_question, number, results, role, replaced_text):
        """

        Args:
            origin_sentence:
            input_question:
            number: for thr non relation question, the origin statement need to be rewritten for more questions
            results:
            role:
            replaced_text:

        Returns:

        """
        new_sentence = self.rewrite(origin_sentence, number)
        if len(new_sentence) != number:
            logger.warn(f"rewrite number is not {number}:{len(new_sentence)}")
        for sentence in new_sentence:
            new_question = input_question.replace(replaced_text, sentence)
            new_question = new_question.replace('role', role)
            results.append(new_question)

    def process_routines(self, input_question, number=2):
        """

        Args:
            input_question:
            number: for thr non relation question, the origin statement need to be rewritten for more questions

        Returns:

        """
        input_question = process_question(input_question)
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
                self.process_non_relation_general(routine, input_question, number, results, role,
                                                  replaced_text="routines or habits"
                                                  )

        return results

    def process_goals(self, input_question, number=2):
        """

        Args:
            input_question:
            number: for thr non relation question, the origin statement need to be rewritten for more questions

        Returns:

        """
        input_question = process_question(input_question)
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
                self.process_non_relation_general(goal, input_question, number, results, role,
                                                  replaced_text="goal")

        return results

    def write_question_roles_non_relation_file(self):
        logger.info(f'generating roles non relation questions into file {self.roles_non_relation_path_benchmark}')

        make_dir(person_name=self.person_name,
                 profile_version=self.profile_version,
                 system_version=self.system_version,
                 benchmark_version=self.benchmark_version,
                 file_type="role_non_relation"
                 )

        raw_questions = load_json_file(self.roles_non_relation_path_template)
        benchmark = []

        for topic in raw_questions.keys():
            questions = raw_questions[topic]

            for raw_question_obj in questions:
                raw_question = raw_question_obj['question']

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

                    })

        with open(self.roles_non_relation_path_benchmark, 'w') as f:
            json.dump(benchmark, f, indent=4)

        logger.info('generating roles non relation questions finished')

    def write_question_roles_relation_file(self):
        logger.info(f'generating roles relation questions into file {self.roles_relation_path_benchmark}')

        make_dir(person_name=self.person_name,
                 profile_version=self.profile_version,
                 system_version=self.system_version,
                 benchmark_version=self.benchmark_version,
                 file_type="role_relation"
                 )

        raw_questions_ = load_json_file(self.roles_relation_path_template)
        raw_questions = []
        for topic in raw_questions_.keys():
            raw_questions.extend(raw_questions_[topic])

        benchmark = []

        roles_and_des = load_roles_categories_and_des_person(person_name=self.person_name)

        for raw_question_obj in raw_questions:
            for role in roles_and_des.keys():
                for des in roles_and_des[role]:
                    raw_question = raw_question_obj['question']
                    processed_question = process_question(raw_question)
                    processed_question = processed_question.replace('<role>', role)
                    processed_question = processed_question.replace('<the individual>', des)
                    benchmark.append({
                        'question': processed_question,

                    })

        with open(self.roles_relation_path_benchmark, 'w') as f:
            json.dump(benchmark, f, indent=4)

        logger.info('generating roles relation questions finished')


class BenchmarkGoldAnswerGenerator(Benchmark):
    def __init__(self, person_name, prompt_name, prompt_kind,
                 profile_version, system_version,
                 benchmark_version, basic_information_path_template=COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 basic_information_path_benchmark=QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 ):
        super(BenchmarkGoldAnswerGenerator, self).__init__(
            basic_information_path_template=basic_information_path_template,
            roles_non_relation_path_template=roles_non_relation_path_template,
            roles_relation_path_template=roles_relation_path_template,
            basic_information_path_benchmark=basic_information_path_benchmark,
            roles_non_relation_path_benchmark=roles_non_relation_path_benchmark,
            roles_relation_path_benchmark=roles_relation_path_benchmark,
            person_name=person_name, prompt_kind=prompt_kind, prompt_name=prompt_name,
            profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version
        )

    def gold_answer(self, path, batch_size=8):
        logger.info(
            f'generating gold answer into file {path}')

        benchmark_q = load_json_file(path)
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
                            f"in the json format as a list as:[<answer for question1>...<answer for question{len(chunk)}>]\n"
            for count in chunk:
                question = benchmark_q[count]['question']

                orig_question += f"question {count + 1}: {question}\n"
            orig_question += f"Your answer for the {len(chunk)} questions:\n"

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

    def gold_answer_question_roles_non_relation(self):
        self.gold_answer(self.roles_non_relation_path_benchmark)

    def gold_answer_question_roles_relation(self):
        self.gold_answer(self.roles_relation_path_benchmark)


class BenchmarkTest(Benchmark):
    def __init__(self, person_name, prompt_name, prompt_kind,
                 profile_version, system_version,
                 benchmark_version, basic_information_path_template=COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 basic_information_path_benchmark=QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
                 roles_non_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
                 roles_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
                 answer_path_basic_information=ANSWER_PATH_BASIC_INFORMATION_Q,
                 answer_path_roles_non_relation=ANSWER_PATH_ROLES_NON_RELATION_Q,
                 answer_path_roles_relation=ANSWER_PATH_ROLES_RELATION_Q,

                 ):
        super(BenchmarkTest, self).__init__(
            basic_information_path_template=basic_information_path_template,
            roles_non_relation_path_template=roles_non_relation_path_template,
            roles_relation_path_template=roles_relation_path_template,
            basic_information_path_benchmark=basic_information_path_benchmark,
            roles_non_relation_path_benchmark=roles_non_relation_path_benchmark,
            roles_relation_path_benchmark=roles_relation_path_benchmark,
            person_name=person_name, prompt_kind=prompt_kind, prompt_name=prompt_name,
            profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version
        )

        self.answer_path_basic_information = answer_path_basic_information.format(
            person_name=person_name, profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name)
        self.answer_path_roles_non_relation = answer_path_roles_non_relation.format(
            person_name=person_name,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version
        )
        self.answer_path_roles_relation = answer_path_roles_relation.format(
            person_name=person_name,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name, profile_version=profile_version, system_version=system_version,
            benchmark_version=benchmark_version)

    def answer_question_roles_non_relation(self, agent, batch_size=10):
        if batch_size == 1:
            self.answer_single(

                agent=agent,
                benchmark_path=self.roles_non_relation_path_benchmark,
                answer_path=self.answer_path_roles_non_relation
            )
        else:
            self.answer_batch(
                benchmark_path=self.roles_non_relation_path_benchmark,
                answer_path=self.answer_path_roles_non_relation,
                agent=agent,
                split=batch_size)

    def answer_question_roles_relation(self, agent, batch_size=10):
        if batch_size == 1:
            self.answer_single(

                agent=agent,
                benchmark_path=self.roles_relation_path_benchmark,
                answer_path=self.answer_path_roles_relation
            )
        else:
            self.answer_batch(
                benchmark_path=self.roles_relation_path_benchmark,
                answer_path=self.answer_path_roles_relation
                ,
                agent=agent,
                split=batch_size)

    def answer_question_basic_information(self, agent, batch_size=10):

        if batch_size == 1:
            self.answer_single(
                agent=agent,
                benchmark_path=self.basic_information_path_benchmark,
                answer_path=self.answer_path_basic_information
            )
        else:
            self.answer_batch(
                benchmark_path=self.basic_information_path_benchmark,
                answer_path=self.answer_path_basic_information
                ,
                agent=agent,
                split=batch_size)

    def answer_single(self, benchmark_path, answer_path, agent):
        logger.info(
            f'generating answer into file {answer_path}')

        benchmark_q = load_json_file(benchmark_path)
        answer_obj = load_json_file(answer_path)

        prompt_key = f'{self.prompt_kind}_{self.prompt_name}'

        if prompt_key not in answer_obj.keys():
            answer_obj[prompt_key] = []

        if len(answer_obj[prompt_key]) != len(benchmark_q):
            answer_obj[prompt_key] = copy.deepcopy(benchmark_q)

        # explain of the question, to make the model understand the question
        prompt_question = load_prompt(self.prompt_name, self.prompt_kind)

        for count in tqdm(range(len(benchmark_q))):
            if f'generated_answer_{agent.model_name}' in answer_obj[prompt_key][count].keys() and \
                    answer_obj[prompt_key][count][f'generated_answer_{agent.model_name}'] is not None:
                continue

            orig_question = prompt_question
            question = benchmark_q[count]['question']

            orig_question += f"question:{question}\n"
            orig_question += "Your answer:\n"

            result = agent.run(orig_question)

            try:
                answer_obj[prompt_key][count][f'generated_answer_{agent.model_name}'] = result

                # save every 5 questions
                # in case of the program crash, lose all the answers
                if (count + 1) % 1 == 0:
                    with open(answer_path, 'w') as f:
                        json.dump(answer_obj, f, indent=4)

                if count == len(benchmark_q) - 1:
                    with open(answer_path, 'w') as f:
                        json.dump(answer_obj, f, indent=4)

            except Exception as e:
                print(e)
            finally:
                agent.clear()

    def answer_batch(self, benchmark_path, answer_path, agent, split=6):
        """
        always make sure the split is an even number;
        when the program crash, the even split can make sure the question not missed
        """

        logger.info(
            f'generating answer into file {answer_path}')

        benchmark_q = load_json_file(benchmark_path)
        answer_obj = load_json_file(answer_path)
        prompt_key = f'{self.prompt_kind}_{self.prompt_name}'
        if prompt_key not in answer_obj.keys():
            answer_obj[prompt_key] = []

        if len(answer_obj[prompt_key]) != len(benchmark_q):
            answer_obj[prompt_key] = copy.deepcopy(benchmark_q)

        # explain of the question, to make the model understand the question
        prompt_question = load_prompt(self.prompt_name, self.prompt_kind)

        count_list = [i for i in range(len(benchmark_q))]
        chunk_list = [count_list[index:index + split] for index in range(0, len(benchmark_q), split)]

        for chunk in tqdm(chunk_list):
            if f'generated_answer_{agent.model_name}' in answer_obj[prompt_key][chunk[0]].keys() and \
                    answer_obj[prompt_key][chunk[0]][f'generated_answer_{agent.model_name}'] is not None:
                continue

            orig_question = prompt_question
            orig_question += f"The answer should be in the format of json list as :" \
                             f"[<answer for question1>...<answer for question{len(chunk)}>]\n"

            for count in chunk:
                question = answer_obj[prompt_key][count]['question']

                orig_question += f"question:{question}\n"

            orig_question += "Your answer:\n"

            result = agent.run_agent_on_benchmark(orig_question)
            print(result)
            results = json.loads(result)

            try:
                if len(results) != len(chunk):
                    raise Exception("the number of answers is not equal to the number of questions")
                else:
                    for count in chunk:
                        answer_obj[prompt_key][count][f'generated_answer_{agent.model_name}'] = results[
                            count % split]
                    with open(answer_path, 'w') as f:
                        json.dump(answer_obj, f, indent=4)
            except Exception as e:
                print(e)
            finally:
                agent.clear()


def run_agent_on_benchmark(
        person_name, prompt_name, profile_version, system_version, benchmark_version, batch_size, agent):
    """
    run agent on the benchmark, to get the answer for the questions in the benchmark
    Args:
        person_name:
        prompt_name:
        profile_version:
        system_version:
        benchmark_version:
        batch_size:
        agent:

    Returns:

    """
    generator = BenchmarkTest(
        person_name=person_name, prompt_name=prompt_name, prompt_kind="few_shot",
        profile_version=profile_version, system_version=system_version,
        benchmark_version=benchmark_version
    )
    generator.answer_question_basic_information(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_non_relation(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_relation(agent=agent, batch_size=batch_size)
    generator = BenchmarkTest(
        person_name=person_name, prompt_name=prompt_name, prompt_kind="zero_shot",
        profile_version=profile_version, system_version=system_version,
        benchmark_version=benchmark_version
    )
    generator.answer_question_basic_information(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_non_relation(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_relation(agent=agent, batch_size=batch_size)


if __name__ == "__main__":
    person_name = 'monica'
    prompt_name = "prompt1"
    profile_version = "profile_v1"
    system_version = "system_v1"
    benchmark_version = "benchmark_v2"
    batch_size = 1
    agent = Agent(profile_version=profile_version,
                  system_version=system_version,
                  person_name=person_name,
                  model_name='gpt-3.5-turbo-16k')
    run_agent_on_benchmark(
        person_name, prompt_name, profile_version, system_version, benchmark_version, batch_size, agent)

    """generator = BenchmarkQuestionGenerator(
        person_name=person_name,
        prompt_name=prompt_name,
        prompt_kind="few_shot",
        profile_version=profile_version,
        system_version=system_version,
        benchmark_version=benchmark_version)
    generator.write_question_basic_information_file()
    generator.write_question_roles_non_relation_file()
    generator.write_question_roles_relation_file()"""

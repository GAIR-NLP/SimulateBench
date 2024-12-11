import json
import time
import copy
import os

from tqdm import tqdm

from benchmark.benchmark_file_util import (
    QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
    QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
    QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
    COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
    COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
    COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
    ANSWER_PATH_BASIC_INFORMATION_Q,
    ANSWER_PATH_ROLES_NON_RELATION_Q,
    ANSWER_PATH_ROLES_RELATION_Q,
)

from benchmark.benchmark_file_util import load_json_file, load_prompt
from inference.model import Llama
from person.profile.role import load_roles_categories_and_des_person


# set available gpu devices
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Benchmark:
    """_summary_"""

    def __init__(
        self,
        person_name,
        prompt_name,
        prompt_kind,
        profile_version,
        system_version,
        benchmark_version,
        template_question_version,
        basic_information_path_template=COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
        roles_non_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
        roles_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
        basic_information_path_benchmark=QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
        roles_non_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
        roles_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
    ):
        """

        Args:
            person_name:
            prompt_name:
            prompt_kind:
            profile_version:
            system_version:
            benchmark_version:
            template_question_version: only useful when add new questions, For class BenchmarkQuestionGenerator
            basic_information_path_template:
            roles_non_relation_path_template:
            roles_relation_path_template:
            basic_information_path_benchmark:
            roles_non_relation_path_benchmark:
            roles_relation_path_benchmark:
        """
        self.basic_information_path_template = basic_information_path_template.format(
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            template_question_version=template_question_version,
        )
        self.roles_non_relation_path_template = roles_non_relation_path_template.format(
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            template_question_version=template_question_version,
        )
        self.roles_relation_path_template = roles_relation_path_template.format(
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            template_question_version=template_question_version,
        )

        self.basic_information_path_benchmark = basic_information_path_benchmark.format(
            person_name=person_name,
            benchmark_version=benchmark_version,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            profile_version=profile_version,
            system_version=system_version,
        )

        self.roles_non_relation_path_benchmark = (
            roles_non_relation_path_benchmark.format(
                person_name=person_name,
                profile_version=profile_version,
                system_version=system_version,
                benchmark_version=benchmark_version,
                prompt_kind=prompt_kind,
                prompt_name=prompt_name,
            )
        )
        self.roles_relation_path_benchmark = roles_relation_path_benchmark.format(
            person_name=person_name,
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
        )

        self.person_name = person_name
        self.prompt_kind = prompt_kind
        self.prompt_name = prompt_name

        self.profile_version = profile_version
        self.system_version = system_version
        self.benchmark_version = benchmark_version

        # include the person's roles and other related in this role
        self.relations = load_roles_categories_and_des_person(self.person_name)


class BenchmarkTest(Benchmark):
    """_summary_

    Args:
        Benchmark (_type_): _description_
    """

    def __init__(
        self,
        person_name,
        prompt_name,
        prompt_kind,
        profile_version,
        system_version,
        benchmark_version,
        template_question_version,
        basic_information_path_template=COMMON_QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
        roles_non_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
        roles_relation_path_template=COMMON_QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
        basic_information_path_benchmark=QUESTIONNAIRE_PATH_BASIC_INFORMATION_Q,
        roles_non_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_NON_RELATION_Q,
        roles_relation_path_benchmark=QUESTIONNAIRE_PATH_ROLES_RELATION_Q,
        answer_path_basic_information=ANSWER_PATH_BASIC_INFORMATION_Q,
        answer_path_roles_non_relation=ANSWER_PATH_ROLES_NON_RELATION_Q,
        answer_path_roles_relation=ANSWER_PATH_ROLES_RELATION_Q,
        time_sleep=0.1,
    ):
        self.time_sleep = time_sleep
        super(BenchmarkTest, self).__init__(
            basic_information_path_template=basic_information_path_template,
            roles_non_relation_path_template=roles_non_relation_path_template,
            roles_relation_path_template=roles_relation_path_template,
            basic_information_path_benchmark=basic_information_path_benchmark,
            roles_non_relation_path_benchmark=roles_non_relation_path_benchmark,
            roles_relation_path_benchmark=roles_relation_path_benchmark,
            person_name=person_name,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            template_question_version=template_question_version,
        )

        self.answer_path_basic_information = answer_path_basic_information.format(
            person_name=person_name,
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
        )
        self.answer_path_roles_non_relation = answer_path_roles_non_relation.format(
            person_name=person_name,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
        )
        self.answer_path_roles_relation = answer_path_roles_relation.format(
            person_name=person_name,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
        )

    def answer_question_roles_non_relation(self, agent, batch_size=10):
        if batch_size == 1:
            self.answer_single(
                agent=agent,
                benchmark_path=self.roles_non_relation_path_benchmark,
                answer_path=self.answer_path_roles_non_relation,
            )
        else:
            self.answer_batch(
                benchmark_path=self.roles_non_relation_path_benchmark,
                answer_path=self.answer_path_roles_non_relation,
                agent=agent,
                split=batch_size,
            )

    def answer_question_roles_relation(self, agent, batch_size=10):
        if batch_size == 1:
            self.answer_single(
                agent=agent,
                benchmark_path=self.roles_relation_path_benchmark,
                answer_path=self.answer_path_roles_relation,
            )
        else:
            self.answer_batch(
                benchmark_path=self.roles_relation_path_benchmark,
                answer_path=self.answer_path_roles_relation,
                agent=agent,
                split=batch_size,
            )

    def answer_question_basic_information(self, agent, batch_size=10):

        if batch_size == 1:
            self.answer_single(
                agent=agent,
                benchmark_path=self.basic_information_path_benchmark,
                answer_path=self.answer_path_basic_information,
            )
        else:
            self.answer_batch(
                benchmark_path=self.basic_information_path_benchmark,
                answer_path=self.answer_path_basic_information,
                agent=agent,
                split=batch_size,
            )

    def answer_single(self, benchmark_path, answer_path, agent):
        print(f"generating answer into file {answer_path}")

        benchmark_q = load_json_file(benchmark_path)
        if os.path.exists(answer_path):
            answer_obj = load_json_file(answer_path)
        else:
            answer_obj = {}

        prompt_key = f"{self.prompt_kind}_{self.prompt_name}"

        if prompt_key not in answer_obj.keys():
            answer_obj[prompt_key] = []

        if len(answer_obj[prompt_key]) == 0:
            answer_obj[prompt_key] = copy.deepcopy(benchmark_q)
        elif len(answer_obj[prompt_key]) != len(benchmark_q):
            # for count in range(len(answer_obj[prompt_key]), len(benchmark_q)):
            answer_obj[prompt_key].extend(
                copy.deepcopy(benchmark_q[len(answer_obj[prompt_key]) :])
            )

        # explain of the question, to make the model understand the question
        prompt_question = load_prompt(self.prompt_name, self.prompt_kind)

        for count in tqdm(range(len(benchmark_q))):
            if (
                f"generated_answer_{agent.model_name}"
                in answer_obj[prompt_key][count].keys()
                and answer_obj[prompt_key][count][
                    f"generated_answer_{agent.model_name}"
                ]
                is not None
            ):
                continue

            orig_question = prompt_question
            question = benchmark_q[count]["question"]

            orig_question += f"question:{question}\n"
            orig_question += "Your answer:\n"

            result = agent.run(orig_question)
            answer_obj[prompt_key][count][
                f"generated_answer_{agent.model_name}"
            ] = result

            try:
                # save every 5 questions
                # in case of the program crash, lose all the answers
                if (count + 1) % 5 == 0:
                    with open(answer_path, "w", encoding="UTF-8") as f:
                        json.dump(answer_obj, f, indent=4)
            except Exception as e:
                print(e)
            finally:
                agent.clear()
        
        
        # make sure all the results is stored, incase some step is skipped in the for loop
        with open(answer_path, "w") as f:

            json.dump(answer_obj, f, indent=4)

    def answer_batch(self, benchmark_path, answer_path, agent, split=6):
        """
        always make sure the split is an even number;
        when the program crash, the even split can make sure the question not missed
        """

        print(f"generating answer into file {answer_path}")

        benchmark_q = load_json_file(benchmark_path)
        if os.path.exists(answer_path):
            answer_obj = load_json_file(answer_path)
        else:
            answer_obj = {}

        prompt_key = f"{self.prompt_kind}_{self.prompt_name}"
        if prompt_key not in answer_obj.keys():
            answer_obj[prompt_key] = []

        if len(answer_obj[prompt_key]) == 0:
            answer_obj[prompt_key] = copy.deepcopy(benchmark_q)
        elif len(answer_obj[prompt_key]) != len(benchmark_q):
            # for count in range(len(answer_obj[prompt_key]), len(benchmark_q)):
            answer_obj[prompt_key].extend(
                copy.deepcopy(benchmark_q[len(answer_obj[prompt_key]) :])
            )

        # explain of the question, to make the model understand the question
        prompt_question = load_prompt(self.prompt_name, self.prompt_kind)

        count_list = [i for i in range(len(benchmark_q))]
        chunk_list = [
            count_list[index : index + split]
            for index in range(0, len(benchmark_q), split)
        ]

        # remove the chunk that is already answered
        new_chunk_list = []
        for chunk in chunk_list:
            new_chunk = []
            for count in chunk:
                if (
                    f"generated_answer_{agent.model_name}"
                    not in answer_obj[prompt_key][count].keys()
                    or answer_obj[prompt_key][count][
                        f"generated_answer_{agent.model_name}"
                    ]
                    is None
                ):
                    new_chunk.append(count)
            if len(new_chunk) > 0:
                new_chunk_list.append(new_chunk)
        chunk_list = new_chunk_list

        for chunk in tqdm(chunk_list):
            if agent.model_name == "gpt-4":
                time.sleep(self.time_sleep)
            elif agent.model_name == "gpt-3.5-turbo":
                time.sleep(self.time_sleep)
            orig_question = prompt_question
            orig_question += (
                f"The answer should be in the format of json list as :"
                f"[<answer for question1>...<answer for question{len(chunk)}>]\n"
            )

            for count in chunk:
                question = answer_obj[prompt_key][count]["question"]

                orig_question += f"question:{question}\n"

            orig_question += "Your answer:\n"

            result = agent.run(orig_question)
            print(result)

            try:
                results = json.loads(result)
                if len(results) != len(chunk):
                    raise Exception(
                        "the number of answers is not equal to the number of questions"
                    )
                else:
                    for count in chunk:
                        answer_obj[prompt_key][count][
                            f"generated_answer_{agent.model_name}"
                        ] = results[chunk.index(count)]
                    with open(answer_path, "w", encoding="UTF-8") as f:
                        json.dump(answer_obj, f, indent=4)
            except Exception as e:
                print(e)
            finally:
                agent.clear()
        # make sure all the questions is answered, in case some step is skipped in the for loop for the reason of program crash
        # self.answer_single(benchmark_path, answer_path, agent)
        


def run_agent_on_benchmark(
    person_name,
    prompt_name,
    profile_version,
    system_version,
    benchmark_version,
    batch_size,
    agent,
    time_sleep=0.1,
    few_shot=True,
    zero_shot=True,
):
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
    if few_shot:
        type_ = "few_shot"

    generator = BenchmarkTest(
        person_name=person_name,
        prompt_name=prompt_name,
        prompt_kind=type_,
        profile_version=profile_version,
        system_version=system_version,
        benchmark_version=benchmark_version,
        template_question_version="v2",
        time_sleep=time_sleep,
    )
    generator.answer_question_basic_information(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_non_relation(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_relation(agent=agent, batch_size=batch_size)
    
    if zero_shot:
        type_ = "zero_shot"
    generator = BenchmarkTest(
        person_name=person_name,
        prompt_name=prompt_name,
        prompt_kind=type_,
        profile_version=profile_version,
        system_version=system_version,
        benchmark_version=benchmark_version,
        template_question_version="v2",
        time_sleep=time_sleep,
    )
    generator.answer_question_basic_information(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_non_relation(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_relation(agent=agent, batch_size=batch_size)


def run_agent_on_benchmark_single(
    person_name,
    prompt_name,
    profile_version,
    system_version,
    benchmark_version,
    batch_size,
    agent,
    prompt_kind,
    time_sleep=0.1,
):
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
        person_name=person_name,
        prompt_name=prompt_name,
        prompt_kind=prompt_kind,
        profile_version=profile_version,
        system_version=system_version,
        benchmark_version=benchmark_version,
        template_question_version="v2",
        time_sleep=time_sleep,
    )
    generator.answer_question_basic_information(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_non_relation(agent=agent, batch_size=batch_size)
    generator.answer_question_roles_relation(agent=agent, batch_size=batch_size)


def run_agent_on_benchmark_roles(
    person_name,
    prompt_name,
    profile_version,
    system_version,
    benchmark_version,
    batch_size,
    agent,
    time_sleep=0.1,
    zero_shot=True,
    few_shot=True,
):
    """
    run agent on the benchmark, to get the answer for the roles questions in the benchmark
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
    if few_shot:
        generator = BenchmarkTest(
            person_name=person_name,
            prompt_name=prompt_name,
            prompt_kind="few_shot",
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            template_question_version="v2",
            time_sleep=time_sleep,
        )

        generator.answer_question_roles_non_relation(agent=agent, batch_size=batch_size)
        generator.answer_question_roles_relation(agent=agent, batch_size=batch_size)

    if zero_shot:
        generator = BenchmarkTest(
            person_name=person_name,
            prompt_name=prompt_name,
            prompt_kind="zero_shot",
            profile_version=profile_version,
            system_version=system_version,
            benchmark_version=benchmark_version,
            template_question_version="v2",
            time_sleep=time_sleep,
        )

        generator.answer_question_roles_non_relation(agent=agent, batch_size=batch_size)
        generator.answer_question_roles_relation(agent=agent, batch_size=batch_size)


if __name__ == "__main__":
    # model_name='gpt-3.5-turbo-16k'
    for model_name in ["meta-llama/Llama-3.2-3B-Instruct"]:
        for person_name in ["homer"]:
            prompt_name = "prompt1"
            profile_version = "profile_v1"
            system_version = "system_v1"
            benchmark_version = "benchmark_v2"
            batch_size = 1
            time_sleep = 0
            if model_name == "gpt-3.5-turbo-16k":
                batch_size = 16
            elif model_name == "gpt-4":
                batch_size = 16
            agent = Llama(
                profile_version=profile_version,
                system_version=system_version,
                person_name=person_name,
                model_name=model_name,
            )
            run_agent_on_benchmark(
                person_name,
                prompt_name,
                profile_version,
                system_version,
                benchmark_version,
                batch_size,
                agent,
                time_sleep=time_sleep,
            )

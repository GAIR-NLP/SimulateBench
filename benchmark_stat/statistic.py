import csv
import os

import numpy as np

from benchmark.benchmark_file_util import ROOT_PATH, load_json_file
from log.logger import logger
csv_head_row_models_single_name = ["model", "mean", "basic_information_answerable",
                                   "basic_information_unanswerable",
                                   "roles_non_relation_answerable", "roles_non_relation_unanswerable",
                                   "roles_relation_answerable", "roles_relation_unanswerable"]
csv_head_row_models_names_mean = ["model"]  # concat with character names
csv_root_path = ROOT_PATH + "/statistic/{ablation_kind}/"


def calculate_mean_1(re_ba, re_non_re, re_re):
    # use the ration of all kinds of questions in the whole benchmark
    number_all_all = re_ba["number_all"] + re_non_re["number_all"] + re_re["number_all"]
    mean_ba_answerable = re_ba['number_answerable'] / number_all_all * re_ba['accuracy_answerable']
    mean_ba_unanswerable = re_ba['number_unanswerable'] / number_all_all * re_ba['accuracy_unanswerable']
    mean_non_re_answerable = re_non_re['number_answerable'] / number_all_all * re_non_re['accuracy_answerable']
    mean_non_re_unanswerable = re_non_re['number_unanswerable'] / number_all_all * re_non_re[
        'accuracy_unanswerable']
    mean_re_answerable = re_re['number_answerable'] / number_all_all * re_re['accuracy_answerable']
    mean_re_unanswerable = re_re['number_unanswerable'] / number_all_all * re_re[
        'accuracy_unanswerable']
    mean = mean_re_unanswerable + mean_re_answerable + mean_non_re_unanswerable + mean_ba_unanswerable + mean_ba_answerable + mean_non_re_answerable

    return round(mean, 3)


def calculate_mean_2(re_ba, re_non_re, re_re):
    # use the ration of all kinds of questions in the whole benchmark
    # answerable count for 0.8; unanswerable count for 0.2
    answerable_coefficient = 0.9
    unanswerable_coefficient = 1 - answerable_coefficient
    number_all_all = re_ba["number_all"] + re_non_re["number_all"] + re_re["number_all"]
    mean_ba_answerable = re_ba['number_all'] / number_all_all * re_ba['accuracy_answerable']
    mean_ba_unanswerable = re_ba['number_all'] / number_all_all * re_ba['accuracy_unanswerable']
    mean_non_re_answerable = re_non_re['number_all'] / number_all_all * re_non_re['accuracy_answerable']
    mean_non_re_unanswerable = re_non_re['number_all'] / number_all_all * re_non_re[
        'accuracy_unanswerable']
    mean_re_answerable = re_re['number_all'] / number_all_all * re_re['accuracy_answerable']
    mean_re_unanswerable = re_re['number_all'] / number_all_all * re_re[
        'accuracy_unanswerable']
    mean = (mean_re_unanswerable + mean_non_re_unanswerable + mean_ba_unanswerable) * unanswerable_coefficient + \
           (mean_ba_answerable + mean_non_re_answerable + mean_re_answerable) * answerable_coefficient

    return round(mean, 3)


def get_calculate_fun(fun_name):
    if fun_name == "ration_of_number":
        return calculate_mean_1
    elif fun_name == "answerable_9":
        return calculate_mean_2
    else:
        return calculate_mean_2


class Statistic:
    def __init__(self, person_name, model_name, prompt_kind, prompt_name, benchmark_version, profile_version,
                 system_version):
        self.benchmark_version = benchmark_version
        self.profile_version = profile_version
        self.system_version = system_version
        self.person_name = person_name
        self.model_name = model_name
        self.prompt_kind = prompt_kind
        self.prompt_name = prompt_name
        self.correct_answerable = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0
        }
        self.correct_unanswerable = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0
        }
        self.number_answerable = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0
        }
        self.number_unanswerable = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0
        }
        self.number_all = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0
        }

    def count_number(self, benchmark_type):
        obj = load_json_file(
            f"{ROOT_PATH}/{benchmark_type}/{self.person_name}/{self.profile_version}/{self.system_version}/{self.benchmark_version}/questions.json")

        unanswerable = 0

        for count in range(len(obj)):
            question_odj = obj[count]
            if "there's not enough information to answer this question" in question_odj['gold_answer'].lower():
                unanswerable += 1

        answer_able = len(obj) - unanswerable

        self.number_answerable[benchmark_type] = answer_able
        self.number_unanswerable[benchmark_type] = unanswerable
        self.number_all[benchmark_type] = len(obj)

    def count_correct(self, benchmark_type):
        obj = load_json_file(
            f"{ROOT_PATH}/{benchmark_type}/{self.person_name}/{self.profile_version}/{self.system_version}/{self.benchmark_version}/{self.prompt_kind}/{self.prompt_name}.json")
        obj = obj[f"{self.prompt_kind}_{self.prompt_name}"]
        answerable_list = []
        unanswerable_list = []

        for count in range(len(obj)):
            question_odj = obj[count]
            if "there's not enough information to answer this question" in question_odj['gold_answer'].lower():
                unanswerable_list.append(question_odj)
            else:
                answerable_list.append(question_odj)

        correct_answerable = self.count_correct_part(answerable_list)

        correct_unanswerable = self.count_correct_part(unanswerable_list)

        self.correct_answerable[benchmark_type] = correct_answerable
        self.correct_unanswerable[benchmark_type] = correct_unanswerable

    def count_correct_part(self, answer_list):
        result = 0
        for question_odj in answer_list:

            if question_odj['gold_answer'].lower() == question_odj[f'generated_answer_{self.model_name}'].lower():
                result += 1
            elif question_odj[f'generated_answer_{self.model_name}'].lower() in question_odj['gold_answer'].lower():
                result += 1
            elif question_odj['gold_answer'].lower() in question_odj[f'generated_answer_{self.model_name}'].lower():
                result += 1
        return result

    def show_accuracy(self, benchmark_type):
        self.count_number(benchmark_type)
        self.count_correct(benchmark_type)

        if self.number_answerable[benchmark_type] != 0:
            accuracy_answerable = self.correct_answerable[benchmark_type] / self.number_answerable[benchmark_type]
        else:
            accuracy_answerable = '~'

        if self.number_unanswerable[benchmark_type] != 0:
            """print(self.correct_unanswerable)
            print(self.number_unanswerable)"""
            accuracy_unanswerable = self.correct_unanswerable[benchmark_type] / self.number_unanswerable[benchmark_type]
        else:
            accuracy_unanswerable = '~'

        print(f"{benchmark_type} {self.model_name} {self.prompt_kind} {self.prompt_name} {self.person_name}\n"
              f"accuracy_answerable: {accuracy_answerable}; accuracy_unanswerable: {accuracy_unanswerable}\n "
              f"number_answerable: {self.number_answerable[benchmark_type]}; "
              f"number_unanswerable: {self.number_unanswerable[benchmark_type]} "
              f"number_all: {self.number_all[benchmark_type]}\n")

        results = {
            "number_all": self.number_all[benchmark_type],
            "number_answerable": self.number_answerable[benchmark_type],
            "number_unanswerable": self.number_unanswerable[benchmark_type],
            "number_answerable_correct": self.correct_answerable[benchmark_type],
            "number_unanswerable_correct": self.correct_unanswerable[benchmark_type],
            "accuracy_answerable": accuracy_answerable,
            "accuracy_unanswerable": accuracy_unanswerable
        }
        return results

    def show_results(self, calculate_mean_fun_name):
        re_ba = self.show_accuracy("basic_information")
        re_non_re = self.show_accuracy("role_non_relation")
        re_re = self.show_accuracy("role_relation")

        # use name to get fun
        # function factory
        mean = get_calculate_fun(calculate_mean_fun_name)(re_ba, re_non_re, re_re)

        results = {
            "basic_information": re_ba,
            "role_non_relation": re_non_re,
            "role_relation": re_re,
            "mean": mean,
            "all_number": re_ba["number_all"] + re_non_re["number_all"] + re_re["number_all"]
        }

        return results


def make_csv_file_models_single_person(prompt_kind, model_name_list, calculate_mean_fun_name, ablation_kind,
                                       full_name_list, benchmark_version, profile_version, system_version, prompt_name):
    for person_name in full_name_list:
        results_all = {}
        for model_name in model_name_list:
            sta1 = Statistic(
                person_name=person_name,
                model_name=model_name,
                prompt_kind=prompt_kind,
                prompt_name=prompt_name,
                benchmark_version=benchmark_version,
                profile_version=profile_version,
                system_version=system_version
            )
            results = sta1.show_results(calculate_mean_fun_name)
            result_results = {
                "model": model_name,
                "mean": results['mean'],
                "basic_information_answerable": round(results["basic_information"]["accuracy_answerable"], 3),
                "basic_information_unanswerable": round(results["basic_information"]["accuracy_unanswerable"], 3),
                "roles_non_relation_answerable": round(results["role_non_relation"]["accuracy_answerable"], 3),
                "roles_non_relation_unanswerable": round(results["role_non_relation"]["accuracy_unanswerable"], 3),
                "roles_relation_answerable": round(results["role_relation"]["accuracy_answerable"], 3),
                "roles_relation_unanswerable": round(results["role_relation"]["accuracy_unanswerable"], 3)
            }
            results_all[model_name] = result_results

        path_ = f"{csv_root_path.format(ablation_kind=ablation_kind)}/{prompt_kind}/"
        if not os.path.exists(path_):
            os.makedirs(path_)

        path_person = f"{path_}/{person_name}_{calculate_mean_fun_name}.csv"
        with open(path_person, mode='w') as _file:
            _writer = csv.DictWriter(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                     fieldnames=csv_head_row_models_single_name)
            _writer.writeheader()
            for model_name in model_name_list:
                _writer.writerow(results_all[model_name])


def make_csv_file_models_all_people_ablation(prompt_kind, model_name_list, benchmark_version, profile_version,
                                             system_version, full_name_list, prompt_name,
                                             ablation_kind, calculate_mean_fun_name):
    """
    all the  names of people origin from the same person
    Args:
        prompt_kind:
        model_name_list:
        benchmark_version:
        profile_version:
        system_version:
        full_name_list:
        prompt_name:
        ablation_kind:
        calculate_mean_fun_name:

    Returns:

    """
    mean_dict = {}

    # collect the mean scores for the model on all person

    number_of_all_question = None
    for model_name in model_name_list:
        mean_list = []
        for person_name in full_name_list:
            sta1 = Statistic(
                person_name=person_name,
                model_name=model_name,
                prompt_kind=prompt_kind,
                prompt_name=prompt_name,
                benchmark_version=benchmark_version,
                profile_version=profile_version,
                system_version=system_version
            )
            results = sta1.show_results(calculate_mean_fun_name)
            mean_list.append(results['mean'])

            if number_of_all_question is None:
                number_of_all_question = results['all_number']

            """# make csv file of the model on the person for all kind of benchmark file type
            path_ = f"{csv_root_path.format(ablation_kind='age')}/{model_name}/"
            if not os.path.exists(path_):
                os.makedirs(path_)
            path_person= f"{path_}/{person_name}.csv"""

        # calculate the variance of the mean scores
        mean = round(np.mean(mean_list).item(), 4)
        var = np.std(mean_list, ddof=1).item()
        var = round(var, 4)
        range_ = round((max(mean_list) - min(mean_list)) * number_of_all_question)
        mean_dict[model_name] = [mean, var] + [range_] + mean_list

    # make the dir for different prompt kind
    path_ = f"{csv_root_path.format(ablation_kind=ablation_kind)}/{prompt_kind}/"
    if not os.path.exists(path_):
        os.makedirs(path_)

    # write file
    path_all = f"{path_}/all_{prompt_kind}_{calculate_mean_fun_name}.csv"
    logger.info(f'write file: {path_all}')
    with open(path_all, mode='w') as _file:
        _writer = csv.writer(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        _writer.writerow(
            csv_head_row_models_names_mean + ["mean", "variance",
                                              f"range(total:{number_of_all_question})"] + full_name_list)
        for model_name in model_name_list:
            _writer.writerow([model_name] + mean_dict[model_name])


if __name__ == "__main__":
    model_name = "longchat-13b-16k"

    benchmark_version = "benchmark_v2"
    profile_version = "profile_v1"
    system_version = "system_v1"

    prompt_name = "prompt1"
    prompt_kind = "few_shot"
    person_name = "monica"
    sta1 = Statistic(
        person_name=person_name,
        model_name=model_name,
        prompt_kind=prompt_kind,
        prompt_name=prompt_name,
        benchmark_version=benchmark_version,
        profile_version=profile_version,
        system_version=system_version
    )
    sta1.show_results()

    prompt_kind = "zero_shot"
    sta1 = Statistic(person_name=person_name,
                     model_name=model_name,
                     prompt_kind=prompt_kind,
                     prompt_name=prompt_name,
                     benchmark_version=benchmark_version,
                     profile_version=profile_version,
                     system_version=system_version)
    sta1.show_results()

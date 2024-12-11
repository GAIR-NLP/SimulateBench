import csv
import os
import re
import numpy as np

from benchmark.benchmark_file_util import ROOT_PATH, load_json_file

csv_head_row_models_single_name = [
    "model",
    "mean",
    "basic_information_answerable",
    "basic_information_unanswerable",
    "roles_non_relation_answerable",
    "roles_non_relation_unanswerable",
    "roles_relation_answerable",
    "roles_relation_unanswerable",
]
csv_head_row_models_names_mean = ["model"]  # concat with character names
csv_root_path_ablation = ROOT_PATH + "/statistic/{ablation_kind}/"
csv_root_path_coherence_task = ROOT_PATH + "/statistic/character/{character_name}/"


def calculate_mean_1(re_ba, re_non_re, re_re):
    # use the ration of all kinds of questions in the whole benchmark
    number_all_all = re_ba["number_all"] + re_non_re["number_all"] + re_re["number_all"]
    mean_ba_answerable = (
        re_ba["number_answerable"] / number_all_all * re_ba["accuracy_answerable"]
    )
    mean_ba_unanswerable = (
        re_ba["number_unanswerable"] / number_all_all * re_ba["accuracy_unanswerable"]
    )
    mean_non_re_answerable = (
        re_non_re["number_answerable"]
        / number_all_all
        * re_non_re["accuracy_answerable"]
    )
    mean_non_re_unanswerable = (
        re_non_re["number_unanswerable"]
        / number_all_all
        * re_non_re["accuracy_unanswerable"]
    )
    mean_re_answerable = (
        re_re["number_answerable"] / number_all_all * re_re["accuracy_answerable"]
    )
    mean_re_unanswerable = (
        re_re["number_unanswerable"] / number_all_all * re_re["accuracy_unanswerable"]
    )
    mean = (
        mean_re_unanswerable
        + mean_re_answerable
        + mean_non_re_unanswerable
        + mean_ba_unanswerable
        + mean_ba_answerable
        + mean_non_re_answerable
    )

    return mean


def calculate_mean_2(re_ba, re_non_re, re_re):
    # use the ration of all kinds of questions in the whole benchmark
    # answerable count for 0.8; unanswerable count for 0.2
    answerable_coefficient = 0.9
    unanswerable_coefficient = 1 - answerable_coefficient
    number_all_all = re_ba["number_all"] + re_non_re["number_all"] + re_re["number_all"]
    mean_ba_answerable = (
        re_ba["number_all"] / number_all_all * re_ba["accuracy_answerable"]
    )
    mean_ba_unanswerable = (
        re_ba["number_all"] / number_all_all * re_ba["accuracy_unanswerable"]
    )
    mean_non_re_answerable = (
        re_non_re["number_all"] / number_all_all * re_non_re["accuracy_answerable"]
    )
    mean_non_re_unanswerable = (
        re_non_re["number_all"] / number_all_all * re_non_re["accuracy_unanswerable"]
    )
    mean_re_answerable = (
        re_re["number_all"] / number_all_all * re_re["accuracy_answerable"]
    )
    mean_re_unanswerable = (
        re_re["number_all"] / number_all_all * re_re["accuracy_unanswerable"]
    )
    mean = (
        mean_re_unanswerable + mean_non_re_unanswerable + mean_ba_unanswerable
    ) * unanswerable_coefficient + (
        mean_ba_answerable + mean_non_re_answerable + mean_re_answerable
    ) * answerable_coefficient

    return mean


def get_calculate_fun(fun_name):
    if fun_name == "ration_of_number":
        return calculate_mean_1
    elif fun_name == "answerable_9":
        return calculate_mean_2
    else:
        return calculate_mean_2


class Statistic:
    def __init__(
        self,
        person_name,
        model_name,
        prompt_kind,
        prompt_name,
        benchmark_version,
        profile_version,
        system_version,
    ):
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
            "role_non_relation": 0,
        }
        self.correct_unanswerable = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0,
        }
        self.number_answerable = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0,
        }
        self.number_unanswerable = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0,
        }
        self.number_all = {
            "basic_information": 0,
            "role_relation": 0,
            "role_non_relation": 0,
        }

    def count_number(self, benchmark_type):
        obj = load_json_file(
            f"{ROOT_PATH}/{benchmark_type}/{self.person_name}/{self.profile_version}/{self.system_version}/{self.benchmark_version}/questions.json"
        )

        unanswerable = 0

        for count in range(len(obj)):
            question_odj = obj[count]
            if (
                "there's not enough information to answer this question"
                in question_odj["gold_answer"].lower()
            ):
                unanswerable += 1

        answer_able = len(obj) - unanswerable

        self.number_answerable[benchmark_type] = answer_able
        self.number_unanswerable[benchmark_type] = unanswerable
        self.number_all[benchmark_type] = len(obj)

    def count_correct(self, benchmark_type):
        obj = load_json_file(
            f"{ROOT_PATH}/{benchmark_type}/{self.person_name}/{self.profile_version}/{self.system_version}/{self.benchmark_version}/{self.prompt_kind}/{self.prompt_name}.json"
        )
        obj = obj[f"{self.prompt_kind}_{self.prompt_name}"]
        answerable_list = []
        unanswerable_list = []

        for count in range(len(obj)):
            question_odj = obj[count]
            if (
                "there's not enough information to answer this question"
                in question_odj["gold_answer"].lower()
            ):
                unanswerable_list.append(question_odj)
            else:
                answerable_list.append(question_odj)

        print(benchmark_type)
        correct_answerable = self.count_correct_part(answerable_list)

        correct_unanswerable = self.count_correct_part(unanswerable_list)

        self.correct_answerable[benchmark_type] = correct_answerable
        self.correct_unanswerable[benchmark_type] = correct_unanswerable

    def count_correct_part(self, answer_list):
        print(
            self.person_name
            + "_"
            + self.model_name
            + "_"
            + self.prompt_kind
            + "_"
            + self.prompt_name
        )
        result = 0
        for question_odj in answer_list:
            print(question_odj.keys())
            if (
                question_odj["gold_answer"].lower()
                == str(question_odj[f"generated_answer_{self.model_name}"]).lower()
            ):
                result += 1
            elif (
                str(question_odj[f"generated_answer_{self.model_name}"]).lower()
                in question_odj["gold_answer"].lower()
            ):
                result += 1
            elif (
                question_odj["gold_answer"].lower()
                in str(question_odj[f"generated_answer_{self.model_name}"]).lower()
            ):
                result += 1
        return result

    def count_correct_part_show_to_list_true_false(self, answer_list):

        result = []
        for question_odj in answer_list:
            print(question_odj.keys())
            if (
                question_odj["gold_answer"].lower()
                == str(question_odj[f"generated_answer_{self.model_name}"]).lower()
            ):
                result.append(True)
            elif (
                str(question_odj[f"generated_answer_{self.model_name}"]).lower()
                in question_odj["gold_answer"].lower()
            ):
                result.append(True)
            elif (
                question_odj["gold_answer"].lower()
                in str(question_odj[f"generated_answer_{self.model_name}"]).lower()
            ):
                result.append(True)
            else:
                result.append(False)
        return result

    def show_accuracy(self, benchmark_type):
        self.count_number(benchmark_type)
        self.count_correct(benchmark_type)

        if self.number_answerable[benchmark_type] != 0:
            accuracy_answerable = (
                self.correct_answerable[benchmark_type]
                / self.number_answerable[benchmark_type]
            )
        else:
            accuracy_answerable = "~"

        if self.number_unanswerable[benchmark_type] != 0:
            """print(self.correct_unanswerable)
            print(self.number_unanswerable)"""
            accuracy_unanswerable = (
                self.correct_unanswerable[benchmark_type]
                / self.number_unanswerable[benchmark_type]
            )
        else:
            accuracy_unanswerable = "~"

        print(
            f"{benchmark_type} {self.model_name} {self.prompt_kind} {self.prompt_name} {self.person_name}\n"
            f"accuracy_answerable: {accuracy_answerable}; accuracy_unanswerable: {accuracy_unanswerable}\n "
            f"number_answerable: {self.number_answerable[benchmark_type]}; "
            f"number_unanswerable: {self.number_unanswerable[benchmark_type]} "
            f"number_all: {self.number_all[benchmark_type]}\n"
        )

        results = {
            "number_all": self.number_all[benchmark_type],
            "number_answerable": self.number_answerable[benchmark_type],
            "number_unanswerable": self.number_unanswerable[benchmark_type],
            "number_answerable_correct": self.correct_answerable[benchmark_type],
            "number_unanswerable_correct": self.correct_unanswerable[benchmark_type],
            "accuracy_answerable": accuracy_answerable,
            "accuracy_unanswerable": accuracy_unanswerable,
        }
        return results

    def show_results(self, calculate_mean_fun_name):
        re_ba = self.show_accuracy("basic_information")
        re_non_re = self.show_accuracy("role_non_relation")
        re_re = self.show_accuracy("role_relation")

        # use name to get fun
        # function factory
        mean = get_calculate_fun(calculate_mean_fun_name)(re_ba, re_non_re, re_re)

        print(f'non relationship answerable number:{re_non_re["number_answerable"]}')
        print(
            f'non relationship unanswerable number:{re_non_re["number_unanswerable"]}'
        )
        print(f'relationship answerable number:{re_re["number_answerable"]}')
        print(f'relationship unanswerable number:{re_re["number_unanswerable"]}')

        results = {
            "basic_information": re_ba,
            "role_non_relation": re_non_re,
            "role_relation": re_re,
            "mean": mean,
            "all_number": re_ba["number_all"]
            + re_non_re["number_all"]
            + re_re["number_all"],
        }

        return results

    def show_to_list_true_false_(self, benchmark_type):
        obj = load_json_file(
            f"{ROOT_PATH}/{benchmark_type}/{self.person_name}/{self.profile_version}/{self.system_version}/{self.benchmark_version}/{self.prompt_kind}/{self.prompt_name}.json"
        )
        obj = obj[f"{self.prompt_kind}_{self.prompt_name}"]
        answerable_list = []
        unanswerable_list = []

        for count in range(len(obj)):
            question_odj = obj[count]
            if (
                "there's not enough information to answer this question"
                in question_odj["gold_answer"].lower()
            ):
                unanswerable_list.append(question_odj)
            else:
                answerable_list.append(question_odj)

        answerable_list_results = self.count_correct_part_show_to_list_true_false(
            answerable_list
        )
        unanswerable_list_results = self.count_correct_part_show_to_list_true_false(
            unanswerable_list
        )
        return answerable_list_results, unanswerable_list_results
        # print(benchmark_type)
        # correct_answerable = self.count_correct_part(answerable_list)

        # correct_unanswerable = self.count_correct_part(unanswerable_list)

        # self.correct_answerable[benchmark_type] = correct_answerable
        # self.correct_unanswerable[benchmark_type] = correct_unanswerable

    def show_to_list_true_false(self):
        re_ba_answer, re_ba_no = self.show_to_list_true_false_("basic_information")
        re_non_re_answer, re_non_re_no = self.show_to_list_true_false_(
            "role_non_relation"
        )
        re_re_answer, re_re_no = self.show_to_list_true_false_("role_relation")

        return (
            re_ba_answer,
            re_ba_no,
            re_non_re_answer,
            re_non_re_no,
            re_re_answer,
            re_re_no,
        )


class StatisticNoBasicInformation:
    def __init__(
        self,
        person_name,
        model_name,
        prompt_kind,
        prompt_name,
        benchmark_version,
        profile_version,
        system_version,
    ):
        self.benchmark_version = benchmark_version
        self.profile_version = profile_version
        self.system_version = system_version
        self.person_name = person_name
        self.model_name = model_name
        self.prompt_kind = prompt_kind
        self.prompt_name = prompt_name
        self.correct_answerable = {"role_relation": 0, "role_non_relation": 0}
        self.correct_unanswerable = {"role_relation": 0, "role_non_relation": 0}
        self.number_answerable = {"role_relation": 0, "role_non_relation": 0}
        self.number_unanswerable = {"role_relation": 0, "role_non_relation": 0}
        self.number_all = {"role_relation": 0, "role_non_relation": 0}

    def count_number(self, benchmark_type):
        obj = load_json_file(
            f"{ROOT_PATH}/{benchmark_type}/{self.person_name}/{self.profile_version}/{self.system_version}/{self.benchmark_version}/questions.json"
        )

        unanswerable = 0

        for count in range(len(obj)):
            question_odj = obj[count]
            if (
                "there's not enough information to answer this question"
                in question_odj["gold_answer"].lower()
            ):
                unanswerable += 1

        answer_able = len(obj) - unanswerable

        self.number_answerable[benchmark_type] = answer_able
        self.number_unanswerable[benchmark_type] = unanswerable
        self.number_all[benchmark_type] = len(obj)

    def count_correct(self, benchmark_type):
        obj = load_json_file(
            f"{ROOT_PATH}/{benchmark_type}/{self.person_name}/{self.profile_version}/{self.system_version}/{self.benchmark_version}/{self.prompt_kind}/{self.prompt_name}.json"
        )
        obj = obj[f"{self.prompt_kind}_{self.prompt_name}"]
        answerable_list = []
        unanswerable_list = []

        for count in range(len(obj)):
            question_odj = obj[count]
            if (
                "there's not enough information to answer this question"
                in question_odj["gold_answer"].lower()
            ):
                unanswerable_list.append(question_odj)
            else:
                answerable_list.append(question_odj)

        print(benchmark_type)
        correct_answerable = self.count_correct_part(answerable_list)

        correct_unanswerable = self.count_correct_part(unanswerable_list)

        self.correct_answerable[benchmark_type] = correct_answerable
        self.correct_unanswerable[benchmark_type] = correct_unanswerable

    def count_correct_part(self, answer_list):
        print(
            self.person_name
            + "_"
            + self.model_name
            + "_"
            + self.prompt_kind
            + "_"
            + self.prompt_name
        )
        result = 0
        for question_odj in answer_list:
            print(question_odj.keys())
            if (
                question_odj["gold_answer"].lower()
                == str(question_odj[f"generated_answer_{self.model_name}"]).lower()
            ):
                result += 1
            elif (
                str(question_odj[f"generated_answer_{self.model_name}"]).lower()
                in question_odj["gold_answer"].lower()
            ):
                result += 1
            elif (
                question_odj["gold_answer"].lower()
                in str(question_odj[f"generated_answer_{self.model_name}"]).lower()
            ):
                result += 1
        return result

    def show_accuracy(self, benchmark_type):
        self.count_number(benchmark_type)
        self.count_correct(benchmark_type)

        if self.number_answerable[benchmark_type] != 0:
            accuracy_answerable = (
                self.correct_answerable[benchmark_type]
                / self.number_answerable[benchmark_type]
            )
        else:
            accuracy_answerable = "~"

        if self.number_unanswerable[benchmark_type] != 0:
            """print(self.correct_unanswerable)
            print(self.number_unanswerable)"""
            accuracy_unanswerable = (
                self.correct_unanswerable[benchmark_type]
                / self.number_unanswerable[benchmark_type]
            )
        else:
            accuracy_unanswerable = "~"

        print(
            f"{benchmark_type} {self.model_name} {self.prompt_kind} {self.prompt_name} {self.person_name}\n"
            f"accuracy_answerable: {accuracy_answerable}; accuracy_unanswerable: {accuracy_unanswerable}\n "
            f"number_answerable: {self.number_answerable[benchmark_type]}; "
            f"number_unanswerable: {self.number_unanswerable[benchmark_type]} "
            f"number_all: {self.number_all[benchmark_type]}\n"
        )

        results = {
            "number_all": self.number_all[benchmark_type],
            "number_answerable": self.number_answerable[benchmark_type],
            "number_unanswerable": self.number_unanswerable[benchmark_type],
            "number_answerable_correct": self.correct_answerable[benchmark_type],
            "number_unanswerable_correct": self.correct_unanswerable[benchmark_type],
            "accuracy_answerable": accuracy_answerable,
            "accuracy_unanswerable": accuracy_unanswerable,
        }
        return results

    def show_results(self, calculate_mean_fun_name):
        re_non_re = self.show_accuracy("role_non_relation")
        re_re = self.show_accuracy("role_relation")

        # use name to get fun
        # function factory
        # mean = get_calculate_fun(calculate_mean_fun_name)(re_non_re, re_re)

        results = {
            "role_non_relation": re_non_re,
            "role_relation": re_re,
            "all_number": re_non_re["number_all"] + re_re["number_all"],
        }

        return results


def make_csv_file_models_single_person(
    prompt_kind,
    model_name_list,
    calculate_mean_fun_name,
    benchmark_version,
    profile_version,
    system_version,
    prompt_name,
    character_name,
):
    results_all = {}
    for model_name in model_name_list:
        sta1 = Statistic(
            person_name=character_name,
            model_name=model_name,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            benchmark_version=benchmark_version,
            profile_version=profile_version,
            system_version=system_version,
        )
        results = sta1.show_results(calculate_mean_fun_name)
        result_results = {
            "model": f"{model_name}",
            "mean": results["mean"],
            "basic_information_answerable": round(
                results["basic_information"]["accuracy_answerable"], 4
            ),
            "basic_information_unanswerable": round(
                results["basic_information"]["accuracy_unanswerable"], 4
            ),
            "roles_non_relation_answerable": round(
                results["role_non_relation"]["accuracy_answerable"], 4
            ),
            "roles_non_relation_unanswerable": round(
                results["role_non_relation"]["accuracy_unanswerable"], 4
            ),
            "roles_relation_answerable": round(
                results["role_relation"]["accuracy_answerable"], 4
            ),
            "roles_relation_unanswerable": round(
                results["role_relation"]["accuracy_unanswerable"], 4
            ),
        }
        results_all[model_name] = result_results

    path_ = f"{csv_root_path_coherence_task.format(character_name=character_name)}/{prompt_kind}_{prompt_name}_{system_version}/"
    if not os.path.exists(path_):
        os.makedirs(path_)

    path_person = f"{path_}/{calculate_mean_fun_name}.csv"
    with open(path_person, mode="w") as _file:
        _writer = csv.DictWriter(
            _file,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            fieldnames=csv_head_row_models_single_name,
        )
        _writer.writeheader()
        for model_name in model_name_list:
            _writer.writerow(results_all[model_name])


def make_csv_file_models_single_person_no_basic_information(
    prompt_kind,
    model_name_list,
    calculate_mean_fun_name,
    benchmark_version,
    profile_version,
    system_version,
    prompt_name,
    character_name,
):
    results_all = {}
    for model_name in model_name_list:
        sta1 = StatisticNoBasicInformation(
            person_name=character_name,
            model_name=model_name,
            prompt_kind=prompt_kind,
            prompt_name=prompt_name,
            benchmark_version=benchmark_version,
            profile_version=profile_version,
            system_version=system_version,
        )
        results = sta1.show_results(calculate_mean_fun_name)
        result_results = {
            "model": f"{model_name}",
            "roles_non_relation_answerable": round(
                results["role_non_relation"]["accuracy_answerable"], 4
            ),
            "roles_non_relation_unanswerable": round(
                results["role_non_relation"]["accuracy_unanswerable"], 4
            ),
            "roles_relation_answerable": round(
                results["role_relation"]["accuracy_answerable"], 4
            ),
            "roles_relation_unanswerable": round(
                results["role_relation"]["accuracy_unanswerable"], 4
            ),
        }
        results_all[model_name] = result_results

    path_ = f"{csv_root_path_coherence_task.format(character_name=character_name + '_' + profile_version)}/{prompt_kind}_{prompt_name}/"
    if not os.path.exists(path_):
        os.makedirs(path_)

    path_person = f"{path_}/{calculate_mean_fun_name}.csv"
    header = [
        "model",
        "roles_non_relation_answerable",
        "roles_non_relation_unanswerable",
        "roles_relation_answerable",
        "roles_relation_unanswerable",
    ]
    with open(path_person, mode="w") as _file:
        _writer = csv.DictWriter(
            _file,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            fieldnames=header,
        )
        _writer.writeheader()
        for model_name in model_name_list:
            _writer.writerow(results_all[model_name])


# used for ablation test
def make_csv_file_models_single_person_ablation(
    prompt_kind,
    model_name_list,
    calculate_mean_fun_name,
    ablation_kind,
    full_name_list,
    benchmark_version,
    profile_version,
    system_version,
    prompt_name,
):
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
                system_version=system_version,
            )
            results = sta1.show_results(calculate_mean_fun_name)
            result_results = {
                "model": model_name,
                "mean": results["mean"],
                "basic_information_answerable": results["basic_information"][
                    "accuracy_answerable"
                ],
                "basic_information_unanswerable": results["basic_information"][
                    "accuracy_unanswerable"
                ],
                "roles_non_relation_answerable": results["role_non_relation"][
                    "accuracy_answerable"
                ],
                "roles_non_relation_unanswerable": results["role_non_relation"][
                    "accuracy_unanswerable"
                ],
                "roles_relation_answerable": results["role_relation"][
                    "accuracy_answerable"
                ],
                "roles_relation_unanswerable": results["role_relation"][
                    "accuracy_unanswerable"
                ],
            }
            results_all[model_name] = result_results

        path_ = f"{csv_root_path_ablation.format(ablation_kind=ablation_kind)}/{prompt_kind}/{prompt_name}_{calculate_mean_fun_name}"
        if not os.path.exists(path_):
            os.makedirs(path_)

        path_person = f"{path_}/{person_name}.csv"
        with open(path_person, mode="w") as _file:
            _writer = csv.DictWriter(
                _file,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                fieldnames=csv_head_row_models_single_name,
            )
            _writer.writeheader()
            for model_name in model_name_list:
                _writer.writerow(results_all[model_name])


# used for test the effect of self consistency
def make_csv_file_models_single_person_self_consistency(
    prompt_kind,
    model_name_list,
    calculate_mean_fun_name,
    ablation_kind,
    full_name_list,
    benchmark_version,
    profile_version,
    system_version,
    prompt_name,
):
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
                system_version=system_version,
            )
            results = sta1.show_results(calculate_mean_fun_name)
            result_results = {
                "model": model_name,
                "mean": results["mean"],
                "basic_information_answerable": results["basic_information"][
                    "accuracy_answerable"
                ],
                "basic_information_unanswerable": results["basic_information"][
                    "accuracy_unanswerable"
                ],
                "roles_non_relation_answerable": results["role_non_relation"][
                    "accuracy_answerable"
                ],
                "roles_non_relation_unanswerable": results["role_non_relation"][
                    "accuracy_unanswerable"
                ],
                "roles_relation_answerable": results["role_relation"][
                    "accuracy_answerable"
                ],
                "roles_relation_unanswerable": results["role_relation"][
                    "accuracy_unanswerable"
                ],
            }
            results_all[model_name] = result_results

        path_ = f"{csv_root_path_ablation.format(ablation_kind=ablation_kind)}/{prompt_kind}/{prompt_name}_{calculate_mean_fun_name}"
        if not os.path.exists(path_):
            os.makedirs(path_)

        path_person = f"{path_}/{person_name}.csv"
        with open(path_person, mode="w") as _file:
            _writer = csv.DictWriter(
                _file,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                fieldnames=csv_head_row_models_single_name,
            )
            _writer.writeheader()
            for model_name in model_name_list:
                _writer.writerow(results_all[model_name])


# used for ablation test
def make_csv_file_models_all_people_ablation(
    prompt_kind,
    model_name_list,
    benchmark_version,
    profile_version,
    system_version,
    full_name_list,
    prompt_name,
    ablation_kind,
    calculate_mean_fun_name,
):
    """
    all the  names of people origin from the same person
    Args:
        prompt_kind:
        model_name_list:
        benchmark_version:
        profile_version:
        system_version:
        full_name_list:        prompt_name:
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
                system_version=system_version,
            )
            results = sta1.show_results(calculate_mean_fun_name)
            mean_list.append(results["mean"])

            if number_of_all_question is None:
                number_of_all_question = results["all_number"]

            """# make csv file of the model on the person for all kind of benchmark file type
            path_ = f"{csv_root_path.format(ablation_kind='age')}/{model_name}/"
            if not os.path.exists(path_):
                os.makedirs(path_)
            path_person= f"{path_}/{person_name}.csv"""

        # calculate the variance of the mean scores
        mean = np.mean(mean_list).item()
        var = np.std(mean_list, ddof=1).item()

        range_ = round((max(mean_list) - min(mean_list)) * number_of_all_question)

        # CoV
        cov = var / mean
        var = var
        mean = mean
        mean_dict[model_name] = [cov] + [mean, var] + [range_] + mean_list

    # make the dir for different prompt kind
    path_ = f"{csv_root_path_ablation.format(ablation_kind=ablation_kind)}/{prompt_kind}/{prompt_name}_{calculate_mean_fun_name}"
    if not os.path.exists(path_):
        os.makedirs(path_)

    # write file
    path_all = f"{path_}/all.csv"
    with open(path_all, mode="w") as _file:
        _writer = csv.writer(
            _file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        _writer.writerow(
            csv_head_row_models_names_mean
            + ["CoV", "mean", "variance", f"range(total:{number_of_all_question})"]
            + full_name_list
        )
        for model_name in model_name_list:
            _writer.writerow([model_name] + mean_dict[model_name])


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    benchmark_version = "benchmark_v2"
    profile_version = "profile_v1"
    system_version = "system_v1"

    prompt_name = "prompt1"
    prompt_kind = "few_shot"
    person_name = "homer"
    sta1 = Statistic(
        person_name=person_name,
        model_name=model_name,
        prompt_kind=prompt_kind,
        prompt_name=prompt_name,
        benchmark_version=benchmark_version,
        profile_version=profile_version,
        system_version=system_version,
    )
    sta1.show_results(calculate_mean_2)

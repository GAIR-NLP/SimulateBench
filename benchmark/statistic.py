from benchmark.benchmark_file_util import ROOT_PATH, load_json_file


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

    def show_results(self):
        self.show_accuracy("basic_information")
        self.show_accuracy("role_non_relation")
        self.show_accuracy("role_relation")


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

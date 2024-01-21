import json
import os


def load_json_file(path_):
    with open(path_, "r", encoding="utf-8") as f:
        return json.load(f)


def makedir_():
    raw_obj_path = "/home/yxiao2/pycharm/GPTMan/db/template/homer/examples_and_requirements/anthropomorphize.json"

    with open(raw_obj_path, "r") as f:
        raw_obj = json.load(f)

    import os

    raw_path = "/home/yxiao2/pycharm/GPTMan/db/template/{person_name}/examples_and_requirements/"
    for person_name in [
        "homer_1985",
        "homer_2000",
        "homer_2010",
        "homer_2015",
        "homer_2020",
        "homer_asian",
        "homer_african",
        "homer_Middle_Eastern",
        "homer_Native_American",
        "homer_Northern_European",
        "homer_Southern_American",
        "rachel",
        "walter_white",
    ]:
        tem_path = raw_path.format(person_name=person_name)
        if not os.path.exists(tem_path):
            os.makedirs(tem_path)
        with open(tem_path + "anthropomorphize.json", "w") as f:
            json.dump(raw_obj, f)


def load_file_list(path: str) -> list[str]:
    files = os.listdir(path)
    files = [f for f in files if os.path.isdir(os.path.join(path, f))]
    files.remove("template_questions")
    files.remove("galadriel")

    return files


def copy_files(person_names: str, file_type: str) -> None:
    import shutil

    for person_name in person_names:
        source = os.path.join(
            "/home/yxiao2/pycharm/GPTMan/db/benchmark",
            file_type,
            person_name,
            "profile_v1",
            "system_v1",
            "benchmark_v2",
            "questions.json",
        )
        print(source)

        target_path = os.path.join(
            "/home/yxiao2/pycharm/GPTMan/db/benchmark_only_QA",
            file_type,
            person_name,
        )
        if not os.path.exists(target_path):
            os.mkdir(
                os.path.join(
                    "/home/yxiao2/pycharm/GPTMan/db/benchmark_only_QA",
                    file_type,
                    person_name,
                )
            )
        dst = os.path.join(
            "/home/yxiao2/pycharm/GPTMan/db/benchmark_only_QA",
            file_type,
            person_name,
            "questions.json",
        )

        shutil.copyfile(source, dst)


if __name__ == "__main__":
    person_names = load_file_list(
        "/home/yxiao2/pycharm/GPTMan/db/benchmark/basic_information"
    )

    copy_files(person_names, "basic_information")
    copy_files(person_names, "role_non_relation")
    copy_files(person_names, "role_relation")

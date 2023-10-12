import json


def load_json_file(path_):
    with open(path_, 'r',encoding="utf-8") as f:
        return json.load(f)


def makedir_():
    raw_obj_path = "/home/yxiao2/pycharm/GPTMan/db/template/homer/examples_and_requirements/anthropomorphize.json"
    with open(raw_obj_path, 'r') as f:
        raw_obj = json.load(f)
    import os
    raw_path = "/home/yxiao2/pycharm/GPTMan/db/template/{person_name}/examples_and_requirements/"
    for person_name in [ 'homer_1985', "homer_2000", "homer_2010", "homer_2015", "homer_2020",
                        "homer_asian", "homer_african", "homer_Middle_Eastern", "homer_Native_American",
                        "homer_Northern_European", "homer_Southern_American", "rachel", "walter_white"]:

        tem_path = raw_path.format(person_name=person_name)
        if not os.path.exists(tem_path):
            os.makedirs(tem_path)
        with open(tem_path + "anthropomorphize.json", 'w') as f:
            json.dump(raw_obj, f)


if __name__ == "__main__":
    makedir_()

import os
import shutil

ROOT = "/home/yxiao2/pycharm/GPTMan/db/benchmark"
original = f"/home/yxiao2/pycharm/GPTMan/db/profile/homer"

entity_mapping = {
    "homer": "james",
    "Simpson": "Brown",
    "742 Evergreen Terrace, Springfield": "Garden Town, New York",
    "Homie": "jamie",
    "May 12, 1956": "May 26, 1956",
    "Springfield": "Garden Town",
    "Marge": "Jane",
    "Bart": "John",
    "Lisa": "Lily",
    "Maggie": "Ava",
}

education_full_name_list = [f"homer_invisible_james"]


def copy_homer():
    for name in education_full_name_list:
        target = f"/home/yxiao2/pycharm/GPTMan/db/profile/homer_invisible_james"
        if not os.path.exists(target):
            shutil.copytree(original, target)


if __name__ == "__main__":
    copy_homer()

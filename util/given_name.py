import os
import shutil
ROOT = "/home/yxiao2/pycharm/GPTMan/db/benchmark"
original = f"/home/yxiao2/pycharm/GPTMan/db/profile/homer"
given_name_list = ['Liam',"Noah","Oliver","Elijah","James"]


def copy_homer():
    for surname in given_name_list:
        target = f"{original}_given_name_{surname}"
        shutil.copytree(original, target)


if __name__ == "__main__":
    copy_homer()
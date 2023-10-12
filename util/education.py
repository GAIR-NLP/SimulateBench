import os
import shutil

ROOT = "/home/yxiao2/pycharm/GPTMan/db/benchmark"
original = f"/home/yxiao2/pycharm/GPTMan/db/profile/homer"
education_list = ['Elementary School', 'Middle School',
                  "Bachelor",
                  "Bachelor(MIT)", "Bachelor(NUS)",
                  "Bachelor(Columbia University)", "Bachelor(University of Toronto)",
                  "Bachelor(Zhejiang University)", "Bachelor(Lomonosov Moscow State University)",
                  "Bachelor(Rice University)", "Bachelor(Technical University of Denmark)",
                  "Bachelor(Purdue University)", "Bachelor(Al-Farabi Kazakh National University)",
                  "Bachelor(Indian Institute of Technology Delhi (IITD))", "Bachelor(University of Liverpool)",
                  "Master", "Doctor"]

education_full_name_list = [f"homer_education_{education}" for education in education_list]


def copy_homer():
    for education in education_list:
        target = f"{original}_education_{education}"
        if not os.path.exists(target):
            shutil.copytree(original, target)


if __name__ == "__main__":
    copy_homer()

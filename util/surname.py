import shutil

ROOT = "/home/yxiao2/pycharm/GPTMan/db/benchmark"
original = f"/home/yxiao2/pycharm/GPTMan/db/profile/homer"
surname_list = ['Begay', 'Clah', 'Keams', 'Bedonie', 'Nguyen', 'Tang', 'Patel', 'Tran', 'Chery', 'Fluellen',
                'Hyppolite', 'Mensah', 'Garcia', 'Guerrero', 'Aguirre', 'Hernandez', 'Jensen', 'Schmidt', 'Hansen',
                'Keller']


def copy_homer():
    for surname in surname_list:
        target = f"{original}_surname_{surname}"
        shutil.copytree(original, target)


if __name__ == "__main__":
    copy_homer()

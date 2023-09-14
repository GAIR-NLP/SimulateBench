import json


def load_json_file(path_):
    with open(path_, 'r') as f:
        return json.load(f)

import json

from GPTMan.config.config import settings_system


def delete_none(_dict):
    """Delete None values recursively from all of the dictionaries"""
    for key, value in list(_dict.items()):
        if isinstance(value, dict):
            delete_none(value)
        elif value is None:
            del _dict[key]
        elif isinstance(value, list):
            for v_i in value:
                if isinstance(v_i, dict):
                    delete_none(v_i)

    return _dict


def process(obj):
    """Process the object to be saved in the file"""
    for count in range(len(obj['basic_information']['habits'])):
        examples_obj = obj['basic_information']['habits'][count]

        examples_ob = examples_obj['examples']
        results = []
        for example in examples_ob:
            results.append(example['example'])
        examples_obj['examples']= results


    return obj


def delete_():
    path = settings_system['monica']['basic_information_path']

    with open(path, 'r') as f:
        data_obj = json.load(f)

    data_obj = process(data_obj)

    with open(path, 'w') as f:
        json.dump(data_obj, f, indent=4)




if __name__ == '__main__':
    delete_()

import json
from GPTMan.config.config import settings_system
from GPTMan.person.profile.base_data_class import load_person_files


def load_examples(person_name: str = "monica"):
    results = ''
    json_path = load_person_files(person_name)['examples_path']
    with open(json_path, 'r') as f:
        examples = json.load(f)
    for _example in examples['examples']:
        results += _example['requirement'] + '\n' + 'some examples are below:\n'

        if 'examples' not in _example:
            results += 'no examples\n'
            continue
        for example in _example['examples']:
            results += example + '\n'

    return results


if __name__ == '__main__':
    print(load_examples())
